"""
Trigger Pairing Validator - Static analysis for ask_user_trigger and pending_user_questions pairing.

This module validates that every `ask_user_trigger` assignment is paired with a corresponding
`pending_user_questions` assignment. This prevents bugs where users see empty prompts.

Usage:
    python tools/validate_trigger_pairing.py [--verbose] [--json] [filepath...]
    
Or import and use in tests:
    from tools.validate_trigger_pairing import validate_src_directory, ViolationType
    result = validate_src_directory()
"""

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


# =============================================================================
# Data Structures
# =============================================================================

class ViolationType(Enum):
    """Types of pairing violations."""
    UNPAIRED_IN_DICT = "unpaired_in_dict"
    UNPAIRED_SUBSCRIPT = "unpaired_subscript"
    SUSPICIOUS_VARIABLE = "suspicious_variable"
    EMPTY_QUESTIONS = "empty_questions"


@dataclass
class TriggerAssignment:
    """Represents a single ask_user_trigger assignment."""
    filepath: str
    line: int
    col: int
    value_type: Literal["none", "string_literal", "variable", "other"]
    value: Optional[str] = None  # The actual value if string literal or variable name
    in_dict_literal: bool = False
    paired_in_dict: bool = False  # If in dict, is pending_user_questions also there?
    function_name: Optional[str] = None


@dataclass
class Violation:
    """Represents a pairing violation."""
    type: ViolationType
    filepath: str
    line: int
    message: str
    function_name: Optional[str] = None
    severity: Literal["error", "warning"] = "error"


@dataclass
class QuestionsAssignment:
    """Represents a pending_user_questions assignment."""
    line: int
    function_name: Optional[str] = None
    is_empty_list: bool = False  # True if assigned to empty list []


@dataclass
class AnalysisResult:
    """Results from analyzing a single file."""
    filepath: str
    trigger_assignments: List[TriggerAssignment] = field(default_factory=list)
    questions_assignments: List[QuestionsAssignment] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    
    @property
    def questions_lines(self) -> Set[int]:
        """Backward compatibility: return set of all question lines."""
        return {q.line for q in self.questions_assignments}


@dataclass
class ValidationResult:
    """Results from validating one or more files."""
    violations: List[Violation] = field(default_factory=list)
    files_analyzed: int = 0
    trigger_count: int = 0  # Non-clearing trigger assignments
    questions_count: int = 0  # Non-empty questions assignments
    parse_errors: List[str] = field(default_factory=list)
    # For verbose mode
    all_trigger_assignments: List[TriggerAssignment] = field(default_factory=list)
    all_questions_assignments: List[QuestionsAssignment] = field(default_factory=list)


# =============================================================================
# AST Visitor
# =============================================================================

class TriggerPairingAnalyzer(ast.NodeVisitor):
    """
    AST visitor that analyzes ask_user_trigger and pending_user_questions pairing.
    
    Strategy:
    1. Collect all trigger assignments (in dicts and subscripts)
    2. Collect all lines where pending_user_questions is set
    3. Track function context for better error messages
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.result = AnalysisResult(filepath=filepath)
        self.current_function: Optional[str] = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track which function we're in."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async functions too."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Dict(self, node: ast.Dict):
        """
        Check dictionary literals for proper pairing.
        
        This catches patterns like:
            return {
                "ask_user_trigger": "foo",
                "pending_user_questions": [...],
            }
        """
        keys: Set[str] = set()
        trigger_value_type: Optional[str] = None
        trigger_value: Optional[str] = None
        questions_value: Optional[ast.AST] = None
        
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
                
                if key.value == "ask_user_trigger":
                    trigger_value_type, trigger_value = self._classify_value(value)
                elif key.value == "pending_user_questions":
                    questions_value = value
        
        if "ask_user_trigger" in keys:
            # Check if questions exist AND are non-empty
            has_questions = "pending_user_questions" in keys
            is_empty_questions = has_questions and self._is_empty_list(questions_value)
            
            assignment = TriggerAssignment(
                filepath=self.filepath,
                line=node.lineno,
                col=node.col_offset,
                value_type=trigger_value_type or "other",
                value=trigger_value,
                in_dict_literal=True,
                paired_in_dict=has_questions and not is_empty_questions,
                function_name=self.current_function,
            )
            self.result.trigger_assignments.append(assignment)
        
        if "pending_user_questions" in keys:
            is_empty = self._is_empty_list(questions_value)
            self.result.questions_assignments.append(QuestionsAssignment(
                line=node.lineno,
                function_name=self.current_function,
                is_empty_list=is_empty,
            ))
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """
        Check subscript assignments like result["ask_user_trigger"] = ...
        
        This catches patterns like:
            result["ask_user_trigger"] = "foo"
            result["pending_user_questions"] = [...]
        """
        for target in node.targets:
            key_name = self._get_subscript_key(target)
            
            if key_name == "ask_user_trigger":
                value_type, value = self._classify_value(node.value)
                assignment = TriggerAssignment(
                    filepath=self.filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    value_type=value_type,
                    value=value,
                    in_dict_literal=False,
                    paired_in_dict=False,
                    function_name=self.current_function,
                )
                self.result.trigger_assignments.append(assignment)
            
            elif key_name == "pending_user_questions":
                is_empty = self._is_empty_list(node.value)
                self.result.questions_assignments.append(QuestionsAssignment(
                    line=node.lineno,
                    function_name=self.current_function,
                    is_empty_list=is_empty,
                ))
        
        self.generic_visit(node)
    
    def _get_subscript_key(self, node: ast.AST) -> Optional[str]:
        """Extract the key from node["key"] pattern."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return node.slice.value
        return None
    
    def _classify_value(self, node: ast.AST) -> tuple:
        """
        Classify what kind of value is being assigned.
        
        Returns: (value_type, value_content)
        """
        if isinstance(node, ast.Constant):
            if node.value is None:
                return ("none", None)
            elif isinstance(node.value, str):
                return ("string_literal", node.value)
            else:
                return ("other", str(node.value))
        elif isinstance(node, ast.Name):
            return ("variable", node.id)
        elif isinstance(node, ast.Call):
            return ("other", None)
        else:
            return ("other", None)
    
    def _is_empty_list(self, node: Optional[ast.AST]) -> bool:
        """Check if node is an empty list literal []."""
        if node is None:
            return False
        if isinstance(node, ast.List) and len(node.elts) == 0:
            return True
        return False


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_file(filepath: Path) -> AnalysisResult:
    """Analyze a single Python file for trigger/questions assignments."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
        
        analyzer = TriggerPairingAnalyzer(str(filepath))
        analyzer.visit(tree)
        return analyzer.result
    except SyntaxError as e:
        result = AnalysisResult(filepath=str(filepath))
        result.parse_errors.append(f"Syntax error: {e}")
        return result
    except Exception as e:
        result = AnalysisResult(filepath=str(filepath))
        result.parse_errors.append(f"Error reading file: {e}")
        return result


def analyze_source(source: str, filepath: str = "<string>") -> AnalysisResult:
    """Analyze source code string for trigger/questions assignments."""
    try:
        tree = ast.parse(source)
        analyzer = TriggerPairingAnalyzer(filepath)
        analyzer.visit(tree)
        return analyzer.result
    except SyntaxError as e:
        result = AnalysisResult(filepath=filepath)
        result.parse_errors.append(f"Syntax error: {e}")
        return result


def check_pairing_violations(result: AnalysisResult) -> List[Violation]:
    """
    Check for pairing violations in analysis results.
    
    Rules:
    1. Dict literals: trigger must be paired with questions in same dict (non-empty)
    2. Subscript assignments: trigger must have questions within ±10 lines IN SAME FUNCTION
    3. Exception: value=None is clearing, doesn't need questions
    4. Exception: value=variable might be preserving existing trigger
    5. Empty questions lists are violations
    """
    violations: List[Violation] = []
    
    # Known preservation variable names
    PRESERVATION_VARS = {"ask_user_trigger", "trigger", "existing_trigger"}
    
    for assignment in result.trigger_assignments:
        # Rule 3: Clearing with None is always OK
        if assignment.value_type == "none":
            continue
        
        # Rule 4: Variable assignment might be preserving
        if assignment.value_type == "variable":
            if assignment.value in PRESERVATION_VARS:
                continue  # Likely preserving, OK
            # Flag for manual review (warning, not error)
            violations.append(Violation(
                type=ViolationType.SUSPICIOUS_VARIABLE,
                filepath=result.filepath,
                line=assignment.line,
                message=(
                    f"ask_user_trigger assigned from variable '{assignment.value}' - "
                    f"verify pending_user_questions is set"
                ),
                function_name=assignment.function_name,
                severity="warning",
            ))
            continue
        
        # Rule 1: Dict literals must be self-contained with non-empty questions
        if assignment.in_dict_literal:
            if not assignment.paired_in_dict:
                violations.append(Violation(
                    type=ViolationType.UNPAIRED_IN_DICT,
                    filepath=result.filepath,
                    line=assignment.line,
                    message=(
                        f"ask_user_trigger='{assignment.value}' in dict literal "
                        f"without pending_user_questions (or questions is empty [])"
                    ),
                    function_name=assignment.function_name,
                    severity="error",
                ))
            continue
        
        # Rule 2: Subscript assignments - check proximity WITHIN SAME FUNCTION
        # Get questions in the same function (or global scope if function_name is None)
        same_function_questions = [
            q for q in result.questions_assignments
            if q.function_name == assignment.function_name and not q.is_empty_list
        ]
        
        nearby_questions = any(
            abs(q.line - assignment.line) <= 10 
            for q in same_function_questions
        )
        
        if not nearby_questions:
            func_context = f" in function '{assignment.function_name}'" if assignment.function_name else ""
            violations.append(Violation(
                type=ViolationType.UNPAIRED_SUBSCRIPT,
                filepath=result.filepath,
                line=assignment.line,
                message=(
                    f"ask_user_trigger='{assignment.value}' set but no "
                    f"pending_user_questions found within 10 lines{func_context}"
                ),
                function_name=assignment.function_name,
                severity="error",
            ))
    
    # Rule 5: Check for empty questions lists (warning)
    for q in result.questions_assignments:
        if q.is_empty_list:
            violations.append(Violation(
                type=ViolationType.EMPTY_QUESTIONS,
                filepath=result.filepath,
                line=q.line,
                message="pending_user_questions is empty [] - users will see no prompt",
                function_name=q.function_name,
                severity="warning",
            ))
    
    return violations


# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_file(filepath: Path) -> ValidationResult:
    """Validate a single file for trigger/questions pairing."""
    result = ValidationResult()
    
    analysis = analyze_file(filepath)
    result.files_analyzed = 1
    result.parse_errors.extend(analysis.parse_errors)
    
    # Store all assignments for verbose mode
    result.all_trigger_assignments.extend(analysis.trigger_assignments)
    result.all_questions_assignments.extend(analysis.questions_assignments)
    
    # Count non-clearing trigger assignments
    for assignment in analysis.trigger_assignments:
        if assignment.value_type != "none":
            result.trigger_count += 1
    
    # Count non-empty questions assignments
    result.questions_count = len([q for q in analysis.questions_assignments if not q.is_empty_list])
    
    violations = check_pairing_violations(analysis)
    result.violations.extend(violations)
    
    return result


def validate_src_directory(src_dir: Optional[Path] = None) -> ValidationResult:
    """
    Validate all Python files in src/ directory.
    
    Args:
        src_dir: Directory to scan. Defaults to PROJECT_ROOT/src
        
    Returns:
        ValidationResult with all violations found
    """
    if src_dir is None:
        src_dir = SRC_DIR
    
    if not src_dir.exists():
        result = ValidationResult()
        result.parse_errors.append(f"Directory not found: {src_dir}")
        return result
    
    result = ValidationResult()
    
    for filepath in src_dir.rglob("*.py"):
        analysis = analyze_file(filepath)
        result.files_analyzed += 1
        result.parse_errors.extend(analysis.parse_errors)
        
        # Store all assignments for verbose mode
        result.all_trigger_assignments.extend(analysis.trigger_assignments)
        result.all_questions_assignments.extend(analysis.questions_assignments)
        
        # Count non-clearing trigger assignments
        for assignment in analysis.trigger_assignments:
            if assignment.value_type != "none":
                result.trigger_count += 1
        
        # Count non-empty questions assignments
        result.questions_count += len([q for q in analysis.questions_assignments if not q.is_empty_list])
        
        violations = check_pairing_violations(analysis)
        result.violations.extend(violations)
    
    return result


def validate_files(filepaths: List[Path]) -> ValidationResult:
    """Validate multiple specific files."""
    result = ValidationResult()
    
    for filepath in filepaths:
        if not filepath.exists():
            result.parse_errors.append(f"File not found: {filepath}")
            continue
        
        file_result = validate_file(filepath)
        result.files_analyzed += file_result.files_analyzed
        result.trigger_count += file_result.trigger_count
        result.questions_count += file_result.questions_count
        result.violations.extend(file_result.violations)
        result.parse_errors.extend(file_result.parse_errors)
    
    return result


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_violation(v: Violation) -> str:
    """Format a single violation for display."""
    severity = "ERROR" if v.severity == "error" else "WARNING"
    func_info = f" (function: {v.function_name})" if v.function_name else ""
    return f"{severity} {v.filepath}:{v.line}: {v.message}{func_info}"


def format_violations(violations: List[Violation]) -> str:
    """Format multiple violations for display."""
    if not violations:
        return "No violations found."
    
    lines = [f"Found {len(violations)} violation(s):\n"]
    for v in violations:
        lines.append(f"  • {format_violation(v)}")
    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for trigger pairing validation."""
    parser = argparse.ArgumentParser(
        description="Validate ask_user_trigger and pending_user_questions pairing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/validate_trigger_pairing.py              # Validate src/ directory
  python tools/validate_trigger_pairing.py --verbose    # Show all assignments
  python tools/validate_trigger_pairing.py --json       # JSON output for CI
  python tools/validate_trigger_pairing.py src/agents/execution.py  # Single file
        """
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to validate (default: all files in src/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all trigger assignments, not just violations",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--warnings-as-errors", "-W",
        action="store_true",
        help="Treat warnings as errors (affects exit code)",
    )
    
    args = parser.parse_args()
    
    # Run validation
    if args.files:
        filepaths = [Path(f) for f in args.files]
        result = validate_files(filepaths)
    else:
        result = validate_src_directory()
    
    # Separate errors and warnings
    errors = [v for v in result.violations if v.severity == "error"]
    warnings = [v for v in result.violations if v.severity == "warning"]
    
    # JSON output
    if args.json:
        output = {
            "files_analyzed": result.files_analyzed,
            "trigger_count": result.trigger_count,
            "questions_count": result.questions_count,
            "violations": [
                {
                    "type": v.type.value,
                    "filepath": v.filepath,
                    "line": v.line,
                    "message": v.message,
                    "function_name": v.function_name,
                    "severity": v.severity,
                }
                for v in result.violations
            ],
            "parse_errors": result.parse_errors,
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(f"\nTrigger Pairing Validation")
        print(f"{'=' * 50}")
        print(f"Files analyzed: {result.files_analyzed}")
        print(f"Trigger assignments (non-clearing): {result.trigger_count}")
        print(f"Questions assignments: {result.questions_count}")
        
        if result.parse_errors:
            print(f"\nParse errors:")
            for err in result.parse_errors:
                print(f"  • {err}")
        
        if args.verbose:
            print(f"\n📋 All Trigger Assignments ({len(result.all_trigger_assignments)}):")
            for t in result.all_trigger_assignments:
                status = "✓" if t.value_type == "none" or t.paired_in_dict else "?"
                func = f" [{t.function_name}]" if t.function_name else " [global]"
                print(f"  {status} {t.filepath}:{t.line}{func} = {t.value_type}:{t.value}")
            
            print(f"\n📋 All Questions Assignments ({len(result.all_questions_assignments)}):")
            for q in result.all_questions_assignments:
                status = "⚠️" if q.is_empty_list else "✓"
                func = f" [{q.function_name}]" if q.function_name else " [global]"
                empty_note = " (EMPTY)" if q.is_empty_list else ""
                print(f"  {status} Line {q.line}{func}{empty_note}")
        
        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for v in errors:
                print(f"  • {format_violation(v)}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for v in warnings:
                print(f"  • {format_violation(v)}")
        
        if not errors and not warnings:
            print("\n✅ No violations found!")
        elif not errors:
            print(f"\n✅ No errors (only {len(warnings)} warning(s))")
    
    # Exit code
    if errors:
        sys.exit(1)
    elif warnings and args.warnings_as_errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
