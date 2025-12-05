"""
Schema Access Validator - Whitelist-based validation for schema-backed variables.

This module validates that:
1. All access to schema-backed variables uses allowed patterns (whitelist)
2. All accessed fields exist in the corresponding JSON schema

Usage:
    python tools/validate_schema_access.py [--verbose]
    
Or import and use in tests:
    from tools.validate_schema_access import validate_agent_files
    violations = validate_agent_files()
"""

import ast
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Set, Dict, List, Optional, Any


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
AGENTS_DIR = PROJECT_ROOT / "src" / "agents"


# =============================================================================
# Configuration: Agent-to-Schema Mapping (Built Dynamically)
# =============================================================================

# Schema naming exceptions (agent_name -> schema_file)
# Most follow the pattern: {agent_name}_output_schema.json
# If all schemas follow convention, this dict should be empty
SCHEMA_NAME_OVERRIDES: Dict[str, str] = {
    # All schemas now follow the {agent_name}_output_schema.json convention
}


def _extract_agent_name_from_call(node: ast.Call) -> Optional[str]:
    """Extract agent_name keyword argument from call_agent_with_metrics call."""
    for keyword in node.keywords:
        if keyword.arg == "agent_name":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _find_enclosing_function(tree: ast.AST, target_lineno: int) -> Optional[str]:
    """Find the function name that contains the given line number."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if target line is within this function
            if hasattr(node, 'end_lineno'):
                if node.lineno <= target_lineno <= node.end_lineno:
                    return node.name
            else:
                # Fallback for Python < 3.8: assume function contains the line
                # if it starts before it
                if node.lineno <= target_lineno:
                    return node.name
    return None


def _derive_schema_filename(agent_name: str) -> str:
    """Derive schema filename from agent_name."""
    if agent_name in SCHEMA_NAME_OVERRIDES:
        return SCHEMA_NAME_OVERRIDES[agent_name]
    return f"{agent_name}_output_schema.json"


@dataclass
class AgentSchemaInfo:
    """Info about an agent's LLM output variable and schema."""
    variable_name: str  # The variable name used (e.g., "agent_output", "result", "llm_response")
    schema_file: str  # The schema file (e.g., "planner_output_schema.json")


def build_agent_schema_mapping() -> Dict[str, Dict[str, AgentSchemaInfo]]:
    """
    Dynamically build agent-to-schema mapping by scanning agent files.
    
    Scans for `<var> = call_agent_with_metrics(agent_name="xxx", ...)` patterns
    and extracts BOTH the variable name AND the agent_name to determine the schema.
    
    Returns:
        Dict mapping relative file paths to {function_name: AgentSchemaInfo}
    """
    mapping: Dict[str, Dict[str, AgentSchemaInfo]] = {}
    warnings: List[str] = []
    
    for agent_file in AGENTS_DIR.rglob("*.py"):
        try:
            source = agent_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError) as e:
            warnings.append(f"Could not parse {agent_file}: {e}")
            continue
        
        rel_path = str(agent_file.relative_to(AGENTS_DIR))
        
        for node in ast.walk(tree):
            # Look for: <any_var> = call_agent_with_metrics(...)
            if isinstance(node, ast.Assign):
                # Check if single assignment to a simple variable
                if (len(node.targets) == 1 and 
                    isinstance(node.targets[0], ast.Name)):
                    
                    var_name = node.targets[0].id
                    
                    # Check if RHS is call_agent_with_metrics(...)
                    if (isinstance(node.value, ast.Call) and
                        isinstance(node.value.func, ast.Name) and
                        node.value.func.id == "call_agent_with_metrics"):
                        
                        agent_name = _extract_agent_name_from_call(node.value)
                        if agent_name:
                            func_name = _find_enclosing_function(tree, node.lineno)
                            if func_name:
                                schema_file = _derive_schema_filename(agent_name)
                                
                                # Verify schema exists
                                schema_path = SCHEMAS_DIR / schema_file
                                if not schema_path.exists():
                                    warnings.append(
                                        f"Schema not found: {schema_file} "
                                        f"(from {rel_path}:{func_name}, agent_name='{agent_name}')"
                                    )
                                    continue
                                
                                if rel_path not in mapping:
                                    mapping[rel_path] = {}
                                mapping[rel_path][func_name] = AgentSchemaInfo(
                                    variable_name=var_name,
                                    schema_file=schema_file,
                                )
    
    # Print warnings if any
    if warnings:
        for warning in warnings:
            print(f"  [warning] {warning}", file=sys.stderr)
    
    return mapping


# Build the mapping at module load time
AGENT_OUTPUT_SCHEMA_MAPPING: Dict[str, Dict[str, AgentSchemaInfo]] = build_agent_schema_mapping()

# Maps derived variable names to their schema location
# Format: "schema_file.json#json/pointer/path"
DERIVED_VAR_SCHEMAS: Dict[str, str] = {
    "adaptation": "prompt_adaptor_output_schema.json#/properties/prompt_modifications/items",
}


def get_base_tracked_variables() -> Dict[str, "TrackedVariable"]:
    """
    Build the base set of tracked variables dynamically.
    
    Extracts all unique variable names from AGENT_OUTPUT_SCHEMA_MAPPING
    (which detects the actual variable names used in each function, not
    just hardcoded "agent_output").
    
    Returns:
        Dict mapping variable names to TrackedVariable with schema info.
    """
    result: Dict[str, TrackedVariable] = {}
    
    # Extract all unique variable names from the schema mapping
    # These are the actual variable names used for LLM output (e.g., "agent_output", "result", etc.)
    for file_funcs in AGENT_OUTPUT_SCHEMA_MAPPING.values():
        for func_name, schema_info in file_funcs.items():
            var_name = schema_info.variable_name
            if var_name not in result:
                # Schema is resolved dynamically per-function, so we use placeholder
                result[var_name] = TrackedVariable(
                    name=var_name,
                    schema_file="",  # Resolved dynamically per function
                    json_pointer="",
                )
    
    # Add derived variables from DERIVED_VAR_SCHEMAS
    for var_name, schema_pointer in DERIVED_VAR_SCHEMAS.items():
        # Parse "schema_file.json#/pointer/path"
        if "#" in schema_pointer:
            schema_file, pointer = schema_pointer.split("#", 1)
        else:
            schema_file = schema_pointer
            pointer = ""
        
        result[var_name] = TrackedVariable(
            name=var_name,
            schema_file=schema_file,
            json_pointer=pointer,
        )
    
    return result




# =============================================================================
# Data Classes
# =============================================================================

class ViolationType(Enum):
    """Types of validation violations."""
    PATTERN_NOT_WHITELISTED = "pattern_not_whitelisted"
    FIELD_NOT_IN_SCHEMA = "field_not_in_schema"
    DYNAMIC_KEY = "dynamic_key"


@dataclass
class FieldAccess:
    """Represents a field access in code."""
    line: int
    col: int
    variable: str
    field: Optional[str]  # None if dynamic
    access_type: str  # "get", "get_with_default", "in_check"
    is_static: bool
    in_function: Optional[str]
    code_snippet: str


@dataclass
class Violation:
    """A validation violation."""
    file: str
    line: int
    col: int
    type: ViolationType
    message: str
    code_snippet: str
    field: Optional[str] = None
    variable: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a file or set of files."""
    violations: List[Violation] = field(default_factory=list)
    field_accesses: List[FieldAccess] = field(default_factory=list)
    files_scanned: int = 0
    
    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one."""
        self.violations.extend(other.violations)
        self.field_accesses.extend(other.field_accesses)
        self.files_scanned += other.files_scanned
        return self


@dataclass
class TrackedVariable:
    """
    Represents a variable being tracked for schema validation.
    
    Tracks the schema context so we can validate nested field accesses.
    E.g., if agent_output.get("geometry") is assigned to geometry_spec,
    we track geometry_spec with pointer "/properties/geometry".
    """
    name: str
    schema_file: str
    json_pointer: str  # e.g., "", "/properties/geometry", "/properties/stages/items"
    source_var: Optional[str] = None  # Parent variable this was derived from
    source_field: Optional[str] = None  # Field accessed on parent to get this


# =============================================================================
# Schema Loading
# =============================================================================

def load_schema(schema_name: str) -> dict:
    """Load a JSON schema file."""
    schema_path = SCHEMAS_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with open(schema_path) as f:
        return json.load(f)


def extract_schema_fields(schema: dict, prefix: str = "") -> Set[str]:
    """Extract all field names from a JSON schema (top-level and nested)."""
    fields = set()
    
    if "properties" in schema:
        for prop_name, prop_def in schema["properties"].items():
            full_name = f"{prefix}.{prop_name}" if prefix else prop_name
            fields.add(prop_name)  # Add just the field name
            fields.add(full_name)  # Add full path too
            if isinstance(prop_def, dict):
                fields.update(extract_schema_fields(prop_def, full_name))
    
    if "items" in schema and isinstance(schema["items"], dict):
        fields.update(extract_schema_fields(schema["items"], prefix))
    
    return fields


def get_schema_fields_for_pointer(schema_file: str, pointer: str = "") -> Set[str]:
    """Get schema fields, optionally following a JSON pointer."""
    schema = load_schema(schema_file)
    
    if pointer:
        # Follow JSON pointer (e.g., "/properties/prompt_modifications/items")
        parts = pointer.strip("/").split("/")
        current = schema
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Invalid JSON pointer: {pointer} in {schema_file}")
        schema = current
    
    return extract_schema_fields(schema)


def get_nested_schema_info(schema_file: str, parent_pointer: str, field: str) -> Optional[tuple]:
    """
    Given a parent schema pointer and field name, return info about the nested schema.
    
    Args:
        schema_file: The schema file name
        parent_pointer: JSON pointer to parent (e.g., "" or "/properties/geometry")
        field: Field being accessed on parent (e.g., "geometry" or "cell_size")
    
    Returns:
        Tuple of (new_pointer, valid_fields, is_array_items) or None if field not found.
        - new_pointer: JSON pointer to the nested schema
        - valid_fields: Set of valid field names at that level
        - is_array_items: True if this is accessing items of an array
    """
    # If schema_file is empty, we can't resolve the schema
    if not schema_file:
        return None
    
    try:
        schema = load_schema(schema_file)
        
        # Navigate to parent location
        current = schema
        if parent_pointer:
            parts = parent_pointer.strip("/").split("/")
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        # Check if field exists in properties
        if "properties" in current and field in current["properties"]:
            field_schema = current["properties"][field]
            new_pointer = f"{parent_pointer}/properties/{field}"
            
            # Check if this field is an array with items
            if field_schema.get("type") == "array" and "items" in field_schema:
                # Return the items schema for array fields
                items_pointer = f"{new_pointer}/items"
                items_fields = extract_schema_fields(field_schema["items"])
                return (items_pointer, items_fields, True)
            else:
                # Return the field's own properties
                nested_fields = extract_schema_fields(field_schema)
                return (new_pointer, nested_fields, False)
        
        return None
    except (FileNotFoundError, ValueError):
        return None


def resolve_tracked_var_field(tracked_var: "TrackedVariable", field: str) -> Optional["TrackedVariable"]:
    """
    Create a new TrackedVariable for a field access on an existing tracked variable.
    
    E.g., if tracked_var is agent_output with pointer "", and field is "geometry",
    returns a new TrackedVariable with pointer "/properties/geometry".
    """
    # If schema_file is empty, we can't resolve nested fields
    # (This happens for agent_output which has per-function schema resolution)
    if not tracked_var.schema_file:
        return None
    
    nested_info = get_nested_schema_info(tracked_var.schema_file, tracked_var.json_pointer, field)
    if nested_info is None:
        return None
    
    new_pointer, _, is_array_items = nested_info
    return TrackedVariable(
        name="",  # Will be set by caller
        schema_file=tracked_var.schema_file,
        json_pointer=new_pointer,
        source_var=tracked_var.name,
        source_field=field,
    )


# =============================================================================
# AST-Based Field Access Extraction
# =============================================================================

class FieldAccessVisitor(ast.NodeVisitor):
    """
    AST visitor that extracts all field accesses on tracked variables.
    
    Supports:
    - Direct access: agent_output.get("field")
    - Nested access: x = agent_output.get("geometry"); x.get("cell_size")
    - For-loop iteration: for item in agent_output.get("stages")
    - Variable aliasing: out = agent_output; out.get("field")
    """
    
    def __init__(self, source_lines: List[str], tracked_vars: Dict[str, TrackedVariable], 
                 rel_file_path: str = ""):
        self.source_lines = source_lines
        # Base tracked variables (e.g., agent_output)
        self.base_tracked_vars = tracked_vars.copy()
        # All tracked variables including derived ones (scoped to current function)
        self.tracked_vars: Dict[str, TrackedVariable] = tracked_vars.copy()
        # Function-scoped derived variables (cleared on function exit)
        self.function_derived_vars: Dict[str, TrackedVariable] = {}
        
        self.accesses: List[FieldAccess] = []
        self.violations: List[Violation] = []
        self.current_function: Optional[str] = None
        self.file_path: str = ""
        # Relative file path (for schema lookup by function)
        self.rel_file_path: str = rel_file_path
        # Track derived variables for verbose output
        self.derived_vars_log: List[str] = []
    
    def get_snippet(self, lineno: int) -> str:
        """Get the source code snippet for a line."""
        if 0 < lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""
    
    def _enter_function(self, func_name: str):
        """Enter a function scope."""
        self.current_function = func_name
        # Clear function-scoped derived vars from previous function
        self.function_derived_vars.clear()
        # Reset tracked_vars to base + any global derived vars
        self.tracked_vars = self.base_tracked_vars.copy()
        
        # Inject schema info for LLM output variable in this function
        # This enables derived variable tracking (e.g., geometry_spec = agent_output.get("geometry"))
        if self.rel_file_path:
            schema_info = get_schema_info_for_function(self.rel_file_path, func_name)
            if schema_info:
                var_name = schema_info.variable_name
                if var_name in self.tracked_vars:
                    # Update the tracked variable with the actual schema for this function
                    self.tracked_vars[var_name] = TrackedVariable(
                        name=var_name,
                        schema_file=schema_info.schema_file,
                        json_pointer="",  # Top-level
                    )
    
    def _exit_function(self, old_func: Optional[str]):
        """Exit a function scope."""
        # Remove function-scoped derived vars
        for var_name in self.function_derived_vars:
            self.tracked_vars.pop(var_name, None)
        self.function_derived_vars.clear()
        self.current_function = old_func
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function context and manage scope."""
        old_func = self.current_function
        self._enter_function(node.name)
        self.generic_visit(node)
        self._exit_function(old_func)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track async function context and manage scope."""
        old_func = self.current_function
        self._enter_function(node.name)
        self.generic_visit(node)
        self._exit_function(old_func)
    
    def _get_var_name(self, node: ast.expr) -> Optional[str]:
        """Get variable name from a node, if it's a simple name."""
        if isinstance(node, ast.Name):
            return node.id
        return None
    
    def _is_tracked_var(self, node: ast.expr) -> bool:
        """Check if node is a tracked variable."""
        var_name = self._get_var_name(node)
        return var_name in self.tracked_vars if var_name else False
    
    def _get_static_string(self, node: ast.expr) -> Optional[str]:
        """Get static string value from node, or None if dynamic."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None
    
    def _is_get_call(self, node: ast.expr) -> bool:
        """Check if node is a .get() call."""
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == "get")
    
    def _is_array_tracked_var(self, var_name: str) -> bool:
        """
        Check if a tracked variable is an array/list type.
        
        Array-typed tracked variables have json_pointer ending with /items
        (meaning they came from iterating over or extracting an array field).
        """
        tracked = self.tracked_vars.get(var_name)
        if tracked and tracked.json_pointer:
            return tracked.json_pointer.endswith("/items")
        return False
    
    def _extract_get_call_info(self, node: ast.Call) -> Optional[tuple]:
        """
        Extract info from a .get() call.
        Returns (var_name, field) or None if not applicable.
        """
        if not self._is_get_call(node):
            return None
        
        var_name = self._get_var_name(node.func.value)
        if var_name not in self.tracked_vars:
            return None
        
        if node.args:
            field = self._get_static_string(node.args[0])
            if field:
                return (var_name, field)
        
        return None
    
    def _add_derived_var(self, new_var_name: str, parent_var: str, field: str, lineno: int):
        """
        Add a derived tracked variable from a .get() assignment.
        E.g., geometry_spec = agent_output.get("geometry", {})
        
        Only tracks variables if the field is an object type (has properties).
        Scalar fields (string, number, etc.) are not tracked since they don't
        have nested fields to validate.
        """
        parent_tracked = self.tracked_vars.get(parent_var)
        if not parent_tracked:
            return
        
        # Skip if parent has no schema (e.g., agent_output with dynamic schema)
        if not parent_tracked.schema_file:
            return
        
        # Check if the field is an object type with properties
        nested_info = get_nested_schema_info(parent_tracked.schema_file, parent_tracked.json_pointer, field)
        if not nested_info:
            return
        
        new_pointer, valid_fields, is_array_items = nested_info
        
        # Only track if the field has nested properties (is an object)
        # Don't track scalar fields like strings, numbers, etc.
        if not valid_fields:
            return
        
        # Resolve the nested schema
        derived = resolve_tracked_var_field(parent_tracked, field)
        if derived:
            derived.name = new_var_name
            self.tracked_vars[new_var_name] = derived
            self.function_derived_vars[new_var_name] = derived
            self.derived_vars_log.append(
                f"  {new_var_name} <- {parent_var}.get('{field}') "
                f"[pointer: {derived.json_pointer}]"
            )
    
    def _add_loop_var(self, loop_var_name: str, parent_var: str, field: str, lineno: int):
        """
        Add a tracked variable from a for-loop iteration.
        E.g., for stage in agent_output.get("stages")
        """
        parent_tracked = self.tracked_vars.get(parent_var)
        if not parent_tracked:
            return
        
        # Get the items schema for the array field
        nested_info = get_nested_schema_info(parent_tracked.schema_file, parent_tracked.json_pointer, field)
        if nested_info:
            new_pointer, _, is_array = nested_info
            if is_array:
                # The loop variable gets the items schema
                derived = TrackedVariable(
                    name=loop_var_name,
                    schema_file=parent_tracked.schema_file,
                    json_pointer=new_pointer,  # Already points to /items
                    source_var=parent_var,
                    source_field=field,
                )
                self.tracked_vars[loop_var_name] = derived
                self.function_derived_vars[loop_var_name] = derived
                self.derived_vars_log.append(
                    f"  {loop_var_name} <- for loop over {parent_var}.get('{field}') "
                    f"[items pointer: {new_pointer}]"
                )
    
    def _add_alias(self, new_name: str, original_name: str, lineno: int):
        """
        Add an alias for an existing tracked variable.
        E.g., out = agent_output
        """
        original_tracked = self.tracked_vars.get(original_name)
        if original_tracked:
            alias = TrackedVariable(
                name=new_name,
                schema_file=original_tracked.schema_file,
                json_pointer=original_tracked.json_pointer,
                source_var=original_name,
                source_field=None,
            )
            self.tracked_vars[new_name] = alias
            self.function_derived_vars[new_name] = alias
            self.derived_vars_log.append(
                f"  {new_name} <- alias of {original_name}"
            )
    
    def visit_Assign(self, node: ast.Assign):
        """
        Handle assignments to detect variable propagation.
        
        Patterns detected:
        - x = tracked_var.get("field", {})  -> derive x from tracked_var
        - x = tracked_var                   -> alias
        """
        # Only handle simple single assignments
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            self.generic_visit(node)
            return
        
        target_name = node.targets[0].id
        
        # Check for: x = tracked_var.get("field", ...)
        get_info = self._extract_get_call_info(node.value)
        if get_info:
            parent_var, field = get_info
            self._add_derived_var(target_name, parent_var, field, node.lineno)
        
        # Check for: x = tracked_var (direct alias)
        elif isinstance(node.value, ast.Name) and node.value.id in self.tracked_vars:
            self._add_alias(target_name, node.value.id, node.lineno)
        
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Handle method calls like obj.get("field")."""
        # Check for tracked_var.get("field") pattern
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            var_name = self._get_var_name(node.func.value)
            if var_name in self.tracked_vars:
                snippet = self.get_snippet(node.lineno)
                
                if node.args:
                    field = self._get_static_string(node.args[0])
                    is_static = field is not None
                    has_default = len(node.args) > 1 or len(node.keywords) > 0
                    
                    access = FieldAccess(
                        line=node.lineno,
                        col=node.col_offset,
                        variable=var_name,
                        field=field,
                        access_type="get_with_default" if has_default else "get",
                        is_static=is_static,
                        in_function=self.current_function,
                        code_snippet=snippet,
                    )
                    self.accesses.append(access)
                    
                    # If dynamic key, it's a violation
                    if not is_static:
                        self.violations.append(Violation(
                            file=self.file_path,
                            line=node.lineno,
                            col=node.col_offset,
                            type=ViolationType.DYNAMIC_KEY,
                            message=f"Dynamic key in .get() call on '{var_name}'. Use static string literal.",
                            code_snippet=snippet,
                            variable=var_name,
                        ))
        
        # Check for disallowed method calls: .keys(), .values(), .items(), .pop(), etc.
        if isinstance(node.func, ast.Attribute):
            var_name = self._get_var_name(node.func.value)
            if var_name in self.tracked_vars:
                method = node.func.attr
                # Only .get() is allowed
                if method != "get":
                    snippet = self.get_snippet(node.lineno)
                    self.violations.append(Violation(
                        file=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        type=ViolationType.PATTERN_NOT_WHITELISTED,
                        message=f"Method '.{method}()' not in whitelist for '{var_name}'. Only .get() is allowed.",
                        code_snippet=snippet,
                        variable=var_name,
                    ))
        
        # Check for getattr(tracked_var, ...)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2:
                var_name = self._get_var_name(node.args[0])
                if var_name in self.tracked_vars:
                    snippet = self.get_snippet(node.lineno)
                    self.violations.append(Violation(
                        file=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        type=ViolationType.PATTERN_NOT_WHITELISTED,
                        message=f"getattr() not in whitelist for '{var_name}'. Use .get() with static key.",
                        code_snippet=snippet,
                        variable=var_name,
                    ))
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node: ast.Subscript):
        """Handle subscript access like obj["field"] or list[0].
        
        Only flags READ access (ctx=Load). WRITE access (ctx=Store) is allowed
        since dict["key"] = value is the standard way to set values.
        
        Also allows numeric index access on array-typed tracked variables,
        since list[0] is a valid pattern for accessing list elements.
        """
        var_name = self._get_var_name(node.value)
        if var_name in self.tracked_vars:
            # Only flag READ access, not WRITE (Store) or DELETE (Del)
            if isinstance(node.ctx, ast.Load):
                # Allow numeric index access on array-typed variables (e.g., adaptations[0])
                is_numeric_index = isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)
                if is_numeric_index and self._is_array_tracked_var(var_name):
                    # This is valid list index access, don't flag
                    pass
                else:
                    snippet = self.get_snippet(node.lineno)
                    field = self._get_static_string(node.slice)
                    
                    # Subscript READ access is NOT in whitelist (only .get() is allowed for reads)
                    self.violations.append(Violation(
                        file=self.file_path,
                        line=node.lineno,
                        col=node.col_offset,
                        type=ViolationType.PATTERN_NOT_WHITELISTED,
                        message=f"Subscript read access obj[key] not in whitelist for '{var_name}'. Use .get() instead.",
                        code_snippet=snippet,
                        field=field,
                        variable=var_name,
                    ))
        
        self.generic_visit(node)
    
    def visit_Compare(self, node: ast.Compare):
        """Handle 'in' checks like "field" in obj."""
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                var_name = self._get_var_name(comparator)
                if var_name in self.tracked_vars:
                    snippet = self.get_snippet(node.lineno)
                    field = self._get_static_string(node.left)
                    is_static = field is not None
                    
                    access = FieldAccess(
                        line=node.lineno,
                        col=node.col_offset,
                        variable=var_name,
                        field=field,
                        access_type="in_check",
                        is_static=is_static,
                        in_function=self.current_function,
                        code_snippet=snippet,
                    )
                    self.accesses.append(access)
                    
                    if not is_static:
                        self.violations.append(Violation(
                            file=self.file_path,
                            line=node.lineno,
                            col=node.col_offset,
                            type=ViolationType.DYNAMIC_KEY,
                            message=f"Dynamic key in 'in' check for '{var_name}'. Use static string literal.",
                            code_snippet=snippet,
                            variable=var_name,
                        ))
        
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """
        Handle for-loop iteration.
        
        - Direct iteration over dict-typed tracked var: flag as violation
        - Direct iteration over array-typed tracked var: allowed and track loop var
        - Iteration over tracked_var.get("array_field"): track loop variable
        """
        # Direct iteration: for x in tracked_var
        var_name = self._get_var_name(node.iter)
        if var_name in self.tracked_vars:
            if self._is_array_tracked_var(var_name):
                # Allowed: iteration over array-typed variable
                # Track the loop variable with the same schema as the array items
                if isinstance(node.target, ast.Name):
                    loop_var_name = node.target.id
                    tracked = self.tracked_vars[var_name]
                    # The array variable already has /items pointer, use it for loop var
                    derived = TrackedVariable(
                        name=loop_var_name,
                        schema_file=tracked.schema_file,
                        json_pointer=tracked.json_pointer,  # Same as array's items pointer
                        source_var=var_name,
                        source_field=None,
                    )
                    self.tracked_vars[loop_var_name] = derived
                    self.function_derived_vars[loop_var_name] = derived
                    self.derived_vars_log.append(
                        f"  {loop_var_name} <- for loop over {var_name} "
                        f"[pointer: {tracked.json_pointer}]"
                    )
            else:
                # Not allowed: direct iteration over dict-typed variable
                snippet = self.get_snippet(node.lineno)
                self.violations.append(Violation(
                    file=self.file_path,
                    line=node.lineno,
                    col=node.col_offset,
                    type=ViolationType.PATTERN_NOT_WHITELISTED,
                    message=f"Direct iteration over '{var_name}' not in whitelist. Access fields explicitly.",
                    code_snippet=snippet,
                    variable=var_name,
                ))
        
        # Iteration over .get() result: for item in tracked_var.get("items")
        # Track the loop variable with the items schema
        if isinstance(node.target, ast.Name):
            get_info = self._extract_get_call_info(node.iter)
            if get_info:
                parent_var, field = get_info
                self._add_loop_var(node.target.id, parent_var, field, node.lineno)
        
        self.generic_visit(node)


def extract_field_accesses(
    source: str,
    file_path: str,
    tracked_vars: Optional[Dict[str, TrackedVariable]] = None,
    tracked_vars_set: Optional[Set[str]] = None,  # Legacy support
    agents_rel_path: str = "",  # Path relative to AGENTS_DIR for schema lookup
) -> tuple:
    """
    Extract all field accesses from source code.
    
    Args:
        source: The source code to parse
        file_path: Path to the file (for error reporting)
        tracked_vars: Dict of tracked variables with schema info
        tracked_vars_set: Legacy Set[str] input (deprecated)
        agents_rel_path: Path relative to AGENTS_DIR for per-function schema lookup
    
    Returns:
        Tuple of (ValidationResult, derived_vars_log) where derived_vars_log
        contains info about dynamically tracked variables.
    """
    # Handle legacy Set[str] input
    if tracked_vars is None:
        if tracked_vars_set is not None:
            # Convert legacy set to dict
            tracked_vars = {}
            for var_name in tracked_vars_set:
                tracked_vars[var_name] = TrackedVariable(
                    name=var_name,
                    schema_file="",  # Unknown schema for legacy input
                    json_pointer="",
                )
        else:
            tracked_vars = get_base_tracked_variables()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ValidationResult(violations=[
            Violation(
                file=file_path,
                line=e.lineno or 0,
                col=e.offset or 0,
                type=ViolationType.PATTERN_NOT_WHITELISTED,
                message=f"Syntax error: {e.msg}",
                code_snippet="",
            )
        ]), []
    
    lines = source.splitlines()
    visitor = FieldAccessVisitor(lines, tracked_vars, agents_rel_path)
    visitor.file_path = file_path
    visitor.visit(tree)
    
    return ValidationResult(
        violations=visitor.violations,
        field_accesses=visitor.accesses,
        files_scanned=1,
    ), visitor.derived_vars_log


# =============================================================================
# Schema Validation
# =============================================================================

def validate_fields_against_schema(
    accesses: List[FieldAccess],
    schema_fields: Set[str],
    file_path: str,
) -> List[Violation]:
    """Validate that accessed fields exist in schema."""
    violations = []
    
    for access in accesses:
        if access.is_static and access.field:
            if access.field not in schema_fields:
                violations.append(Violation(
                    file=file_path,
                    line=access.line,
                    col=access.col,
                    type=ViolationType.FIELD_NOT_IN_SCHEMA,
                    message=f"Field '{access.field}' accessed on '{access.variable}' not found in schema.",
                    code_snippet=access.code_snippet,
                    field=access.field,
                    variable=access.variable,
                ))
    
    return violations


# =============================================================================
# Main Validation Functions
# =============================================================================

def validate_file(
    file_path: Path,
    schema_fields: Optional[Set[str]] = None,
    tracked_vars: Optional[Dict[str, TrackedVariable]] = None,
    tracked_vars_set: Optional[Set[str]] = None,  # Legacy support
) -> tuple:
    """
    Validate a single file.
    
    Returns:
        Tuple of (ValidationResult, derived_vars_log)
    """
    if tracked_vars is None and tracked_vars_set is None:
        tracked_vars = get_base_tracked_variables()
    
    with open(file_path) as f:
        source = f.read()
    
    rel_path = str(file_path.relative_to(PROJECT_ROOT) if file_path.is_relative_to(PROJECT_ROOT) else file_path)
    
    # Compute agents-relative path for per-function schema lookup
    agents_rel_path = ""
    try:
        if file_path.is_relative_to(AGENTS_DIR):
            agents_rel_path = str(file_path.relative_to(AGENTS_DIR))
    except (ValueError, TypeError):
        pass
    
    # Extract field accesses and whitelist violations
    result, derived_vars_log = extract_field_accesses(
        source, rel_path, tracked_vars, tracked_vars_set, agents_rel_path
    )
    
    # If schema fields provided, validate against schema
    if schema_fields:
        schema_violations = validate_fields_against_schema(
            result.field_accesses,
            schema_fields,
            rel_path,
        )
        result.violations.extend(schema_violations)
    
    return result, derived_vars_log


def get_schema_info_for_function(file_name: str, function_name: str) -> Optional[AgentSchemaInfo]:
    """Get schema info for a specific function in a file."""
    # Normalize file name
    for key in AGENT_OUTPUT_SCHEMA_MAPPING:
        if file_name.endswith(key):
            func_schemas = AGENT_OUTPUT_SCHEMA_MAPPING[key]
            # Direct match on function name
            if function_name in func_schemas:
                return func_schemas[function_name]
            # Try partial match (for flexibility)
            for func_name, schema_info in func_schemas.items():
                if func_name in function_name.lower() or function_name.lower() in func_name:
                    return schema_info
    return None


def get_schema_fields_for_function(file_name: str, function_name: str) -> Optional[Set[str]]:
    """Get schema fields for a specific function based on file and function context."""
    schema_info = get_schema_info_for_function(file_name, function_name)
    if schema_info:
        try:
            return extract_schema_fields(load_schema(schema_info.schema_file))
        except FileNotFoundError:
            return None
    return None


def validate_agent_files(verbose: bool = False) -> tuple:
    """
    Validate all agent files for schema consistency.
    
    Returns:
        Tuple of (ValidationResult, all_derived_vars_log)
    """
    result = ValidationResult()
    all_derived_vars_log: List[str] = []
    
    # Find all Python files in agents directory
    agent_files = list(AGENTS_DIR.rglob("*.py"))
    
    for file_path in agent_files:
        if file_path.name.startswith("__"):
            continue
        
        if verbose:
            print(f"Scanning: {file_path.relative_to(PROJECT_ROOT)}")
        
        file_result, derived_vars_log = validate_file(file_path)
        result.merge(file_result)
        all_derived_vars_log.extend(derived_vars_log)
        
        # Validate LLM output variable accesses against their schemas
        rel_path = str(file_path.relative_to(AGENTS_DIR))
        for access in file_result.field_accesses:
            if access.in_function and access.is_static:
                # Get schema info for this function (includes variable name used)
                schema_info = get_schema_info_for_function(rel_path, access.in_function)
                if schema_info:
                    # Check if this access is on the LLM output variable for this function
                    if access.variable == schema_info.variable_name:
                        try:
                            schema_fields = extract_schema_fields(load_schema(schema_info.schema_file))
                            if access.field and access.field not in schema_fields:
                                result.violations.append(Violation(
                                    file=str(file_path.relative_to(PROJECT_ROOT)),
                                    line=access.line,
                                    col=access.col,
                                    type=ViolationType.FIELD_NOT_IN_SCHEMA,
                                    message=f"Field '{access.field}' not in schema for agent in function '{access.in_function}'.",
                                    code_snippet=access.code_snippet,
                                    field=access.field,
                                    variable=access.variable,
                                ))
                        except FileNotFoundError:
                            pass  # Schema file not found, skip validation
    
    return result, all_derived_vars_log


def validate_prompts_module() -> tuple:
    """
    Validate the prompts module for adaptation field access.
    
    Returns:
        Tuple of (ValidationResult, derived_vars_log)
    """
    prompts_file = PROJECT_ROOT / "src" / "prompts.py"
    
    if not prompts_file.exists():
        return ValidationResult(), []
    
    # Get schema fields for adaptation items
    try:
        adaptation_fields = get_schema_fields_for_pointer(
            "prompt_adaptor_output_schema.json",
            "/properties/prompt_modifications/items"
        )
    except (FileNotFoundError, ValueError):
        adaptation_fields = set()
    
    # Build tracked vars for adaptation
    adaptation_tracked = {
        "adaptation": TrackedVariable(
            name="adaptation",
            schema_file="prompt_adaptor_output_schema.json",
            json_pointer="/properties/prompt_modifications/items",
        )
    }
    
    return validate_file(prompts_file, adaptation_fields, adaptation_tracked)


# =============================================================================
# Schema Naming Convention Validation
# =============================================================================

def validate_schema_naming_conventions() -> List[str]:
    """
    Validate that all agent output schemas follow the naming convention.
    
    Expected patterns:
    - Agent output schemas: {agent_name}_output_schema.json
    - Other schemas: {name}_schema.json (for non-agent schemas like plan, metrics, etc.)
    
    Returns:
        List of warning messages for schemas that don't follow conventions.
    """
    warnings = []
    
    # Known non-agent schemas that don't need _output_ in the name
    NON_AGENT_SCHEMAS = {
        "plan_schema.json",
        "metrics_schema.json", 
        "progress_schema.json",
        "assumptions_schema.json",
        "prompt_adaptations_schema.json",
    }
    
    for schema_file in SCHEMAS_DIR.glob("*.json"):
        name = schema_file.name
        
        # Skip non-agent schemas
        if name in NON_AGENT_SCHEMAS:
            continue
        
        # Check if it follows _output_schema.json convention
        if not name.endswith("_output_schema.json"):
            warnings.append(
                f"Schema '{name}' doesn't follow naming convention. "
                f"Expected: *_output_schema.json for agent schemas, or *_schema.json for data schemas."
            )
    
    return warnings


def print_tracking_summary(verbose: bool = False):
    """Print summary of what the validator is tracking."""
    print("\n--- Tracking Configuration ---")
    print(f"  Tracked variables: {sorted(get_base_tracked_variables().keys())}")
    
    print(f"\n  Agent schema mappings (dynamically discovered):")
    for file_path, func_schemas in sorted(AGENT_OUTPUT_SCHEMA_MAPPING.items()):
        print(f"    {file_path}:")
        for func_name, schema_info in sorted(func_schemas.items()):
            print(f"      {func_name}: {schema_info.variable_name} -> {schema_info.schema_file}")
    
    print(f"\n  Derived variable schemas:")
    for var_name, schema_pointer in sorted(DERIVED_VAR_SCHEMAS.items()):
        print(f"    {var_name} -> {schema_pointer}")
    
    if SCHEMA_NAME_OVERRIDES:
        print(f"\n  Schema naming overrides:")
        for agent_name, schema_file in sorted(SCHEMA_NAME_OVERRIDES.items()):
            print(f"    {agent_name} -> {schema_file}")
    else:
        print(f"\n  Schema naming overrides: None (all follow convention)")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run validation and print results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate schema access patterns in code")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Schema Access Validator")
    print("=" * 70)
    
    # Validate schema naming conventions first
    naming_warnings = validate_schema_naming_conventions()
    if naming_warnings:
        print("\n⚠️  Schema Naming Convention Warnings:")
        for warning in naming_warnings:
            print(f"    - {warning}")
    
    # Print tracking summary if verbose
    if args.verbose:
        print_tracking_summary(args.verbose)
    
    # Validate agent files
    print("\nValidating agent files...")
    result, agent_derived_vars = validate_agent_files(verbose=args.verbose)
    
    # Validate prompts module
    print("\nValidating prompts module...")
    prompts_result, prompts_derived_vars = validate_prompts_module()
    result.merge(prompts_result)
    
    # Print derived variable tracking if verbose
    all_derived_vars = agent_derived_vars + prompts_derived_vars
    if args.verbose and all_derived_vars:
        print("\n--- Derived Variables Tracked ---")
        for log_entry in all_derived_vars:
            print(log_entry)
    
    # Print field access summary if verbose
    if args.verbose:
        print("\n--- Field Access Summary ---")
        # Group by file
        accesses_by_file: Dict[str, List[FieldAccess]] = {}
        for access in result.field_accesses:
            file_key = f"{access.variable}@{access.in_function or 'global'}"
            if file_key not in accesses_by_file:
                accesses_by_file[file_key] = []
            accesses_by_file[file_key].append(access)
        
        for key, accesses in sorted(accesses_by_file.items()):
            fields = sorted(set(a.field for a in accesses if a.field))
            print(f"  {key}: {fields}")
    
    # Print results
    print("\n" + "=" * 70)
    print(f"Files scanned: {result.files_scanned}")
    print(f"Field accesses found: {len(result.field_accesses)}")
    print(f"Violations found: {len(result.violations)}")
    if naming_warnings:
        print(f"Schema naming warnings: {len(naming_warnings)}")
    print("=" * 70)
    
    if result.violations:
        print("\nViolations:")
        for v in result.violations:
            print(f"\n  [{v.type.value}] {v.file}:{v.line}")
            print(f"    {v.message}")
            if v.code_snippet:
                print(f"    Code: {v.code_snippet}")
        
        sys.exit(1)
    else:
        print("\n✅ No violations found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
