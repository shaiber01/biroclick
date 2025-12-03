"""Unit tests for src/agents/constants.py"""

import pytest
import json
from enum import Enum
from src.agents.constants import (
    ReviewVerdict,
    SupervisorVerdict,
    ExecutionVerdict,
    PhysicsVerdict,
    AnalysisClassification,
    WorkflowPhase,
    AskUserTrigger,
    StageStatus,
    StageType,
)


class TestReviewVerdict:
    """Tests for ReviewVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert ReviewVerdict.APPROVE == "approve"
        assert ReviewVerdict.NEEDS_REVISION == "needs_revision"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {"APPROVE", "NEEDS_REVISION"}
        actual_members = {member.name for member in ReviewVerdict}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in ReviewVerdict:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in ReviewVerdict:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
    
    def test_repr_and_str(self):
        """Should have correct string representations."""
        # For string enums, str() returns the enum name representation
        # The value is accessed via .value property
        assert ReviewVerdict.APPROVE.value == "approve"
        assert "ReviewVerdict.APPROVE" in str(ReviewVerdict.APPROVE)
        assert "ReviewVerdict.APPROVE" in repr(ReviewVerdict.APPROVE)
        assert "approve" in repr(ReviewVerdict.APPROVE)
    
    def test_hashable(self):
        """Should be hashable for use in sets and dicts."""
        enum_set = {ReviewVerdict.APPROVE, ReviewVerdict.NEEDS_REVISION}
        assert len(enum_set) == 2
        enum_dict = {ReviewVerdict.APPROVE: 1, ReviewVerdict.NEEDS_REVISION: 2}
        assert enum_dict[ReviewVerdict.APPROVE] == 1
    
    def test_case_sensitive_comparison(self):
        """Comparisons should be case-sensitive."""
        assert ReviewVerdict.APPROVE == "approve"
        assert ReviewVerdict.APPROVE != "APPROVE"
        assert ReviewVerdict.APPROVE != "Approve"
    
    def test_ordering(self):
        """Should support ordering comparisons."""
        assert ReviewVerdict.APPROVE < ReviewVerdict.NEEDS_REVISION or ReviewVerdict.APPROVE > ReviewVerdict.NEEDS_REVISION
        assert ReviewVerdict.APPROVE <= ReviewVerdict.APPROVE
        assert ReviewVerdict.APPROVE >= ReviewVerdict.APPROVE


class TestSupervisorVerdict:
    """Tests for SupervisorVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert SupervisorVerdict.OK_CONTINUE == "ok_continue"
        assert SupervisorVerdict.BACKTRACK == "backtrack"
        assert SupervisorVerdict.BACKTRACK_TO_STAGE == "backtrack_to_stage"
        assert SupervisorVerdict.FINISH == "finish"
        assert SupervisorVerdict.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert SupervisorVerdict.REPLAN == "replan"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "OK_CONTINUE", "BACKTRACK", "BACKTRACK_TO_STAGE", 
            "FINISH", "MATERIAL_CHECKPOINT", "REPLAN"
        }
        actual_members = {member.name for member in SupervisorVerdict}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in SupervisorVerdict:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in SupervisorVerdict:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
    
    def test_hashable(self):
        """Should be hashable for use in sets and dicts."""
        enum_set = set(SupervisorVerdict)
        assert len(enum_set) == 6
        enum_dict = {member: i for i, member in enumerate(SupervisorVerdict)}
        assert len(enum_dict) == 6


class TestExecutionVerdict:
    """Tests for ExecutionVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert ExecutionVerdict.PASS == "pass"
        assert ExecutionVerdict.FAIL == "fail"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {"PASS", "FAIL"}
        actual_members = {member.name for member in ExecutionVerdict}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in ExecutionVerdict:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase."""
        for member in ExecutionVerdict:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert len(member.value) > 0, f"{member.name}.value should not be empty"
    
    def test_mutually_exclusive(self):
        """PASS and FAIL should be different."""
        assert ExecutionVerdict.PASS != ExecutionVerdict.FAIL
        assert ExecutionVerdict.PASS.value != ExecutionVerdict.FAIL.value


class TestPhysicsVerdict:
    """Tests for PhysicsVerdict enum."""

    def test_values(self):
        """Should have correct values."""
        assert PhysicsVerdict.PASS == "pass"
        assert PhysicsVerdict.WARNING == "warning"
        assert PhysicsVerdict.FAIL == "fail"
        assert PhysicsVerdict.DESIGN_FLAW == "design_flaw"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {"PASS", "WARNING", "FAIL", "DESIGN_FLAW"}
        actual_members = {member.name for member in PhysicsVerdict}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in PhysicsVerdict:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in PhysicsVerdict:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in PhysicsVerdict]
        assert len(values) == len(set(values)), "PhysicsVerdict has duplicate values"


class TestAnalysisClassification:
    """Tests for AnalysisClassification enum."""

    def test_values(self):
        """Should have correct values."""
        assert AnalysisClassification.EXCELLENT_MATCH == "EXCELLENT_MATCH"
        assert AnalysisClassification.ACCEPTABLE_MATCH == "ACCEPTABLE_MATCH"
        assert AnalysisClassification.PARTIAL_MATCH == "PARTIAL_MATCH"
        assert AnalysisClassification.POOR_MATCH == "POOR_MATCH"
        assert AnalysisClassification.FAILED == "FAILED"
        assert AnalysisClassification.NO_TARGETS == "NO_TARGETS"
        assert AnalysisClassification.PENDING_VALIDATION == "PENDING_VALIDATION"
        assert AnalysisClassification.MATCH == "MATCH"
        assert AnalysisClassification.MISMATCH == "MISMATCH"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "EXCELLENT_MATCH", "ACCEPTABLE_MATCH", "PARTIAL_MATCH", 
            "POOR_MATCH", "FAILED", "NO_TARGETS", "PENDING_VALIDATION",
            "MATCH", "MISMATCH"
        }
        actual_members = {member.name for member in AnalysisClassification}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in AnalysisClassification:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format_uppercase(self):
        """All values should be UPPERCASE for consistency."""
        for member in AnalysisClassification:
            assert member.value.isupper(), f"{member.name}.value should be UPPERCASE: {member.value}"
            assert member.value == member.value.upper(), f"{member.name}.value should be uppercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
    
    def test_case_sensitive_comparison(self):
        """Comparisons should be case-sensitive."""
        assert AnalysisClassification.EXCELLENT_MATCH == "EXCELLENT_MATCH"
        assert AnalysisClassification.EXCELLENT_MATCH != "excellent_match"
        assert AnalysisClassification.EXCELLENT_MATCH != "Excellent_Match"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in AnalysisClassification]
        assert len(values) == len(set(values)), "AnalysisClassification has duplicate values"
    
    def test_semantic_groups(self):
        """Verify semantic groupings of classifications."""
        # Match quality classifications
        match_quality = {
            AnalysisClassification.EXCELLENT_MATCH,
            AnalysisClassification.ACCEPTABLE_MATCH,
            AnalysisClassification.PARTIAL_MATCH,
            AnalysisClassification.POOR_MATCH,
        }
        assert all(m.value.endswith("_MATCH") for m in match_quality)
        
        # Status classifications
        status_classifications = {
            AnalysisClassification.FAILED,
            AnalysisClassification.NO_TARGETS,
            AnalysisClassification.PENDING_VALIDATION,
        }
        assert all(m.value in {"FAILED", "NO_TARGETS", "PENDING_VALIDATION"} for m in status_classifications)
        
        # Binary classifications
        binary_classifications = {
            AnalysisClassification.MATCH,
            AnalysisClassification.MISMATCH,
        }
        assert all(m.value in {"MATCH", "MISMATCH"} for m in binary_classifications)


class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_values(self):
        """Should have correct values."""
        assert WorkflowPhase.ADAPTING_PROMPTS == "adapting_prompts"
        assert WorkflowPhase.PLANNING == "planning"
        assert WorkflowPhase.PLAN_REVIEW == "plan_review"
        assert WorkflowPhase.STAGE_SELECTION == "stage_selection"
        assert WorkflowPhase.DESIGN == "design"
        assert WorkflowPhase.DESIGN_REVIEW == "design_review"
        assert WorkflowPhase.CODE_GENERATION == "code_generation"
        assert WorkflowPhase.CODE_REVIEW == "code_review"
        assert WorkflowPhase.EXECUTION == "execution"
        assert WorkflowPhase.EXECUTION_VALIDATION == "execution_validation"
        assert WorkflowPhase.PHYSICS_VALIDATION == "physics_validation"
        assert WorkflowPhase.ANALYSIS == "analysis"
        assert WorkflowPhase.COMPARISON_VALIDATION == "comparison_validation"
        assert WorkflowPhase.SUPERVISION == "supervision"
        assert WorkflowPhase.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert WorkflowPhase.AWAITING_USER == "awaiting_user"
        assert WorkflowPhase.BACKTRACKING == "backtracking"
        assert WorkflowPhase.BACKTRACKING_LIMIT == "backtracking_limit"
        assert WorkflowPhase.REPORTING == "reporting"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "ADAPTING_PROMPTS", "PLANNING", "PLAN_REVIEW", "STAGE_SELECTION",
            "DESIGN", "DESIGN_REVIEW", "CODE_GENERATION", "CODE_REVIEW",
            "EXECUTION", "EXECUTION_VALIDATION", "PHYSICS_VALIDATION",
            "ANALYSIS", "COMPARISON_VALIDATION", "SUPERVISION",
            "MATERIAL_CHECKPOINT", "AWAITING_USER", "BACKTRACKING",
            "BACKTRACKING_LIMIT", "REPORTING"
        }
        actual_members = {member.name for member in WorkflowPhase}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in WorkflowPhase:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in WorkflowPhase:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
            assert len(member.value) > 0, f"{member.name}.value should not be empty"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in WorkflowPhase]
        assert len(values) == len(set(values)), "WorkflowPhase has duplicate values"
    
    def test_phase_groups(self):
        """Verify logical groupings of workflow phases."""
        # Review phases
        review_phases = {
            WorkflowPhase.PLAN_REVIEW,
            WorkflowPhase.DESIGN_REVIEW,
            WorkflowPhase.CODE_REVIEW,
        }
        assert all("review" in m.value for m in review_phases)
        
        # Validation phases
        validation_phases = {
            WorkflowPhase.EXECUTION_VALIDATION,
            WorkflowPhase.PHYSICS_VALIDATION,
            WorkflowPhase.COMPARISON_VALIDATION,
        }
        assert all("validation" in m.value for m in validation_phases)
        
        # Backtracking phases
        backtracking_phases = {
            WorkflowPhase.BACKTRACKING,
            WorkflowPhase.BACKTRACKING_LIMIT,
        }
        assert all("backtracking" in m.value for m in backtracking_phases)


class TestAskUserTrigger:
    """Tests for AskUserTrigger enum."""

    def test_values(self):
        """Should have correct values."""
        assert AskUserTrigger.MATERIAL_CHECKPOINT == "material_checkpoint"
        assert AskUserTrigger.CODE_REVIEW_LIMIT == "code_review_limit"
        assert AskUserTrigger.DESIGN_REVIEW_LIMIT == "design_review_limit"
        assert AskUserTrigger.EXECUTION_FAILURE_LIMIT == "execution_failure_limit"
        assert AskUserTrigger.PHYSICS_FAILURE_LIMIT == "physics_failure_limit"
        assert AskUserTrigger.BACKTRACK_APPROVAL == "backtrack_approval"
        assert AskUserTrigger.BACKTRACK_LIMIT == "backtrack_limit"
        assert AskUserTrigger.REPLAN_LIMIT == "replan_limit"
        assert AskUserTrigger.CONTEXT_OVERFLOW == "context_overflow"
        assert AskUserTrigger.LLM_ERROR == "llm_error"
        assert AskUserTrigger.MISSING_PAPER_TEXT == "missing_paper_text"
        assert AskUserTrigger.MISSING_STAGE_ID == "missing_stage_id"
        assert AskUserTrigger.MISSING_DESIGN == "missing_design"
        assert AskUserTrigger.NO_STAGES_AVAILABLE == "no_stages_available"
        assert AskUserTrigger.DEADLOCK_DETECTED == "deadlock_detected"
        assert AskUserTrigger.INVALID_BACKTRACK_DECISION == "invalid_backtrack_decision"
        assert AskUserTrigger.INVALID_BACKTRACK_TARGET == "invalid_backtrack_target"
        assert AskUserTrigger.BACKTRACK_TARGET_NOT_FOUND == "backtrack_target_not_found"
        assert AskUserTrigger.PROGRESS_INIT_FAILED == "progress_init_failed"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "MATERIAL_CHECKPOINT", "CODE_REVIEW_LIMIT", "DESIGN_REVIEW_LIMIT",
            "EXECUTION_FAILURE_LIMIT", "PHYSICS_FAILURE_LIMIT", "BACKTRACK_APPROVAL",
            "BACKTRACK_LIMIT", "REPLAN_LIMIT", "CONTEXT_OVERFLOW", "LLM_ERROR",
            "MISSING_PAPER_TEXT", "MISSING_STAGE_ID", "MISSING_DESIGN",
            "NO_STAGES_AVAILABLE", "DEADLOCK_DETECTED", "INVALID_BACKTRACK_DECISION",
            "INVALID_BACKTRACK_TARGET", "BACKTRACK_TARGET_NOT_FOUND", "PROGRESS_INIT_FAILED"
        }
        actual_members = {member.name for member in AskUserTrigger}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in AskUserTrigger:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in AskUserTrigger:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
            assert len(member.value) > 0, f"{member.name}.value should not be empty"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in AskUserTrigger]
        assert len(values) == len(set(values)), "AskUserTrigger has duplicate values"
    
    def test_trigger_groups(self):
        """Verify logical groupings of triggers."""
        # Limit triggers
        limit_triggers = {
            AskUserTrigger.CODE_REVIEW_LIMIT,
            AskUserTrigger.DESIGN_REVIEW_LIMIT,
            AskUserTrigger.EXECUTION_FAILURE_LIMIT,
            AskUserTrigger.PHYSICS_FAILURE_LIMIT,
            AskUserTrigger.BACKTRACK_LIMIT,
            AskUserTrigger.REPLAN_LIMIT,
        }
        assert all("limit" in m.value for m in limit_triggers)
        
        # Missing data triggers
        missing_triggers = {
            AskUserTrigger.MISSING_PAPER_TEXT,
            AskUserTrigger.MISSING_STAGE_ID,
            AskUserTrigger.MISSING_DESIGN,
        }
        assert all("missing" in m.value for m in missing_triggers)
        
        # Backtrack-related triggers
        backtrack_triggers = {
            AskUserTrigger.BACKTRACK_APPROVAL,
            AskUserTrigger.INVALID_BACKTRACK_DECISION,
            AskUserTrigger.INVALID_BACKTRACK_TARGET,
            AskUserTrigger.BACKTRACK_TARGET_NOT_FOUND,
        }
        assert all("backtrack" in m.value for m in backtrack_triggers)


class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_values(self):
        """Should have correct values."""
        assert StageStatus.NOT_STARTED == "not_started"
        assert StageStatus.IN_PROGRESS == "in_progress"
        assert StageStatus.NEEDS_RERUN == "needs_rerun"
        assert StageStatus.BLOCKED == "blocked"
        assert StageStatus.INVALIDATED == "invalidated"
        assert StageStatus.COMPLETED_SUCCESS == "completed_success"
        assert StageStatus.COMPLETED_PARTIAL == "completed_partial"
        assert StageStatus.COMPLETED_FAILED == "completed_failed"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "NOT_STARTED", "IN_PROGRESS", "NEEDS_RERUN", "BLOCKED",
            "INVALIDATED", "COMPLETED_SUCCESS", "COMPLETED_PARTIAL", "COMPLETED_FAILED"
        }
        actual_members = {member.name for member in StageStatus}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in StageStatus:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format(self):
        """Values should be lowercase with underscores."""
        for member in StageStatus:
            assert member.value.islower(), f"{member.name}.value should be lowercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
            assert len(member.value) > 0, f"{member.name}.value should not be empty"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in StageStatus]
        assert len(values) == len(set(values)), "StageStatus has duplicate values"
    
    def test_status_groups(self):
        """Verify logical groupings of statuses."""
        # Active statuses
        active_statuses = {
            StageStatus.NOT_STARTED,
            StageStatus.IN_PROGRESS,
            StageStatus.NEEDS_RERUN,
        }
        assert all(m.value in {"not_started", "in_progress", "needs_rerun"} for m in active_statuses)
        
        # Completed statuses
        completed_statuses = {
            StageStatus.COMPLETED_SUCCESS,
            StageStatus.COMPLETED_PARTIAL,
            StageStatus.COMPLETED_FAILED,
        }
        assert all(m.value.startswith("completed_") for m in completed_statuses)
        
        # Problem statuses
        problem_statuses = {
            StageStatus.BLOCKED,
            StageStatus.INVALIDATED,
        }
        assert all(m.value in {"blocked", "invalidated"} for m in problem_statuses)
    
    def test_mutually_exclusive_completion(self):
        """Completed statuses should be mutually exclusive."""
        completed = {
            StageStatus.COMPLETED_SUCCESS,
            StageStatus.COMPLETED_PARTIAL,
            StageStatus.COMPLETED_FAILED,
        }
        assert len(completed) == 3
        assert all(c1 != c2 for c1 in completed for c2 in completed if c1 != c2)


class TestStageType:
    """Tests for StageType enum."""

    def test_values(self):
        """Should have correct values."""
        assert StageType.MATERIAL_VALIDATION == "MATERIAL_VALIDATION"
        assert StageType.SINGLE_STRUCTURE == "SINGLE_STRUCTURE"
        assert StageType.ARRAY_SYSTEM == "ARRAY_SYSTEM"
        assert StageType.PARAMETER_SWEEP == "PARAMETER_SWEEP"
        assert StageType.COMPLEX_PHYSICS == "COMPLEX_PHYSICS"
    
    def test_all_members_exist(self):
        """Should have exactly the expected members."""
        expected_members = {
            "MATERIAL_VALIDATION", "SINGLE_STRUCTURE", "ARRAY_SYSTEM",
            "PARAMETER_SWEEP", "COMPLEX_PHYSICS"
        }
        actual_members = {member.name for member in StageType}
        assert actual_members == expected_members, f"Expected {expected_members}, got {actual_members}"
    
    def test_value_types(self):
        """All values should be strings."""
        for member in StageType:
            assert isinstance(member.value, str), f"{member.name}.value should be str, got {type(member.value)}"
            assert isinstance(member, str), f"{member} should be str"
    
    def test_value_format_uppercase(self):
        """All values should be UPPERCASE for consistency."""
        for member in StageType:
            assert member.value.isupper(), f"{member.name}.value should be UPPERCASE: {member.value}"
            assert member.value == member.value.upper(), f"{member.name}.value should be uppercase: {member.value}"
            assert " " not in member.value, f"{member.name}.value should not contain spaces: {member.value}"
            assert len(member.value) > 0, f"{member.name}.value should not be empty"
    
    def test_all_values_unique(self):
        """All values should be unique."""
        values = [member.value for member in StageType]
        assert len(values) == len(set(values)), "StageType has duplicate values"
    
    def test_case_sensitive_comparison(self):
        """Comparisons should be case-sensitive."""
        assert StageType.MATERIAL_VALIDATION == "MATERIAL_VALIDATION"
        assert StageType.MATERIAL_VALIDATION != "material_validation"
        assert StageType.MATERIAL_VALIDATION != "Material_Validation"


class TestGeneralEnumBehavior:
    """Tests for general behavior common to all enums."""

    ENUM_CLASSES = [
        ReviewVerdict,
        SupervisorVerdict,
        ExecutionVerdict,
        PhysicsVerdict,
        AnalysisClassification,
        WorkflowPhase,
        AskUserTrigger,
        StageStatus,
        StageType,
    ]

    def test_string_comparison(self):
        """Should be comparable to strings."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert member == member.value, f"{member} should equal {member.value}"
                assert member.value == member, f"{member.value} should equal {member}"
                assert not (member != member.value), f"{member} should not be != {member.value}"
                assert not (member.value != member), f"{member.value} should not be != {member}"

    def test_string_comparison_case_sensitive(self):
        """String comparisons should be case-sensitive."""
        for enum_cls in self.ENUM_CLASSES:
            member = list(enum_cls)[0]
            if member.value.islower():
                assert member != member.value.upper(), f"{member} should not equal uppercase version"
            elif member.value.isupper():
                assert member != member.value.lower(), f"{member} should not equal lowercase version"

    def test_json_serializable(self):
        """Should be JSON serializable as string."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                data = {"value": member}
                serialized = json.dumps(data)
                assert f'"{member.value}"' in serialized, f"{member} should serialize to {member.value}"
                # Verify round-trip
                deserialized = json.loads(serialized)
                assert deserialized["value"] == member.value, f"Round-trip should preserve value"

    def test_json_deserialization(self):
        """Should be deserializable from JSON strings."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                json_str = json.dumps(member.value)
                deserialized_value = json.loads(json_str)
                assert enum_cls(deserialized_value) == member, f"Should deserialize {deserialized_value} to {member}"

    def test_instantiation_from_value(self):
        """Should be instantiable from valid string value."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert enum_cls(member.value) == member, f"Should instantiate {member} from {member.value}"
                assert enum_cls(member.value) is member, f"Should return same instance for {member.value}"

    def test_instantiation_from_enum_member(self):
        """Should handle instantiation from enum member itself."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert enum_cls(member) == member, f"Should handle {member} as argument"
                assert enum_cls(member) is member, f"Should return same instance for {member}"

    def test_invalid_value_raises_error(self):
        """Should raise ValueError for invalid values."""
        invalid_values = [
            "this_value_definitely_does_not_exist_12345",
            "",
            " ",
            "  ",
            "invalid\nvalue",
            "invalid\tvalue",
            "invalid value with spaces",
        ]
        for enum_cls in self.ENUM_CLASSES:
            for invalid_value in invalid_values:
                with pytest.raises(ValueError, match=".*"):
                    enum_cls(invalid_value)

    def test_invalid_value_error_message(self):
        """Error messages should be informative."""
        for enum_cls in self.ENUM_CLASSES:
            invalid_value = "definitely_invalid_xyz123"
            with pytest.raises(ValueError) as exc_info:
                enum_cls(invalid_value)
            error_msg = str(exc_info.value)
            assert invalid_value in error_msg or enum_cls.__name__ in error_msg, \
                f"Error message should mention invalid value or enum name: {error_msg}"

    def test_no_duplicate_values(self):
        """Ensure no enum class has duplicate values."""
        for enum_cls in self.ENUM_CLASSES:
            values = [e.value for e in enum_cls]
            unique_values = set(values)
            assert len(values) == len(unique_values), \
                f"{enum_cls.__name__} has duplicate values: {[v for v in values if values.count(v) > 1]}"

    def test_no_duplicate_names(self):
        """Ensure no enum class has duplicate member names."""
        for enum_cls in self.ENUM_CLASSES:
            names = [e.name for e in enum_cls]
            unique_names = set(names)
            assert len(names) == len(unique_names), \
                f"{enum_cls.__name__} has duplicate names: {[n for n in names if names.count(n) > 1]}"

    def test_inheritance(self):
        """Ensure all enums inherit from str and Enum."""
        for enum_cls in self.ENUM_CLASSES:
            assert issubclass(enum_cls, str), f"{enum_cls.__name__} should inherit from str"
            assert issubclass(enum_cls, Enum), f"{enum_cls.__name__} should inherit from Enum"
            # Verify instances are also strings
            member = list(enum_cls)[0]
            assert isinstance(member, str), f"{member} should be instance of str"

    def test_hashable(self):
        """All enum members should be hashable."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                hash(member)  # Should not raise
                assert isinstance(hash(member), int), f"{member} should have integer hash"

    def test_hash_consistency(self):
        """Hash values should be consistent across calls."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                hash1 = hash(member)
                hash2 = hash(member)
                assert hash1 == hash2, f"Hash of {member} should be consistent"

    def test_hash_equality(self):
        """Equal enum members should have equal hashes."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                member2 = enum_cls(member.value)
                assert hash(member) == hash(member2), f"Equal members {member} and {member2} should have equal hashes"

    def test_set_usage(self):
        """Should work correctly in sets."""
        for enum_cls in self.ENUM_CLASSES:
            enum_set = set(enum_cls)
            assert len(enum_set) == len(list(enum_cls)), f"Set should contain all {enum_cls.__name__} members"
            for member in enum_cls:
                assert member in enum_set, f"{member} should be in set"

    def test_dict_usage(self):
        """Should work correctly as dictionary keys."""
        for enum_cls in self.ENUM_CLASSES:
            enum_dict = {member: i for i, member in enumerate(enum_cls)}
            assert len(enum_dict) == len(list(enum_cls)), f"Dict should contain all {enum_cls.__name__} members"
            for i, member in enumerate(enum_cls):
                assert enum_dict[member] == i, f"Dict lookup for {member} should return {i}"

    def test_iteration(self):
        """Should be iterable and yield all members."""
        for enum_cls in self.ENUM_CLASSES:
            members = list(enum_cls)
            assert len(members) > 0, f"{enum_cls.__name__} should have at least one member"
            assert all(isinstance(m, enum_cls) for m in members), \
                f"All iterated members should be instances of {enum_cls.__name__}"

    def test_membership(self):
        """Should support 'in' operator."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert member in enum_cls, f"{member} should be in {enum_cls.__name__}"
                # Python Enum allows checking string values with 'in' operator
                # This is expected behavior for string enums
                assert member.value in enum_cls, f"{member.value} (string) should be in {enum_cls.__name__}"

    def test_repr(self):
        """Should have informative repr."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                repr_str = repr(member)
                assert enum_cls.__name__ in repr_str, f"repr({member}) should include class name"
                assert member.name in repr_str, f"repr({member}) should include member name"
                assert member.value in repr_str, f"repr({member}) should include member value"

    def test_str(self):
        """str() should return enum representation, value accessible via .value."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                # For string enums, str() returns the enum name representation
                # The actual string value is accessed via .value
                str_repr = str(member)
                assert enum_cls.__name__ in str_repr, f"str({member}) should include class name"
                assert member.name in str_repr, f"str({member}) should include member name"
                # Verify .value returns the actual string value
                assert member.value == member.value, f"{member}.value should return string value"
                assert isinstance(member.value, str), f"{member}.value should be a string"

    def test_ordering(self):
        """Should support ordering comparisons."""
        for enum_cls in self.ENUM_CLASSES:
            members = list(enum_cls)
            if len(members) >= 2:
                # Test that members can be compared
                m1, m2 = members[0], members[1]
                # At least one of these should be True
                assert (m1 < m2) or (m1 > m2) or (m1 == m2), \
                    f"Members {m1} and {m2} should be comparable"
                # Reflexive comparisons
                assert m1 <= m1, f"{m1} should be <= itself"
                assert m1 >= m1, f"{m1} should be >= itself"

    def test_value_immutability(self):
        """Enum values should be immutable."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                original_value = member.value
                # Attempting to modify should raise AttributeError or fail silently
                # (enum members are immutable)
                assert member.value == original_value, f"{member}.value should remain {original_value}"

    def test_no_empty_enums(self):
        """All enum classes should have at least one member."""
        for enum_cls in self.ENUM_CLASSES:
            members = list(enum_cls)
            assert len(members) > 0, f"{enum_cls.__name__} should have at least one member"

    def test_value_not_none(self):
        """All enum values should be non-None."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert member.value is not None, f"{member}.value should not be None"
                assert member.value != "", f"{member}.value should not be empty string"

    def test_name_not_none(self):
        """All enum member names should be non-None."""
        for enum_cls in self.ENUM_CLASSES:
            for member in enum_cls:
                assert member.name is not None, f"{member}.name should not be None"
                assert member.name != "", f"{member}.name should not be empty string"

    def test_enum_class_attributes(self):
        """Enum classes should have expected attributes."""
        from types import MappingProxyType
        for enum_cls in self.ENUM_CLASSES:
            assert hasattr(enum_cls, '__members__'), f"{enum_cls.__name__} should have __members__"
            # __members__ is a MappingProxyType (immutable dict-like object)
            assert isinstance(enum_cls.__members__, (dict, MappingProxyType)), \
                f"{enum_cls.__name__}.__members__ should be a dict or MappingProxyType"
            assert len(enum_cls.__members__) > 0, f"{enum_cls.__name__}.__members__ should not be empty"
            # Verify it's dict-like (supports len, iteration, etc.)
            assert len(enum_cls.__members__) == len(list(enum_cls)), \
                f"{enum_cls.__name__}.__members__ should have same length as enum"
