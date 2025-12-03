"""Supervisor recovery and logging tests."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from src.agents.supervision import supervisor_node
from src.agents.supervision.supervisor import (
    _get_dependent_stages,
    _derive_stage_completion_outcome,
    _retry_archive_errors,
    _log_user_interaction,
)


class TestGetDependentStages:
    """Tests for _get_dependent_stages() helper function."""

    def test_returns_empty_for_no_dependents(self):
        """Should return empty list when no stages depend on target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": []},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == []

    def test_finds_direct_dependents(self):
        """Should find stages that directly depend on target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert set(result) == {"stage1", "stage2"}

    def test_finds_transitive_dependents(self):
        """Should find stages that transitively depend on target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage1"]},
                {"stage_id": "stage3", "dependencies": ["stage2"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert set(result) == {"stage1", "stage2", "stage3"}

    def test_handles_diamond_dependency(self):
        """Should handle diamond dependency pattern (no duplicates)."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage0"]},
                {"stage_id": "stage3", "dependencies": ["stage1", "stage2"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert set(result) == {"stage1", "stage2", "stage3"}
        # Verify no duplicates
        assert len(result) == 3

    def test_handles_empty_plan(self):
        """Should return empty list for empty plan."""
        plan = {"stages": []}
        result = _get_dependent_stages(plan, "stage0")
        assert result == []

    def test_handles_missing_stages_key(self):
        """Should return empty list when stages key is missing."""
        plan = {}
        result = _get_dependent_stages(plan, "stage0")
        assert result == []

    def test_handles_none_stages(self):
        """Should handle None stages list."""
        plan = {"stages": None}
        result = _get_dependent_stages(plan, "stage0")
        assert result == []

    def test_handles_none_dependencies(self):
        """Should handle None dependencies list in stage."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": None},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == ["stage1"]

    def test_handles_missing_dependencies_key(self):
        """Should handle stage without dependencies key."""
        plan = {
            "stages": [
                {"stage_id": "stage0"},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == ["stage1"]

    def test_handles_stage_without_stage_id(self):
        """Should skip stages without stage_id."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"name": "invalid_stage"},  # Missing stage_id
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == ["stage1"]

    def test_handles_non_dict_stage(self):
        """Should skip non-dict stages."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                "invalid_stage",  # Not a dict
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == ["stage1"]

    def test_handles_non_list_dependencies(self):
        """Should skip stages with non-list/tuple dependencies."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": "stage0"},  # String, not list
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert result == ["stage2"]

    def test_target_not_in_plan(self):
        """Should return empty list when target stage not in plan."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "nonexistent")
        assert result == []

    def test_multiple_dependency_chains(self):
        """Should find all dependents across multiple chains."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1a", "dependencies": ["stage0"]},
                {"stage_id": "stage1b", "dependencies": ["stage0"]},
                {"stage_id": "stage2a", "dependencies": ["stage1a"]},
                {"stage_id": "stage2b", "dependencies": ["stage1b"]},
                {"stage_id": "stage3", "dependencies": ["stage2a", "stage2b"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert set(result) == {"stage1a", "stage1b", "stage2a", "stage2b", "stage3"}

    def test_does_not_include_target_stage(self):
        """Should not include the target stage itself in results."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert "stage0" not in result


class TestDeriveStageCompletionOutcome:
    """Tests for _derive_stage_completion_outcome() helper function."""

    def test_excellent_match_returns_success(self):
        """Should return completed_success for EXCELLENT_MATCH classification."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_success"

    def test_acceptable_match_returns_success(self):
        """Should return completed_success for ACCEPTABLE_MATCH classification."""
        state = {
            "analysis_overall_classification": "ACCEPTABLE_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_success"

    def test_partial_match_returns_partial(self):
        """Should return completed_partial for PARTIAL_MATCH classification."""
        state = {
            "analysis_overall_classification": "PARTIAL_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_partial"

    def test_poor_match_returns_failed(self):
        """Should return completed_failed for POOR_MATCH classification."""
        state = {
            "analysis_overall_classification": "POOR_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_failed"

    def test_failed_returns_failed(self):
        """Should return completed_failed for FAILED classification."""
        state = {
            "analysis_overall_classification": "FAILED",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_failed"

    def test_no_targets_returns_success(self):
        """Should return completed_success for NO_TARGETS classification."""
        state = {
            "analysis_overall_classification": "NO_TARGETS",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_success"

    def test_comparison_needs_revision_downgrades_to_partial(self):
        """Should downgrade to partial when comparison_verdict is needs_revision."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": "needs_revision",
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_partial"

    def test_physics_warning_downgrades_to_partial(self):
        """Should downgrade to partial when physics_verdict is warning."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": "warning",
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_partial"

    def test_physics_fail_sets_failed(self):
        """Should set completed_failed when physics_verdict is fail."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": "fail",
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_failed"

    def test_missing_comparisons_sets_failed(self):
        """Should set completed_failed when stage has missing figure comparisons."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            # Uses figure_comparisons (not stage_comparisons) with classification
            # values that map to "missing": missing_output, fail, not_reproduced, mismatch, poor_match
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "fig1", "classification": "missing_output"}
            ],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_failed"
        assert "Missing outputs" in summary

    def test_pending_comparisons_downgrades_to_partial(self):
        """Should downgrade to partial when figure comparisons are pending."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            # Uses figure_comparisons with classification values that map to "pending":
            # pending_validation, partial_match, match_pending, partial
            "figure_comparisons": [
                {"stage_id": "stage1", "figure_id": "fig1", "classification": "pending_validation"}
            ],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_partial"
        assert "pending" in summary.lower()

    def test_empty_classification_with_physics_fail(self):
        """Should handle empty classification with physics fail."""
        state = {
            "analysis_overall_classification": "",
            "comparison_verdict": None,
            "physics_verdict": "fail",
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_failed"

    def test_none_classification(self):
        """Should handle None classification."""
        state = {
            "analysis_overall_classification": None,
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        # Default to completed_success when classification is None
        assert status == "completed_success"

    def test_summary_from_analysis_summary_dict(self):
        """Should extract summary from analysis_summary dict."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
            "analysis_summary": {
                "notes": "Test summary notes",
                "totals": {"matches": 5, "targets": 5}
            },
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert summary == "Test summary notes"

    def test_summary_fallback_to_totals(self):
        """Should use totals for summary when notes is missing."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
            "analysis_summary": {
                "totals": {"matches": 3, "targets": 5}
            },
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert "3/5" in summary

    def test_summary_from_string(self):
        """Should use string analysis_summary directly."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
            "analysis_summary": "Direct string summary",
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert summary == "Direct string summary"

    def test_summary_default_fallback(self):
        """Should provide default summary when no data available."""
        state = {
            "analysis_overall_classification": "EXCELLENT_MATCH",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert "EXCELLENT_MATCH" in summary

    def test_unknown_classification_defaults_to_success(self):
        """Should default to completed_success for unknown classification."""
        state = {
            "analysis_overall_classification": "UNKNOWN_VALUE",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_success"

    def test_case_insensitive_classification(self):
        """Should handle lowercase classification values."""
        state = {
            "analysis_overall_classification": "excellent_match",
            "comparison_verdict": None,
            "physics_verdict": None,
            "stage_comparisons": [],
        }
        status, summary = _derive_stage_completion_outcome(state, "stage1")
        assert status == "completed_success"


class TestArchiveErrorRecovery:
    """Tests for archive error recovery logic."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry failed archive operations and clear errors on success."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None  # Successful retry
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive errors are cleared
        assert result["archive_errors"] == []
        # Verify archive was called with correct arguments
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify normal supervision still runs
        assert result["supervisor_verdict"] == "ok_continue"
        assert "workflow_phase" in result

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_keeps_failed_retries(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should keep archive errors that still fail on retry."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Still failing")
        
        state = {
            "archive_errors": [{"stage_id": "stage1", "error": "Failed"}],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify error is preserved
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert result["archive_errors"][0]["error"] == "Failed"
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify normal supervision still runs
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_retries_multiple_failed_archives(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should retry all failed archive operations."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None  # All succeed
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed"},
                {"stage_id": "stage2", "error": "Failed"},
                {"stage_id": "stage3", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify all errors are cleared
        assert result["archive_errors"] == []
        # Verify all archives were called
        assert mock_archive.call_count == 3
        assert mock_archive.call_args_list[0][0] == (state, "stage1")
        assert mock_archive.call_args_list[1][0] == (state, "stage2")
        assert mock_archive.call_args_list[2][0] == (state, "stage3")

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_partial_retry_success(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should keep only failed retries when some succeed."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        def archive_side_effect(state, stage_id):
            if stage_id == "stage2":
                raise Exception("Still failing")
            return None
        
        mock_archive.side_effect = archive_side_effect
        
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Failed"},
                {"stage_id": "stage2", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify only failed retry is kept
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage2"
        assert mock_archive.call_count == 2

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_empty_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle empty archive_errors list."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive_errors is cleared
        assert result["archive_errors"] == []
        # Verify archive was not called
        mock_archive.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_missing_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle missing archive_errors key."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive_errors is set to empty list
        assert result["archive_errors"] == []
        # Verify archive was not called
        mock_archive.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_archive_error_without_stage_id(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should skip archive errors without stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                {"error": "Failed"},  # Missing stage_id
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was only called for stage1
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify error without stage_id is kept (not retried)
        assert len(result["archive_errors"]) == 1
        assert "stage_id" not in result["archive_errors"][0] or result["archive_errors"][0].get("stage_id") != "stage1"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_archive_error_with_none_stage_id(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should skip archive errors with None stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": [
                {"stage_id": None, "error": "Failed"},
                {"stage_id": "stage1", "error": "Failed"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was only called for stage1
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify error with None stage_id is kept (not retried)
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    def test_handles_non_list_archive_errors(self, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle archive_errors that is not a list."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "archive_errors": "not a list",  # Invalid type
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        # This should not crash - the code should handle it gracefully
        # If it crashes, that's a bug in the component
        result = supervisor_node(state)
        
        # Verify function completes
        assert "supervisor_verdict" in result
        # Archive should not be called with invalid archive_errors
        mock_archive.assert_not_called()
class TestUserInteractionLogging:
    """Tests for user interaction logging."""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction(self, mock_context, mock_handle_trigger):
        """Should log user interaction to progress with complete structure."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify progress is updated
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1
        
        # Verify interaction details - comprehensive check of all required fields
        interaction = result["progress"]["user_interactions"][0]
        
        # Check ID format
        assert interaction["id"] == "U1"
        
        # Check timestamp is valid ISO format
        assert "timestamp" in interaction
        assert isinstance(interaction["timestamp"], str)
        # Verify it's parseable as ISO datetime
        from datetime import datetime
        parsed_ts = datetime.fromisoformat(interaction["timestamp"].replace("Z", "+00:00"))
        assert parsed_ts is not None
        
        # Check interaction type
        assert interaction["interaction_type"] == "material_checkpoint"
        
        # Check context structure completely
        assert "context" in interaction
        assert interaction["context"]["stage_id"] == "stage0"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert interaction["context"]["reason"] == "material_checkpoint"
        
        # Check question and response
        assert interaction["question"] == "Material question"
        assert interaction["user_response"] == "APPROVE"
        
        # Check impact field exists
        assert "impact" in interaction
        
        # Check alternatives_considered exists and is a list
        assert "alternatives_considered" in interaction
        assert isinstance(interaction["alternatives_considered"], list)
        
        # Verify trigger is cleared
        assert result.get("ask_user_trigger") is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_multiple_user_interactions(self, mock_context, mock_handle_trigger):
        """Should append to existing user interactions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Material question"],
            "current_stage_id": "stage0",
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous"}
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # Verify new interaction is appended
        assert len(result["progress"]["user_interactions"]) == 2
        assert result["progress"]["user_interactions"][0]["id"] == "U1"
        assert result["progress"]["user_interactions"][1]["id"] == "U2"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_empty_questions(self, mock_context, mock_handle_trigger):
        """Should handle empty pending_user_questions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": [],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_questions(self, mock_context, mock_handle_trigger):
        """Should handle missing pending_user_questions."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["question"] == "(question cleared)"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_none_stage_id(self, mock_context, mock_handle_trigger):
        """Should handle None current_stage_id."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": None,
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["context"]["stage_id"] is None

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_multiple_responses(self, mock_context, mock_handle_trigger):
        """Should use last user response when multiple exist."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {
                "Question1": "REJECT",
                "Question2": "APPROVE",
            },
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        # Should use last value from dict
        assert interaction["user_response"] == "APPROVE"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_empty_responses(self, mock_context, mock_handle_trigger):
        """Should handle empty user_responses."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        interaction = result["progress"]["user_interactions"][0]
        assert interaction["user_response"] == ""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_does_not_log_when_no_trigger(self, mock_context, mock_handle_trigger):
        """Should not log interaction when ask_user_trigger is None."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "user_responses": {"Question": "APPROVE"},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should not have user_interactions in result
        assert "user_interactions" not in result.get("progress", {})

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_interaction_even_with_empty_responses(self, mock_context, mock_handle_trigger):
        """Should log interaction even when user_responses is empty (to track that user was asked)."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should log interaction even with empty responses (to track that user was asked)
        assert "user_interactions" in result.get("progress", {})
        assert len(result["progress"]["user_interactions"]) == 1
        assert result["progress"]["user_interactions"][0]["user_response"] == ""

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_progress(self, mock_context, mock_handle_trigger):
        """Should handle missing progress key."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
        }
        
        result = supervisor_node(state)
        
        # Should create progress with user_interactions
        assert "progress" in result
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_user_interaction_with_missing_user_interactions(self, mock_context, mock_handle_trigger):
        """Should handle progress without user_interactions key."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Question": "APPROVE"},
            "pending_user_questions": ["Question"],
            "current_stage_id": "stage0",
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should create user_interactions list
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_preserves_existing_progress_fields(self, mock_context, mock_handle_trigger):
        """Should preserve other progress fields when adding interaction."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Q": "APPROVE"},
            "pending_user_questions": ["Q"],
            "current_stage_id": "stage0",
            "progress": {
                "stages": [{"stage_id": "stage0", "status": "in_progress"}],
                "user_interactions": [],
                "custom_field": "preserved_value",
            },
        }
        
        result = supervisor_node(state)
        
        # Verify existing fields are preserved
        assert result["progress"]["custom_field"] == "preserved_value"
        assert len(result["progress"]["stages"]) == 1
        assert result["progress"]["stages"][0]["stage_id"] == "stage0"

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_logs_different_trigger_types(self, mock_context, mock_handle_trigger):
        """Should log correct interaction_type for different triggers."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        trigger_types = [
            "code_review_limit",
            "design_review_limit",
            "execution_failure_limit",
            "physics_failure_limit",
            "context_overflow",
            "replan_limit",
            "backtrack_approval",
        ]
        
        for trigger in trigger_types:
            state = {
                "ask_user_trigger": trigger,
                "user_responses": {"Q": "APPROVE"},
                "pending_user_questions": ["Q"],
                "current_stage_id": "stage0",
                "progress": {"stages": [], "user_interactions": []},
            }
            
            result = supervisor_node(state)
            
            assert result["progress"]["user_interactions"][0]["interaction_type"] == trigger

    @patch("src.agents.supervision.supervisor.handle_trigger")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_interaction_id_increments_correctly(self, mock_context, mock_handle_trigger):
        """Should increment interaction ID based on existing count."""
        mock_context.return_value = None
        mock_handle_trigger.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Q": "APPROVE"},
            "pending_user_questions": ["Q"],
            "current_stage_id": "stage0",
            "progress": {
                "stages": [],
                "user_interactions": [
                    {"id": "U1", "interaction_type": "previous1"},
                    {"id": "U2", "interaction_type": "previous2"},
                    {"id": "U3", "interaction_type": "previous3"},
                ]
            },
        }
        
        result = supervisor_node(state)
        
        # New interaction should be U4
        new_interaction = result["progress"]["user_interactions"][-1]
        assert new_interaction["id"] == "U4"
        assert len(result["progress"]["user_interactions"]) == 4


class TestInvalidUserResponses:
    """Tests for handling invalid user_responses."""

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_string(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle string user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": "invalid string",  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify function completes successfully
        assert result["supervisor_verdict"] == "ok_continue"
        # Verify warning was logged
        assert mock_logger.warning.called, "Expected warning to be logged for string user_responses"
        # Verify warning message contains expected info
        warning_call_args = mock_logger.warning.call_args[0][0]
        assert "user_responses" in warning_call_args.lower(), \
            f"Warning should mention 'user_responses', got: {warning_call_args}"
        assert "dict" in warning_call_args.lower(), \
            f"Warning should mention 'dict', got: {warning_call_args}"
        # Verify actual type is mentioned
        assert "str" in warning_call_args.lower(), \
            f"Warning should mention the actual type 'str', got: {warning_call_args}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_list(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle list user_responses gracefully and log warning with type info."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": ["response1", "response2"],  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert mock_logger.warning.called, "Expected warning to be logged for list user_responses"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "list" in warning_msg.lower(), f"Warning should mention type 'list', got: {warning_msg}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_int(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle int user_responses gracefully and log warning with type info."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": 42,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert mock_logger.warning.called, "Expected warning to be logged for int user_responses"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "int" in warning_msg.lower(), f"Warning should mention type 'int', got: {warning_msg}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_none(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle None user_responses gracefully and log warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": None,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # None should be handled (isinstance(None, dict) is False)
        assert mock_logger.warning.called, "Expected warning to be logged for None user_responses"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "nonetype" in warning_msg.lower(), f"Warning should mention type 'NoneType', got: {warning_msg}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_bool(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle bool user_responses gracefully and log warning with type info."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": True,  # Should be dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert mock_logger.warning.called, "Expected warning to be logged for bool user_responses"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "bool" in warning_msg.lower(), f"Warning should mention type 'bool', got: {warning_msg}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_non_dict_user_responses_set(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle set user_responses gracefully and log warning with type info."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": {"response1", "response2"},  # Should be dict (this is a set!)
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert mock_logger.warning.called, "Expected warning to be logged for set user_responses"
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "set" in warning_msg.lower(), f"Warning should mention type 'set', got: {warning_msg}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_valid_dict_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle valid dict user_responses without warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": {"Question": "APPROVE"},  # Valid dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Should not log warning for valid dict
        assert not mock_logger.warning.called, \
            f"Should NOT log warning for valid dict, but got: {mock_logger.warning.call_args}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_missing_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle missing user_responses key by defaulting to empty dict (no warning)."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Code uses state.get("user_responses", {}) with default of {},
        # so missing key returns empty dict (valid), no warning logged.
        assert not mock_logger.warning.called, \
            f"Should NOT log warning for missing key (defaults to empty dict), " \
            f"but got: {mock_logger.warning.call_args}"

    @patch("src.agents.supervision.supervisor.logging.getLogger")
    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_handles_empty_dict_user_responses(self, mock_prompt, mock_context, mock_call, mock_get_logger):
        """Should handle empty dict user_responses without warning."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        state = {
            "user_responses": {},  # Empty but valid dict
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        # Empty dict is still a valid dict, should not warn
        assert not mock_logger.warning.called, \
            f"Should NOT log warning for empty dict, but got: {mock_logger.warning.call_args}"


class TestNormalSupervision:
    """Tests for normal supervision path (when not handling user response)."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_calls_llm(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should call LLM for normal supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "system_prompt"
        mock_call.return_value = {"verdict": "ok_continue", "reasoning": "All good"}
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,  # No trigger = normal supervision
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
            "workflow_phase": "design",
        }
        
        result = supervisor_node(state)
        
        # Verify LLM was called
        mock_call.assert_called_once()
        call_args = mock_call.call_args
        assert call_args.kwargs["agent_name"] == "supervisor"
        assert call_args.kwargs["system_prompt"] == "system_prompt"
        assert "user_content" in call_args.kwargs
        assert call_args.kwargs["state"] == state
        
        # Verify verdict is set
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["supervisor_feedback"] == "All good"
        assert result["workflow_phase"] == "supervision"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_archives_current_stage(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should archive outputs for current stage."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive was called for current stage
        mock_archive.assert_called_once_with(state, "stage1")
        # Verify status update was called
        mock_update_status.assert_called_once()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_handles_archive_error(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle archive errors during normal supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Archive failed")
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive error is recorded
        assert "archive_errors" in result
        assert len(result["archive_errors"]) == 1
        assert result["archive_errors"][0]["stage_id"] == "stage1"
        assert "error" in result["archive_errors"][0]
        assert "timestamp" in result["archive_errors"][0]
        # Verify supervision still completes
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_no_current_stage(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should handle normal supervision without current_stage_id."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify archive is not called when no current stage
        mock_archive.assert_not_called()
        mock_update_status.assert_not_called()
        # Verify supervision still completes
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_handles_llm_exception(self, mock_prompt, mock_context, mock_call):
        """Should handle LLM call exceptions gracefully."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("LLM unavailable")
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify default verdict is set
        assert result["supervisor_verdict"] == "ok_continue"
        assert "supervisor_feedback" in result
        assert "LLM unavailable" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_backtrack_verdict(self, mock_prompt, mock_context, mock_call):
        """Should handle backtrack_to_stage verdict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "backtrack_to_stage",
            "backtrack_target": "stage0",
            "reasoning": "Need to restart"
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify backtrack decision is set
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage0"
        assert result["backtrack_decision"]["reason"] == "Need to restart"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_missing_verdict(self, mock_prompt, mock_context, mock_call):
        """Should default verdict when LLM doesn't return one."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {}  # No verdict
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify default verdict
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_should_stop_propagation(self, mock_prompt, mock_context, mock_call):
        """Should propagate should_stop from LLM response."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "all_complete",
            "should_stop": True,
            "reasoning": "All stages complete"
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify should_stop is propagated
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_replan_verdict(self, mock_prompt, mock_context, mock_call):
        """Should handle replan_needed verdict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "replan_needed",
            "reasoning": "Plan needs adjustment"
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert result["supervisor_feedback"] == "Plan needs adjustment"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_ask_user_verdict(self, mock_prompt, mock_context, mock_call):
        """Should handle ask_user verdict."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "ask_user",
            "reasoning": "Need user input"
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ask_user"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_backtrack_decision_structure(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should set complete backtrack_decision structure."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "backtrack_to_stage",
            "backtrack_target": "stage0",
            "reasoning": "Need to restart from stage0"
        }
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify backtrack decision structure
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage0"
        assert result["backtrack_decision"]["reason"] == "Need to restart from stage0"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_backtrack_without_target(self, mock_prompt, mock_context, mock_call):
        """Should not set backtrack_decision when target is missing."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {
            "verdict": "backtrack_to_stage",
            "reasoning": "Need to restart"
            # No backtrack_target
        }
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verdict is set, but backtrack_decision should not be created
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" not in result

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_user_content_includes_status(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should include workflow status in user_content."""
        mock_context.return_value = None
        mock_prompt.return_value = "system_prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.return_value = None
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage1",
            "workflow_phase": "design",
            "analysis_summary": {"notes": "test summary"},
            "plan": {"stages": []},
            "progress": {"stages": [
                {"stage_id": "stage0", "status": "completed_success"},
                {"stage_id": "stage1", "status": "in_progress"},
            ]},
        }
        
        result = supervisor_node(state)
        
        # Verify call_agent_with_metrics was called with proper content
        call_args = mock_call.call_args
        user_content = call_args.kwargs["user_content"]
        
        # Check user_content contains expected info
        assert "stage1" in user_content
        assert "design" in user_content
        assert "Completed: 1" in user_content
        assert "Pending: 1" in user_content

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    @patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress")
    @patch("src.agents.supervision.supervisor.update_progress_stage_status")
    def test_normal_supervision_multiple_archive_errors(self, mock_update_status, mock_archive, mock_prompt, mock_context, mock_call):
        """Should accumulate archive errors across retries and new failures."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        mock_archive.side_effect = Exception("Archive failed")
        
        state = {
            "ask_user_trigger": None,
            "current_stage_id": "stage2",
            "archive_errors": [
                {"stage_id": "stage0", "error": "Previous failure"},
            ],
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify both old and new errors are in result
        # Note: The retry for stage0 will fail again, and stage2 will also fail
        assert len(result["archive_errors"]) >= 1
        stage_ids = [e.get("stage_id") for e in result["archive_errors"]]
        # stage0 should be preserved (retry failed), stage2 should be added
        assert "stage0" in stage_ids
        assert "stage2" in stage_ids

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_llm_exception_includes_error_in_feedback(self, mock_prompt, mock_context, mock_call):
        """Should include error details in supervisor_feedback on LLM failure."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.side_effect = Exception("Connection timeout after 30s")
        
        state = {
            "ask_user_trigger": None,
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "Connection timeout" in result["supervisor_feedback"]

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_normal_supervision_workflow_phase_set(self, mock_prompt, mock_context, mock_call):
        """Should always set workflow_phase to supervision."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "ask_user_trigger": None,
            "workflow_phase": "design",  # Different initial phase
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        assert result["workflow_phase"] == "supervision"


class TestContextCheck:
    """Tests for context check and escalation handling."""

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_returns_awaiting_user_input(self, mock_prompt, mock_context, mock_call):
        """Should return early when context check requires user input."""
        mock_context.return_value = {"awaiting_user_input": True}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Should return context_update directly
        assert result["awaiting_user_input"] is True
        # Should not call LLM
        mock_call.assert_not_called()

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_updates_state(self, mock_prompt, mock_context, mock_call):
        """Should merge context_update into state."""
        mock_context.return_value = {"some_update": "value"}
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify function completes (state is merged internally)
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.call_agent_with_metrics")
    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    @patch("src.agents.supervision.supervisor.build_agent_prompt")
    def test_context_check_returns_none(self, mock_prompt, mock_context, mock_call):
        """Should continue normally when context check returns None."""
        mock_context.return_value = None
        mock_prompt.return_value = "prompt"
        mock_call.return_value = {"verdict": "ok_continue"}
        
        state = {
            "plan": {"stages": []},
            "progress": {"stages": []},
        }
        
        result = supervisor_node(state)
        
        # Verify function continues normally
        assert result["supervisor_verdict"] == "ok_continue"


class TestTriggerHandlerIntegration:
    """Tests for trigger handler integration through supervisor_node."""

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_material_checkpoint_approve(self, mock_context):
        """Should handle material_checkpoint with APPROVE response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Material question?": "APPROVE"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Material question?"],
            "current_stage_id": "stage0",
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify approval outcome
        assert result["supervisor_verdict"] == "ok_continue"
        assert result["supervisor_feedback"] == "Material validation approved by user."
        assert result["validated_materials"] == [{"material_id": "gold"}]
        assert result["pending_validated_materials"] == []
        # Verify trigger is cleared
        assert result["ask_user_trigger"] is None
        # Verify interaction was logged
        assert "user_interactions" in result["progress"]
        assert len(result["progress"]["user_interactions"]) == 1

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_material_checkpoint_change_database(self, mock_context):
        """Should handle material_checkpoint with CHANGE_DATABASE response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Q?": "CHANGE_DATABASE please update"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage0",
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify replan is triggered
        assert result["supervisor_verdict"] == "replan_needed"
        assert "database" in result["planner_feedback"].lower()
        # Materials should be cleared
        assert result["pending_validated_materials"] == []
        assert result["validated_materials"] == []

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_material_checkpoint_stop(self, mock_context):
        """Should handle material_checkpoint with STOP response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Q?": "STOP"},
            "pending_validated_materials": [{"material_id": "gold"}],
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage0",
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify stop outcome
        assert result["supervisor_verdict"] == "all_complete"
        assert result["should_stop"] is True

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_code_review_limit_provide_hint(self, mock_context):
        """Should handle code_review_limit with PROVIDE_HINT response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "code_review_limit",
            "user_responses": {"Q?": "PROVIDE_HINT: Try using numpy array instead"},
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage0",
            "code_revision_count": 5,
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter is reset
        assert result["code_revision_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"
        # Verify hint is passed to reviewer
        assert "numpy" in result["reviewer_feedback"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_design_review_limit_skip(self, mock_context):
        """Should handle design_review_limit with SKIP response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "design_review_limit",
            "user_responses": {"Q?": "SKIP"},
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage0",
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify skip outcome
        assert result["supervisor_verdict"] == "ok_continue"

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_backtrack_approval_approve(self, mock_context):
        """Should handle backtrack_approval with APPROVE response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Q?": "APPROVE"},
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage1",
            "backtrack_decision": {
                "target_stage_id": "stage0",
                "reason": "Need to restart"
            },
            "plan": {
                "stages": [
                    {"stage_id": "stage0", "dependencies": []},
                    {"stage_id": "stage1", "dependencies": ["stage0"]},
                    {"stage_id": "stage2", "dependencies": ["stage1"]},
                ]
            },
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify backtrack is approved
        assert result["supervisor_verdict"] == "backtrack_to_stage"
        assert "backtrack_decision" in result
        assert result["backtrack_decision"]["target_stage_id"] == "stage0"
        # Verify dependent stages are identified
        assert "stages_to_invalidate" in result["backtrack_decision"]
        assert set(result["backtrack_decision"]["stages_to_invalidate"]) == {"stage1", "stage2"}

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_backtrack_approval_reject(self, mock_context):
        """Should handle backtrack_approval with REJECT response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "backtrack_approval",
            "user_responses": {"Q?": "REJECT"},
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage1",
            "backtrack_decision": {"target_stage_id": "stage0"},
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify rejection clears backtrack suggestion
        assert result["supervisor_verdict"] == "ok_continue"
        assert result.get("backtrack_suggestion") is None

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_replan_limit_guidance(self, mock_context):
        """Should handle replan_limit with GUIDANCE response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "replan_limit",
            "user_responses": {"Q?": "GUIDANCE: Focus on optical properties"},
            "pending_user_questions": ["Q?"],
            "replan_count": 3,
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter reset and guidance passed
        assert result["replan_count"] == 0
        assert result["supervisor_verdict"] == "replan_needed"
        assert "optical" in result["planner_feedback"].lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_context_overflow_truncate(self, mock_context):
        """Should handle context_overflow with TRUNCATE response."""
        mock_context.return_value = None
        
        # Create a paper_text long enough to trigger truncation
        long_paper = "A" * 50000  # 50k chars
        
        state = {
            "ask_user_trigger": "context_overflow",
            "user_responses": {"Q?": "TRUNCATE"},
            "pending_user_questions": ["Q?"],
            "paper_text": long_paper,
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify truncation occurred
        assert result["supervisor_verdict"] == "ok_continue"
        assert "paper_text" in result
        assert len(result["paper_text"]) < len(long_paper)
        assert "TRUNCATED" in result["paper_text"]

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_unknown_trigger_defaults_to_continue(self, mock_context):
        """Should handle unknown trigger by defaulting to ok_continue."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "unknown_trigger_xyz",
            "user_responses": {"Q?": "some response"},
            "pending_user_questions": ["Q?"],
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should continue without crashing
        assert result["supervisor_verdict"] == "ok_continue"
        assert "unknown trigger" in result.get("supervisor_feedback", "").lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_trigger_with_unclear_response_asks_again(self, mock_context):
        """Should ask again when user response is unclear."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "material_checkpoint",
            "user_responses": {"Q?": "maybe I'm not sure"},
            "pending_user_questions": ["Q?"],
            "pending_validated_materials": [{"material_id": "gold"}],
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Should ask user again for clarification
        assert result["supervisor_verdict"] == "ask_user"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_llm_error_retry(self, mock_context):
        """Should handle llm_error with RETRY response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "llm_error",
            "user_responses": {"Q?": "RETRY"},
            "pending_user_questions": ["Q?"],
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"
        assert "retry" in result.get("supervisor_feedback", "").lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_deadlock_detected_replan(self, mock_context):
        """Should handle deadlock_detected with REPLAN response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "deadlock_detected",
            "user_responses": {"Q?": "REPLAN with different approach"},
            "pending_user_questions": ["Q?"],
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "replan_needed"
        assert "deadlock" in result.get("planner_feedback", "").lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_execution_failure_limit_retry(self, mock_context):
        """Should handle execution_failure_limit with RETRY response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "execution_failure_limit",
            "user_responses": {"Q?": "RETRY_WITH_GUIDANCE: increase mesh resolution"},
            "pending_user_questions": ["Q?"],
            "execution_failure_count": 5,
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        # Verify counter is reset
        assert result["execution_failure_count"] == 0
        assert result["supervisor_verdict"] == "ok_continue"
        assert "mesh" in result.get("supervisor_feedback", "").lower()

    @patch("src.agents.supervision.supervisor.check_context_or_escalate")
    def test_physics_failure_limit_accept_partial(self, mock_context):
        """Should handle physics_failure_limit with ACCEPT_PARTIAL response."""
        mock_context.return_value = None
        
        state = {
            "ask_user_trigger": "physics_failure_limit",
            "user_responses": {"Q?": "ACCEPT_PARTIAL"},
            "pending_user_questions": ["Q?"],
            "current_stage_id": "stage0",
            "plan": {"stages": []},
            "progress": {"stages": [], "user_interactions": []},
        }
        
        result = supervisor_node(state)
        
        assert result["supervisor_verdict"] == "ok_continue"


class TestRetryArchiveErrors:
    """Direct tests for _retry_archive_errors function."""

    def test_handles_dict_entry_in_archive_errors(self):
        """Should process valid dict entries in archive_errors."""
        mock_logger = MagicMock()
        state = {
            "archive_errors": [
                {"stage_id": "stage1", "error": "Original error"},
            ]
        }
        result = {}
        
        with patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress") as mock_archive:
            mock_archive.return_value = None  # Success
            _retry_archive_errors(state, result, mock_logger)
        
        assert result["archive_errors"] == []
        mock_archive.assert_called_once_with(state, "stage1")

    def test_handles_non_dict_entry_in_archive_errors(self):
        """Should preserve non-dict entries (skip them but keep in errors)."""
        mock_logger = MagicMock()
        state = {
            "archive_errors": [
                "string_error",  # Invalid - should be dict
                {"stage_id": "stage1", "error": "Valid error"},
            ]
        }
        result = {}
        
        with patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress") as mock_archive:
            mock_archive.return_value = None  # Success
            _retry_archive_errors(state, result, mock_logger)
        
        # Non-dict entry should be preserved
        assert "string_error" in result["archive_errors"]
        # Valid entry should be removed (retry succeeded)
        assert not any(e.get("stage_id") == "stage1" for e in result["archive_errors"] if isinstance(e, dict))
        # Warning should be logged for invalid entry
        assert mock_logger.warning.called

    def test_handles_empty_stage_id(self):
        """Should skip entries with empty string stage_id."""
        mock_logger = MagicMock()
        state = {
            "archive_errors": [
                {"stage_id": "", "error": "Error with empty stage_id"},
            ]
        }
        result = {}
        
        with patch("src.agents.supervision.supervisor.archive_stage_outputs_to_progress") as mock_archive:
            _retry_archive_errors(state, result, mock_logger)
        
        # Should be kept (not retried because empty stage_id is falsy)
        assert len(result["archive_errors"]) == 1
        mock_archive.assert_not_called()


class TestLogUserInteraction:
    """Direct tests for _log_user_interaction function."""

    def test_creates_complete_interaction_entry(self):
        """Should create interaction entry with all required fields."""
        state = {
            "progress": {"stages": [], "user_interactions": []},
            "pending_user_questions": ["Test question?"],
        }
        result = {}
        
        _log_user_interaction(
            state=state,
            result=result,
            ask_user_trigger="material_checkpoint",
            user_responses={"Q": "APPROVE"},
            current_stage_id="stage0",
        )
        
        # Verify complete structure
        interaction = result["progress"]["user_interactions"][0]
        
        assert interaction["id"] == "U1"
        assert interaction["interaction_type"] == "material_checkpoint"
        assert "timestamp" in interaction
        assert interaction["context"]["stage_id"] == "stage0"
        assert interaction["context"]["agent"] == "SupervisorAgent"
        assert interaction["context"]["reason"] == "material_checkpoint"
        assert interaction["question"] == "Test question?"
        assert interaction["user_response"] == "APPROVE"
        assert "impact" in interaction
        assert "alternatives_considered" in interaction
        assert isinstance(interaction["alternatives_considered"], list)

    def test_handles_multiple_questions_uses_first(self):
        """Should use first question when multiple are pending."""
        state = {
            "progress": {"stages": [], "user_interactions": []},
            "pending_user_questions": ["First question?", "Second question?"],
        }
        result = {}
        
        _log_user_interaction(
            state=state,
            result=result,
            ask_user_trigger="test",
            user_responses={"Q": "Response"},
            current_stage_id="stage0",
        )
        
        assert result["progress"]["user_interactions"][0]["question"] == "First question?"

    def test_handles_non_string_question(self):
        """Should convert non-string questions to string."""
        state = {
            "progress": {"stages": [], "user_interactions": []},
            "pending_user_questions": [123],  # Non-string question
        }
        result = {}
        
        _log_user_interaction(
            state=state,
            result=result,
            ask_user_trigger="test",
            user_responses={"Q": "Response"},
            current_stage_id="stage0",
        )
        
        assert result["progress"]["user_interactions"][0]["question"] == "123"
