"""Unit tests for src/agents/helpers/metrics.py"""

from datetime import datetime, timezone
import time

import pytest

from src.agents.helpers.metrics import (
    log_agent_call,
    record_discrepancy,
)


class TestLogAgentCall:
    """Tests for log_agent_call function."""

    def test_records_metric_to_state(self):
        """Should record metric to state with all required fields."""
        start_time = datetime.now(timezone.utc)
        state = {"current_stage_id": "stage1"}
        
        recorder = log_agent_call("TestAgent", "test_node", start_time)
        recorder(state, {"execution_verdict": "pass"})
        
        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert len(state["metrics"]["agent_calls"]) == 1
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["agent"] == "TestAgent"
        assert metric["node"] == "test_node"
        assert metric["stage_id"] == "stage1"
        assert metric["verdict"] == "pass"
        assert "timestamp" in metric
        assert "duration_seconds" in metric
        assert isinstance(metric["timestamp"], str)
        assert isinstance(metric["duration_seconds"], (int, float))

    def test_initializes_metrics_if_missing(self):
        """Should initialize metrics structure if missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert isinstance(state["metrics"]["agent_calls"], list)
        assert len(state["metrics"]["agent_calls"]) == 1

    def test_initializes_agent_calls_if_missing(self):
        """Should initialize agent_calls list if metrics exists but agent_calls doesn't."""
        start_time = datetime.now(timezone.utc)
        state = {"metrics": {"stage_metrics": []}}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        assert "agent_calls" in state["metrics"]
        assert isinstance(state["metrics"]["agent_calls"], list)
        assert len(state["metrics"]["agent_calls"]) == 1
        assert len(state["metrics"]["stage_metrics"]) == 0  # Should preserve existing keys

    def test_records_duration(self):
        """Should record duration in seconds accurately."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        time.sleep(0.01)  # Small delay to ensure non-zero duration
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert "duration_seconds" in metric
        assert isinstance(metric["duration_seconds"], (int, float))
        assert metric["duration_seconds"] >= 0
        assert metric["duration_seconds"] >= 0.01  # Should be at least the sleep time

    def test_records_timestamp_in_iso_format(self):
        """Should record timestamp in ISO format."""
        start_time = datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["timestamp"] == "2024-01-15T12:30:45+00:00"
        # Verify it's parseable
        parsed = datetime.fromisoformat(metric["timestamp"])
        assert parsed == start_time

    def test_extracts_execution_verdict(self):
        """Should extract execution_verdict when present."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {"execution_verdict": "pass"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "pass"

    def test_extracts_physics_verdict(self):
        """Should extract physics_verdict when execution_verdict is missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {"physics_verdict": "valid"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "valid"

    def test_extracts_supervisor_verdict(self):
        """Should extract supervisor_verdict when execution and physics verdicts are missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {"supervisor_verdict": "approve"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "approve"

    def test_extracts_last_plan_review_verdict(self):
        """Should extract last_plan_review_verdict when other verdicts are missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {"last_plan_review_verdict": "needs_revision"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "needs_revision"

    def test_verdict_priority_execution_over_physics(self):
        """Should prioritize execution_verdict over physics_verdict."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {
            "execution_verdict": "pass",
            "physics_verdict": "invalid"
        })
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "pass"

    def test_verdict_priority_execution_over_supervisor(self):
        """Should prioritize execution_verdict over supervisor_verdict."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {
            "execution_verdict": "fail",
            "supervisor_verdict": "approve"
        })
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "fail"

    def test_verdict_priority_physics_over_supervisor(self):
        """Should prioritize physics_verdict over supervisor_verdict when execution is missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {
            "physics_verdict": "valid",
            "supervisor_verdict": "reject"
        })
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] == "valid"

    def test_verdict_none_when_no_verdict_keys(self):
        """Should set verdict to None when no verdict keys are present."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        log_agent_call("A", "n", start_time)(state, {"some_other_key": "value"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["verdict"] is None

    def test_records_error_when_present(self):
        """Should record run_error when present."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {"run_error": "Something went wrong"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["error"] == "Something went wrong"

    def test_error_none_when_not_present(self):
        """Should set error to None when run_error is not present."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {"execution_verdict": "pass"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["error"] is None

    def test_error_none_when_result_dict_is_none(self):
        """Should handle None result_dict gracefully."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, None)
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["error"] is None
        assert metric["verdict"] is None

    def test_error_none_when_result_dict_is_empty(self):
        """Should handle empty result_dict."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["error"] is None
        assert metric["verdict"] is None

    def test_stage_id_none_when_missing(self):
        """Should set stage_id to None when current_stage_id is missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["stage_id"] is None

    def test_stage_id_none_when_explicitly_none(self):
        """Should set stage_id to None when current_stage_id is explicitly None."""
        start_time = datetime.now(timezone.utc)
        state = {"current_stage_id": None}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["stage_id"] is None

    def test_multiple_calls_accumulate(self):
        """Should accumulate multiple agent calls in order."""
        start_time1 = datetime.now(timezone.utc)
        start_time2 = datetime.now(timezone.utc)
        state = {}
        
        recorder1 = log_agent_call("Agent1", "node1", start_time1)
        recorder2 = log_agent_call("Agent2", "node2", start_time2)
        
        recorder1(state, {"execution_verdict": "pass"})
        recorder2(state, {"execution_verdict": "fail"})
        
        assert len(state["metrics"]["agent_calls"]) == 2
        assert state["metrics"]["agent_calls"][0]["agent"] == "Agent1"
        assert state["metrics"]["agent_calls"][0]["node"] == "node1"
        assert state["metrics"]["agent_calls"][0]["verdict"] == "pass"
        assert state["metrics"]["agent_calls"][1]["agent"] == "Agent2"
        assert state["metrics"]["agent_calls"][1]["node"] == "node2"
        assert state["metrics"]["agent_calls"][1]["verdict"] == "fail"

    def test_all_required_fields_present(self):
        """Should include all required fields in metric."""
        start_time = datetime.now(timezone.utc)
        state = {"current_stage_id": "stage1"}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {"execution_verdict": "pass", "run_error": "error"})
        
        metric = state["metrics"]["agent_calls"][0]
        required_fields = ["agent", "node", "stage_id", "timestamp", "duration_seconds", "verdict", "error"]
        for field in required_fields:
            assert field in metric, f"Missing required field: {field}"


class TestRecordDiscrepancy:
    """Tests for record_discrepancy function."""

    def test_creates_discrepancy_entry_with_all_fields(self):
        """Should create complete discrepancy entry with all fields."""
        state = {"discrepancies_log": []}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="peak_wavelength",
            paper_value="500nm",
            simulation_value="520nm",
            classification="investigate",
            difference_percent=4.0,
            likely_cause="mesh resolution",
            action_taken="increased mesh",
            blocking=True,
        )
        
        assert "discrepancy" in result
        disc = result["discrepancy"]
        assert disc["id"] == "D1"
        assert disc["figure"] == "Fig1"
        assert disc["quantity"] == "peak_wavelength"
        assert disc["paper_value"] == "500nm"
        assert disc["simulation_value"] == "520nm"
        assert disc["classification"] == "investigate"
        assert disc["difference_percent"] == 4.0
        assert disc["likely_cause"] == "mesh resolution"
        assert disc["action_taken"] == "increased mesh"
        assert disc["blocking"] is True

    def test_all_required_fields_present(self):
        """Should include all required fields in discrepancy."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        required_fields = ["id", "figure", "quantity", "paper_value", "simulation_value", 
                          "classification", "difference_percent", "likely_cause", 
                          "action_taken", "blocking"]
        for field in required_fields:
            assert field in disc, f"Missing required field: {field}"

    def test_generates_unique_id_from_empty_log(self):
        """Should generate D1 when log is empty."""
        state = {"discrepancies_log": []}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert result["discrepancy"]["id"] == "D1"

    def test_generates_unique_id_from_existing_log(self):
        """Should generate unique discrepancy ID based on log length."""
        state = {"discrepancies_log": [{"id": "D1"}]}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert result["discrepancy"]["id"] == "D2"

    def test_generates_unique_id_with_multiple_entries(self):
        """Should generate correct ID when log has multiple entries."""
        state = {"discrepancies_log": [{"id": "D1"}, {"id": "D2"}, {"id": "D3"}]}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert result["discrepancy"]["id"] == "D4"

    def test_appends_to_log(self):
        """Should append new entry to discrepancies log."""
        state = {"discrepancies_log": [{"id": "D1", "figure": "Fig0"}]}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert len(result["discrepancies_log"]) == 2
        assert result["discrepancies_log"][0]["id"] == "D1"
        assert result["discrepancies_log"][0]["figure"] == "Fig0"
        assert result["discrepancies_log"][1]["id"] == "D2"
        assert result["discrepancies_log"][1]["figure"] == "Fig1"

    def test_uses_default_classification(self):
        """Should use default classification 'investigate'."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["classification"] == "investigate"

    def test_uses_custom_classification(self):
        """Should use provided classification value."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            classification="acceptable",
        )
        
        disc = result["discrepancy"]
        assert disc["classification"] == "acceptable"

    def test_all_classification_values(self):
        """Should accept all valid classification values."""
        state = {}
        classifications = ["acceptable", "investigate", "blocking"]
        
        for classification in classifications:
            result = record_discrepancy(
                state=state,
                stage_id="stage1",
                figure_id="Fig1",
                quantity="test",
                paper_value="1",
                simulation_value="2",
                classification=classification,
            )
            assert result["discrepancy"]["classification"] == classification
            state = {"discrepancies_log": state.get("discrepancies_log", []) + [result["discrepancy"]]}

    def test_uses_default_difference_percent(self):
        """Should use default difference_percent of 100.0."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["difference_percent"] == 100.0
        assert isinstance(disc["difference_percent"], float)

    def test_uses_custom_difference_percent(self):
        """Should use provided difference_percent value."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            difference_percent=5.5,
        )
        
        disc = result["discrepancy"]
        assert disc["difference_percent"] == 5.5

    def test_handles_negative_difference_percent(self):
        """Should accept negative difference_percent values."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            difference_percent=-10.0,
        )
        
        disc = result["discrepancy"]
        assert disc["difference_percent"] == -10.0

    def test_handles_large_difference_percent(self):
        """Should accept very large difference_percent values."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            difference_percent=999999.99,
        )
        
        disc = result["discrepancy"]
        assert disc["difference_percent"] == 999999.99

    def test_uses_default_blocking(self):
        """Should use default blocking value of True."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["blocking"] is True
        assert isinstance(disc["blocking"], bool)

    def test_uses_custom_blocking(self):
        """Should use provided blocking value."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            blocking=False,
        )
        
        disc = result["discrepancy"]
        assert disc["blocking"] is False

    def test_uses_default_likely_cause(self):
        """Should use default likely_cause of empty string."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["likely_cause"] == ""
        assert isinstance(disc["likely_cause"], str)

    def test_uses_custom_likely_cause(self):
        """Should use provided likely_cause value."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            likely_cause="mesh too coarse",
        )
        
        disc = result["discrepancy"]
        assert disc["likely_cause"] == "mesh too coarse"

    def test_uses_default_action_taken(self):
        """Should use default action_taken of empty string."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["action_taken"] == ""
        assert isinstance(disc["action_taken"], str)

    def test_uses_custom_action_taken(self):
        """Should use provided action_taken value."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
            action_taken="refined mesh",
        )
        
        disc = result["discrepancy"]
        assert disc["action_taken"] == "refined mesh"

    def test_handles_missing_discrepancies_log(self):
        """Should handle missing discrepancies_log by creating new list."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id=None,
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert "discrepancies_log" in result
        assert isinstance(result["discrepancies_log"], list)
        assert len(result["discrepancies_log"]) == 1
        assert result["discrepancy"]["id"] == "D1"

    def test_handles_empty_discrepancies_log(self):
        """Should handle empty discrepancies_log list."""
        state = {"discrepancies_log": []}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert len(result["discrepancies_log"]) == 1
        assert result["discrepancy"]["id"] == "D1"

    def test_handles_none_stage_id(self):
        """Should handle None stage_id."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id=None,
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        disc = result["discrepancy"]
        assert disc["id"] == "D1"
        assert disc["figure"] == "Fig1"

    def test_multiple_discrepancies_accumulate(self):
        """Should accumulate multiple discrepancies correctly."""
        state = {}
        
        result1 = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="wavelength",
            paper_value="500nm",
            simulation_value="520nm",
        )
        
        state["discrepancies_log"] = result1["discrepancies_log"]
        
        result2 = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig2",
            quantity="intensity",
            paper_value="0.5",
            simulation_value="0.6",
        )
        
        assert len(result2["discrepancies_log"]) == 2
        assert result2["discrepancies_log"][0]["id"] == "D1"
        assert result2["discrepancies_log"][0]["figure"] == "Fig1"
        assert result2["discrepancies_log"][1]["id"] == "D2"
        assert result2["discrepancies_log"][1]["figure"] == "Fig2"

    def test_returns_discrepancy_and_log(self):
        """Should return both discrepancy entry and updated log."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert "discrepancy" in result
        assert "discrepancies_log" in result
        assert result["discrepancy"] in result["discrepancies_log"]
        assert result["discrepancies_log"][0] == result["discrepancy"]

    def test_does_not_mutate_original_state(self):
        """Should not mutate the original state dict."""
        state = {"discrepancies_log": [{"id": "D1"}]}
        original_log = state["discrepancies_log"].copy()
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        # Original state should be unchanged
        assert len(state["discrepancies_log"]) == 1
        assert state["discrepancies_log"] == original_log
        # Result should have new log
        assert len(result["discrepancies_log"]) == 2

    def test_handles_empty_strings(self):
        """Should handle empty strings for string fields."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="",
            figure_id="",
            quantity="",
            paper_value="",
            simulation_value="",
            likely_cause="",
            action_taken="",
        )
        
        disc = result["discrepancy"]
        assert disc["figure"] == ""
        assert disc["quantity"] == ""
        assert disc["paper_value"] == ""
        assert disc["simulation_value"] == ""
        assert disc["likely_cause"] == ""
        assert disc["action_taken"] == ""

    def test_handles_unicode_strings(self):
        """Should handle unicode characters in string fields."""
        state = {}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="wavelength_λ",
            paper_value="500nm ± 5%",
            simulation_value="520nm",
            likely_cause="网格分辨率",
            action_taken="refined mesh",
        )
        
        disc = result["discrepancy"]
        assert disc["quantity"] == "wavelength_λ"
        assert disc["paper_value"] == "500nm ± 5%"
        assert disc["likely_cause"] == "网格分辨率"

    def test_handles_stage_id_with_existing_progress_stage(self):
        """Should handle stage_id when progress_stage exists."""
        state = {
            "progress": {
                "stages": [
                    {"stage_id": "stage1", "status": "in_progress"}
                ]
            }
        }
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        # Should still return discrepancy and log
        assert "discrepancy" in result
        assert "discrepancies_log" in result
        assert result["discrepancy"]["id"] == "D1"
        assert result["discrepancy"]["figure"] == "Fig1"

    def test_handles_stage_id_with_nonexistent_progress_stage(self):
        """Should handle stage_id when progress_stage doesn't exist."""
        state = {
            "progress": {
                "stages": [
                    {"stage_id": "other_stage", "status": "in_progress"}
                ]
            }
        }
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",  # This stage doesn't exist in progress
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        # Should still return discrepancy and log
        assert "discrepancy" in result
        assert "discrepancies_log" in result
        assert result["discrepancy"]["id"] == "D1"



