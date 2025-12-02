"""Unit tests for src/agents/helpers/metrics.py"""

from datetime import datetime, timezone

import pytest

from src.agents.helpers.metrics import (
    log_agent_call,
    record_discrepancy,
)


class TestLogAgentCall:
    """Tests for log_agent_call function."""

    def test_records_metric_to_state(self):
        """Should record metric to state."""
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

    def test_initializes_metrics_if_missing(self):
        """Should initialize metrics structure if missing."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        assert "metrics" in state
        assert "agent_calls" in state["metrics"]

    def test_records_duration(self):
        """Should record duration in seconds."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {})
        
        metric = state["metrics"]["agent_calls"][0]
        assert "duration_seconds" in metric
        assert metric["duration_seconds"] >= 0

    def test_extracts_various_verdicts(self):
        """Should extract verdict from various result keys."""
        start_time = datetime.now(timezone.utc)
        
        # Test execution_verdict
        state1 = {}
        log_agent_call("A", "n", start_time)(state1, {"execution_verdict": "pass"})
        assert state1["metrics"]["agent_calls"][0]["verdict"] == "pass"
        
        # Test physics_verdict
        state2 = {}
        log_agent_call("A", "n", start_time)(state2, {"physics_verdict": "valid"})
        assert state2["metrics"]["agent_calls"][0]["verdict"] == "valid"
        
        # Test supervisor_verdict
        state3 = {}
        log_agent_call("A", "n", start_time)(state3, {"supervisor_verdict": "approve"})
        assert state3["metrics"]["agent_calls"][0]["verdict"] == "approve"

    def test_records_error(self):
        """Should record run_error if present."""
        start_time = datetime.now(timezone.utc)
        state = {}
        
        recorder = log_agent_call("Agent", "node", start_time)
        recorder(state, {"run_error": "Something went wrong"})
        
        metric = state["metrics"]["agent_calls"][0]
        assert metric["error"] == "Something went wrong"


class TestRecordDiscrepancy:
    """Tests for record_discrepancy function."""

    def test_creates_discrepancy_entry(self):
        """Should create discrepancy entry."""
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
        assert disc["figure"] == "Fig1"
        assert disc["quantity"] == "peak_wavelength"
        assert disc["paper_value"] == "500nm"
        assert disc["simulation_value"] == "520nm"
        assert disc["difference_percent"] == 4.0
        assert disc["blocking"] is True

    def test_generates_unique_id(self):
        """Should generate unique discrepancy ID."""
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

    def test_appends_to_log(self):
        """Should append to discrepancies log."""
        state = {"discrepancies_log": [{"id": "D1"}]}
        
        result = record_discrepancy(
            state=state,
            stage_id="stage1",
            figure_id="Fig1",
            quantity="test",
            paper_value="1",
            simulation_value="2",
        )
        
        assert len(result["discrepancies_log"]) == 2

    def test_uses_default_values(self):
        """Should use default values for optional parameters."""
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
        assert disc["difference_percent"] == 100.0
        assert disc["blocking"] is True

    def test_handles_empty_log(self):
        """Should handle missing discrepancies_log."""
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
        assert len(result["discrepancies_log"]) == 1
        assert result["discrepancy"]["id"] == "D1"



