"""Integration-oriented tests for `run_code_node`."""

from unittest.mock import patch

from src.code_runner import run_code_node


class TestRunCodeNode:
    """Tests for run_code_node integration."""

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config(self, mock_run, code_runner_state_factory):
        """Test that config is extracted correctly from state."""
        mock_run.return_value = {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "output_files": [],
            "runtime_seconds": 1.0,
            "error": None,
            "memory_exceeded": False,
            "timeout_exceeded": False,
        }

        state = code_runner_state_factory()
        run_code_node(state)

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["code"] == state["code"]
        assert kwargs["stage_id"] == state["current_stage_id"]
        assert kwargs["config"]["timeout_seconds"] == 600  # 10 * 60
        assert kwargs["config"]["max_memory_gb"] == state["runtime_config"]["max_memory_gb"]
        assert kwargs["config"]["max_cpu_cores"] == state["runtime_config"]["max_cpu_cores"]

    def test_run_code_node_missing_code(self):
        """Test that missing code returns error."""
        state = {"current_stage_id": "test"}  # no 'code' key
        result = run_code_node(state)
        assert "No simulation code provided" in result["run_error"]

    def test_run_code_node_blocking_validation(self):
        """Test that blocking code validation prevents execution."""
        state = {
            "code": "input('block')",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert "Code contains blocking patterns" in result["run_error"]
        assert "validation_warnings" in result["stage_outputs"]

    def test_run_code_node_propagates_run_error(self):
        """Test that execution errors are propagated to state."""
        state = {
            "code": "raise Exception('fail')",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Execution failed" in result["run_error"] or "exit code" in result["run_error"]

    def test_run_code_node_propagates_flags(self, code_runner_state_factory):
        """Test that memory_exceeded and timeout_exceeded flags are propagated."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "output_files": [],
                "runtime_seconds": 10.0,
                "error": "Timeout",
                "memory_exceeded": False,
                "timeout_exceeded": True,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert result["stage_outputs"]["timeout_exceeded"] is True
            assert result["stage_outputs"]["memory_exceeded"] is False

    def test_run_code_node_integration_real_run(self, code_runner_state_factory):
        """
        Test that run_code_node works with real execution (no mocks)
        for a trivial script.
        """
        stage_id = "test_stage"
        state = code_runner_state_factory(
            code="print('real integration test')",
            current_stage_id=stage_id,
            plan={"stages": [{"stage_id": stage_id, "runtime_budget_minutes": 1}]},
        )

        result = run_code_node(state)

        assert result["run_error"] is None
        assert "real integration test" in result["stage_outputs"]["stdout"]
        assert result["stage_outputs"]["exit_code"] == 0

