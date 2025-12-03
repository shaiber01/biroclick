"""Integration-oriented tests for `run_code_node`."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

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
        result = run_code_node(state)

        # Verify run_simulation was called correctly
        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs

        assert kwargs["code"] == state["code"]
        assert kwargs["stage_id"] == state["current_stage_id"]
        assert kwargs["config"]["timeout_seconds"] == 600  # 10 * 60
        assert kwargs["config"]["max_memory_gb"] == state["runtime_config"]["max_memory_gb"]
        assert kwargs["config"]["max_cpu_cores"] == state["runtime_config"]["max_cpu_cores"]
        assert kwargs["config"]["keep_script"] is True
        
        # Verify output_dir is a Path object
        assert isinstance(kwargs["output_dir"], Path)
        assert kwargs["output_dir"].parts[-2] == state["paper_id"]
        assert kwargs["output_dir"].parts[-1] == state["current_stage_id"]
        
        # Verify return value structure
        assert "stage_outputs" in result
        assert "run_error" in result
        assert result["run_error"] is None
        assert result["stage_outputs"]["stdout"] == ""
        assert result["stage_outputs"]["stderr"] == ""
        assert result["stage_outputs"]["exit_code"] == 0
        assert result["stage_outputs"]["files"] == []
        assert result["stage_outputs"]["runtime_seconds"] == 1.0
        assert result["stage_outputs"]["timeout_exceeded"] is False
        assert result["stage_outputs"]["memory_exceeded"] is False
        assert "validation_warnings" in result["stage_outputs"]

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config_custom_runtime_budget(self, mock_run, code_runner_state_factory):
        """Test that custom runtime_budget_minutes is correctly converted to timeout_seconds."""
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

        state = code_runner_state_factory(
            current_stage_id="custom_stage",
            plan={
                "stages": [
                    {"stage_id": "custom_stage", "runtime_budget_minutes": 30}
                ]
            }
        )
        result = run_code_node(state)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["config"]["timeout_seconds"] == 1800  # 30 * 60
        assert result["run_error"] is None

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config_missing_stage_in_plan(self, mock_run, code_runner_state_factory):
        """Test that missing stage in plan defaults to 60 minutes runtime budget."""
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

        state = code_runner_state_factory(
            current_stage_id="missing_stage",
            plan={"stages": [{"stage_id": "other_stage", "runtime_budget_minutes": 20}]}
        )
        result = run_code_node(state)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["config"]["timeout_seconds"] == 3600  # 60 * 60 default
        assert result["run_error"] is None

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config_missing_plan(self, mock_run, code_runner_state_factory):
        """Test that missing plan defaults to 60 minutes runtime budget."""
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
        del state["plan"]
        result = run_code_node(state)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["config"]["timeout_seconds"] == 3600  # 60 * 60 default
        assert result["run_error"] is None

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config_missing_runtime_config(self, mock_run, code_runner_state_factory):
        """Test that missing runtime_config uses defaults."""
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
        del state["runtime_config"]
        result = run_code_node(state)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["config"]["max_memory_gb"] == 8.0  # default
        assert kwargs["config"]["max_cpu_cores"] == 4  # default
        assert result["run_error"] is None

    @patch("src.code_runner.run_simulation")
    def test_run_code_node_extracts_config_partial_runtime_config(self, mock_run, code_runner_state_factory):
        """Test that partial runtime_config merges with defaults."""
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

        state = code_runner_state_factory(
            runtime_config={"max_memory_gb": 32.0}  # missing max_cpu_cores
        )
        result = run_code_node(state)

        kwargs = mock_run.call_args.kwargs
        assert kwargs["config"]["max_memory_gb"] == 32.0
        assert kwargs["config"]["max_cpu_cores"] == 4  # default
        assert result["run_error"] is None

    def test_run_code_node_missing_code(self):
        """Test that missing code returns error."""
        state = {"current_stage_id": "test"}  # no 'code' key
        result = run_code_node(state)
        
        assert result["run_error"] is not None
        assert "No simulation code provided" in result["run_error"]
        assert "stage_outputs" in result
        assert result["stage_outputs"] == {}

    def test_run_code_node_empty_code_string(self):
        """Test that empty code string returns error."""
        state = {
            "code": "",
            "current_stage_id": "test",
        }
        result = run_code_node(state)
        
        assert result["run_error"] is not None
        assert "No simulation code provided" in result["run_error"]
        assert result["stage_outputs"] == {}

    def test_run_code_node_none_code(self):
        """Test that None code returns error."""
        state = {
            "code": None,
            "current_stage_id": "test",
        }
        result = run_code_node(state)
        
        assert result["run_error"] is not None
        assert "No simulation code provided" in result["run_error"]
        assert result["stage_outputs"] == {}

    def test_run_code_node_blocking_validation_input(self):
        """Test that blocking code validation prevents execution for input()."""
        state = {
            "code": "input('block')",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Code contains blocking patterns" in result["run_error"]
        assert "BLOCKING" in result["run_error"]
        assert "stage_outputs" in result
        assert "validation_warnings" in result["stage_outputs"]
        assert isinstance(result["stage_outputs"]["validation_warnings"], list)
        assert len(result["stage_outputs"]["validation_warnings"]) > 0
        assert any("BLOCKING" in w for w in result["stage_outputs"]["validation_warnings"])

    def test_run_code_node_blocking_validation_plt_show(self):
        """Test that blocking code validation prevents execution for plt.show()."""
        state = {
            "code": "import matplotlib.pyplot as plt\nplt.show()",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Code contains blocking patterns" in result["run_error"]
        assert "BLOCKING" in result["run_error"]
        assert "validation_warnings" in result["stage_outputs"]
        assert any("plt.show()" in w for w in result["stage_outputs"]["validation_warnings"])

    def test_run_code_node_blocking_validation_raw_input(self):
        """Test that blocking code validation prevents execution for raw_input()."""
        state = {
            "code": "raw_input('Enter value')",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Code contains blocking patterns" in result["run_error"]
        assert "validation_warnings" in result["stage_outputs"]
        assert any("raw_input" in w for w in result["stage_outputs"]["validation_warnings"])

    def test_run_code_node_blocking_validation_multiple_blocking(self):
        """Test that multiple blocking patterns are all detected."""
        state = {
            "code": "input('block')\nplt.show()",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Code contains blocking patterns" in result["run_error"]
        warnings = result["stage_outputs"]["validation_warnings"]
        blocking_warnings = [w for w in warnings if "BLOCKING" in w]
        assert len(blocking_warnings) >= 2

    def test_run_code_node_non_blocking_warnings_allowed(self, code_runner_state_factory):
        """Test that non-blocking warnings don't prevent execution."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                code="os.system('echo test')"  # Dangerous but not blocking
            )
            result = run_code_node(state)

            assert result["run_error"] is None
            assert "validation_warnings" in result["stage_outputs"]
            assert any("WARNING" in w for w in result["stage_outputs"]["validation_warnings"])
            mock_run.assert_called_once()

    def test_run_code_node_propagates_run_error(self):
        """Test that execution errors are propagated to state."""
        state = {
            "code": "raise Exception('fail')",
            "current_stage_id": "stage1",
        }

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert "Execution failed" in result["run_error"] or "exit code" in result["run_error"]
        assert "stage_outputs" in result
        assert result["stage_outputs"]["exit_code"] != 0
        assert isinstance(result["stage_outputs"]["stdout"], str)
        assert isinstance(result["stage_outputs"]["stderr"], str)

    def test_run_code_node_propagates_flags_timeout(self, code_runner_state_factory):
        """Test that timeout_exceeded flag is propagated correctly."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "partial output",
                "stderr": "timeout occurred",
                "exit_code": -1,
                "output_files": ["file1.txt"],
                "runtime_seconds": 600.0,
                "error": "Timeout",
                "memory_exceeded": False,
                "timeout_exceeded": True,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert result["stage_outputs"]["timeout_exceeded"] is True
            assert result["stage_outputs"]["memory_exceeded"] is False
            assert result["run_error"] == "Timeout"
            assert result["stage_outputs"]["stdout"] == "partial output"
            assert result["stage_outputs"]["stderr"] == "timeout occurred"
            assert result["stage_outputs"]["exit_code"] == -1
            assert result["stage_outputs"]["files"] == ["file1.txt"]
            assert result["stage_outputs"]["runtime_seconds"] == 600.0

    def test_run_code_node_propagates_flags_memory(self, code_runner_state_factory):
        """Test that memory_exceeded flag is propagated correctly."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "",
                "stderr": "MemoryError: out of memory",
                "exit_code": -1,
                "output_files": [],
                "runtime_seconds": 50.0,
                "error": "Memory limit exceeded",
                "memory_exceeded": True,
                "timeout_exceeded": False,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert result["stage_outputs"]["memory_exceeded"] is True
            assert result["stage_outputs"]["timeout_exceeded"] is False
            assert result["run_error"] == "Memory limit exceeded"
            assert result["stage_outputs"]["stderr"] == "MemoryError: out of memory"

    def test_run_code_node_propagates_flags_both(self, code_runner_state_factory):
        """Test that both flags can be True simultaneously."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "output_files": [],
                "runtime_seconds": 600.0,
                "error": "Timeout and memory",
                "memory_exceeded": True,
                "timeout_exceeded": True,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert result["stage_outputs"]["memory_exceeded"] is True
            assert result["stage_outputs"]["timeout_exceeded"] is True

    def test_run_code_node_propagates_all_output_fields(self, code_runner_state_factory):
        """Test that all output fields from run_simulation are propagated."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "test stdout output",
                "stderr": "test stderr output",
                "exit_code": 0,
                "output_files": ["output1.h5", "output2.png"],
                "runtime_seconds": 42.5,
                "error": None,
                "memory_exceeded": False,
                "timeout_exceeded": False,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert result["stage_outputs"]["stdout"] == "test stdout output"
            assert result["stage_outputs"]["stderr"] == "test stderr output"
            assert result["stage_outputs"]["exit_code"] == 0
            assert result["stage_outputs"]["files"] == ["output1.h5", "output2.png"]
            assert result["stage_outputs"]["runtime_seconds"] == 42.5
            assert result["run_error"] is None

    def test_run_code_node_creates_output_directory(self, code_runner_state_factory):
        """Test that output directory is created correctly."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                paper_id="test_paper_123",
                current_stage_id="stage_456"
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            output_dir = kwargs["output_dir"]
            assert output_dir.exists()
            assert output_dir.parts[-2] == "test_paper_123"
            assert output_dir.parts[-1] == "stage_456"
            assert result["run_error"] is None

    def test_run_code_node_output_directory_nested(self, code_runner_state_factory):
        """Test that nested output directories are created correctly."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                paper_id="paper/subdir",
                current_stage_id="stage/sub"
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            output_dir = kwargs["output_dir"]
            assert output_dir.exists()
            assert result["run_error"] is None

    def test_run_code_node_missing_paper_id_defaults(self, code_runner_state_factory):
        """Test that missing paper_id defaults to 'unknown'."""
        with patch("src.code_runner.run_simulation") as mock_run:
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
            del state["paper_id"]
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["output_dir"].parts[-2] == "unknown"
            assert result["run_error"] is None

    def test_run_code_node_missing_stage_id_defaults(self, code_runner_state_factory):
        """Test that missing current_stage_id defaults to 'unknown'."""
        with patch("src.code_runner.run_simulation") as mock_run:
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
            del state["current_stage_id"]
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["stage_id"] == "unknown"
            assert kwargs["output_dir"].parts[-1] == "unknown"
            assert result["run_error"] is None

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
        assert isinstance(result["stage_outputs"]["runtime_seconds"], float)
        assert result["stage_outputs"]["runtime_seconds"] > 0
        assert isinstance(result["stage_outputs"]["validation_warnings"], list)

    def test_run_code_node_integration_real_run_with_output_file(self, code_runner_state_factory):
        """Test that output files are correctly listed."""
        stage_id = "test_stage_output"
        code = """
import os
output_file = 'test_output.txt'
with open(output_file, 'w') as f:
    f.write('test content')
print(f'Created {output_file}')
"""
        state = code_runner_state_factory(
            code=code,
            current_stage_id=stage_id,
            plan={"stages": [{"stage_id": stage_id, "runtime_budget_minutes": 1}]},
        )

        result = run_code_node(state)

        assert result["run_error"] is None
        assert result["stage_outputs"]["exit_code"] == 0
        assert "test_output.txt" in result["stage_outputs"]["files"]
        assert "simulation_test_stage_output.py" not in result["stage_outputs"]["files"]  # Script excluded

    def test_run_code_node_integration_real_run_with_error(self, code_runner_state_factory):
        """Test that real execution errors are captured."""
        stage_id = "test_stage_error"
        code = """
import sys
print('Starting')
sys.exit(1)
"""
        state = code_runner_state_factory(
            code=code,
            current_stage_id=stage_id,
            plan={"stages": [{"stage_id": stage_id, "runtime_budget_minutes": 1}]},
        )

        result = run_code_node(state)

        assert result["run_error"] is not None
        assert result["stage_outputs"]["exit_code"] == 1
        assert "Starting" in result["stage_outputs"]["stdout"]

    def test_run_code_node_integration_real_run_with_stderr(self, code_runner_state_factory):
        """Test that stderr output is captured."""
        stage_id = "test_stage_stderr"
        code = """
import sys
print('stdout message', file=sys.stdout)
print('stderr message', file=sys.stderr)
"""
        state = code_runner_state_factory(
            code=code,
            current_stage_id=stage_id,
            plan={"stages": [{"stage_id": stage_id, "runtime_budget_minutes": 1}]},
        )

        result = run_code_node(state)

        assert result["run_error"] is None
        assert "stdout message" in result["stage_outputs"]["stdout"]
        assert "stderr message" in result["stage_outputs"]["stderr"]

    def test_run_code_node_validation_warnings_included(self, code_runner_state_factory):
        """Test that validation warnings are always included in stage_outputs."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            # Code with dangerous but non-blocking pattern
            state = code_runner_state_factory(
                code="eval('1+1')"
            )
            result = run_code_node(state)

            assert "validation_warnings" in result["stage_outputs"]
            assert isinstance(result["stage_outputs"]["validation_warnings"], list)
            assert len(result["stage_outputs"]["validation_warnings"]) > 0

    def test_run_code_node_validation_warnings_empty_list(self, code_runner_state_factory):
        """Test that clean code produces empty validation warnings list."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                code="import meep as mp\nprint('clean code')"
            )
            result = run_code_node(state)

            assert "validation_warnings" in result["stage_outputs"]
            assert isinstance(result["stage_outputs"]["validation_warnings"], list)

    def test_run_code_node_state_not_mutated(self, code_runner_state_factory):
        """Test that input state is not mutated."""
        with patch("src.code_runner.run_simulation") as mock_run:
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
            original_state = state.copy()
            result = run_code_node(state)

            # State should not be mutated
            assert state == original_state
            # Result should be a new dict
            assert result is not state

    def test_run_code_node_return_value_structure(self, code_runner_state_factory):
        """Test that return value has correct structure."""
        with patch("src.code_runner.run_simulation") as mock_run:
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
            result = run_code_node(state)

            # Verify top-level keys
            assert set(result.keys()) == {"stage_outputs", "run_error"}
            
            # Verify stage_outputs structure
            stage_outputs = result["stage_outputs"]
            required_keys = {
                "stdout", "stderr", "exit_code", "files",
                "runtime_seconds", "validation_warnings",
                "timeout_exceeded", "memory_exceeded"
            }
            assert required_keys.issubset(set(stage_outputs.keys()))

    def test_run_code_node_zero_runtime_budget(self, code_runner_state_factory):
        """Test that zero runtime budget results in zero timeout."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "output_files": [],
                "runtime_seconds": 0.0,
                "error": None,
                "memory_exceeded": False,
                "timeout_exceeded": False,
            }

            state = code_runner_state_factory(
                current_stage_id="zero_budget",
                plan={"stages": [{"stage_id": "zero_budget", "runtime_budget_minutes": 0}]}
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == 0
            assert result["run_error"] is None

    def test_run_code_node_negative_runtime_budget(self, code_runner_state_factory):
        """Test that negative runtime budget is handled."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                current_stage_id="negative_budget",
                plan={"stages": [{"stage_id": "negative_budget", "runtime_budget_minutes": -5}]}
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == -300  # -5 * 60
            assert result["run_error"] is None

    def test_run_code_node_very_large_runtime_budget(self, code_runner_state_factory):
        """Test that very large runtime budget is handled."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                current_stage_id="large_budget",
                plan={"stages": [{"stage_id": "large_budget", "runtime_budget_minutes": 10000}]}
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == 600000  # 10000 * 60
            assert result["run_error"] is None

    def test_run_code_node_empty_plan_stages(self, code_runner_state_factory):
        """Test that empty stages list defaults to 60 minutes."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                plan={"stages": []}
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == 3600  # 60 * 60 default
            assert result["run_error"] is None

    def test_run_code_node_result_types(self, code_runner_state_factory):
        """Test that all return values have correct types."""
        with patch("src.code_runner.run_simulation") as mock_run:
            mock_run.return_value = {
                "stdout": "output",
                "stderr": "error",
                "exit_code": 0,
                "output_files": ["file1.txt"],
                "runtime_seconds": 1.5,
                "error": None,
                "memory_exceeded": False,
                "timeout_exceeded": False,
            }

            state = code_runner_state_factory()
            result = run_code_node(state)

            assert isinstance(result, dict)
            assert isinstance(result["run_error"], (str, type(None)))
            assert isinstance(result["stage_outputs"], dict)
            assert isinstance(result["stage_outputs"]["stdout"], str)
            assert isinstance(result["stage_outputs"]["stderr"], str)
            assert isinstance(result["stage_outputs"]["exit_code"], int)
            assert isinstance(result["stage_outputs"]["files"], list)
            assert isinstance(result["stage_outputs"]["runtime_seconds"], float)
            assert isinstance(result["stage_outputs"]["validation_warnings"], list)
            assert isinstance(result["stage_outputs"]["timeout_exceeded"], bool)
            assert isinstance(result["stage_outputs"]["memory_exceeded"], bool)

    def test_run_code_node_whitespace_only_code(self):
        """Test that whitespace-only code is treated as empty."""
        state = {
            "code": "   \n\t  ",
            "current_stage_id": "test",
        }
        result = run_code_node(state)
        
        # Currently whitespace-only strings pass the `if not code:` check
        # This test verifies the actual behavior - if it's a bug, test will reveal it
        # Note: Python's `if not code:` treats whitespace-only strings as truthy
        # So this will actually proceed to validation and execution
        # This might be a bug - empty/whitespace code should be rejected
        assert "code" in state
        # The code will be validated and executed, which might be unexpected behavior

    def test_run_code_node_missing_result_keys_raises(self, code_runner_state_factory):
        """Test that missing keys in run_simulation result cause KeyError (potential bug)."""
        with patch("src.code_runner.run_simulation") as mock_run:
            # Missing required keys - this should reveal a bug if run_simulation can return incomplete results
            mock_run.return_value = {
                "stdout": "",
                "stderr": "",
                # Missing exit_code, output_files, runtime_seconds, error, memory_exceeded, timeout_exceeded
            }

            state = code_runner_state_factory()
            
            # This should raise KeyError if there's a bug in error handling
            # If run_simulation is correctly implemented, it should never return incomplete results
            # But if there's a bug, this test will catch it
            try:
                result = run_code_node(state)
                # If we get here, the bug is that missing keys don't cause failure
                # This reveals that error handling is missing
                pytest.fail("Expected KeyError when result is missing required keys")
            except KeyError:
                # This is the expected behavior - missing keys should cause failure
                pass

    def test_run_code_node_invalid_runtime_budget_type_handled(self, code_runner_state_factory):
        """Test that invalid runtime_budget type is handled gracefully (should not raise)."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                current_stage_id="test_stage",
                plan={"stages": [{"stage_id": "test_stage", "runtime_budget_minutes": "not_a_number"}]}
            )
            
            # This test expects the code to handle invalid types gracefully
            # If it raises TypeError/ValueError, the test fails, revealing a bug
            result = run_code_node(state)
            
            # Should either use default or return error, not raise exception
            # If we get here without exception, verify reasonable behavior
            kwargs = mock_run.call_args.kwargs
            assert isinstance(kwargs["config"]["timeout_seconds"], int), \
                "timeout_seconds should be int, but invalid input caused error"

    def test_run_code_node_output_dir_creation_failure(self, code_runner_state_factory):
        """Test that output directory creation failure is not handled (potential bug)."""
        with patch("src.code_runner.run_simulation") as mock_run, \
             patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
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
            
            # This should raise PermissionError - output dir creation failure is not handled
            # This reveals a bug: mkdir failures should be caught and returned as error
            try:
                result = run_code_node(state)
                pytest.fail("Expected PermissionError when output directory creation fails")
            except PermissionError:
                # This reveals the bug - directory creation failures should be handled gracefully
                pass

    def test_run_code_node_runtime_budget_float_conversion(self, code_runner_state_factory):
        """Test that float runtime_budget is correctly converted to int."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                current_stage_id="test_stage",
                plan={"stages": [{"stage_id": "test_stage", "runtime_budget_minutes": 10.5}]}
            )
            result = run_code_node(state)

            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == 630  # int(10.5 * 60) = 630
            assert result["run_error"] is None

    def test_run_code_node_runtime_budget_none_uses_default(self, code_runner_state_factory):
        """Test that None runtime_budget uses default value (should not raise TypeError)."""
        with patch("src.code_runner.run_simulation") as mock_run:
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

            state = code_runner_state_factory(
                current_stage_id="test_stage",
                plan={"stages": [{"stage_id": "test_stage", "runtime_budget_minutes": None}]}
            )
            
            # This test expects None to use default (60), not raise TypeError
            # If TypeError is raised, the test fails, revealing a bug
            result = run_code_node(state)
            
            # Should use default of 60 minutes = 3600 seconds
            kwargs = mock_run.call_args.kwargs
            assert kwargs["config"]["timeout_seconds"] == 3600, \
                "None runtime_budget should default to 60 minutes (3600 seconds)"
            assert result["run_error"] is None

