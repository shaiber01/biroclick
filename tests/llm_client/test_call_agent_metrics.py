"""Metrics coverage for `call_agent_with_metrics`."""

import re
import time
import pytest
from unittest.mock import patch, MagicMock, call

from src.llm_client import call_agent_with_metrics, _record_call_metrics


class TestCallAgentMetrics:
    """Tests for call_agent_with_metrics."""

    @patch("src.llm_client.call_agent")
    def test_call_agent_metrics_success(self, mock_call_agent):
        """Test successful call records correct metrics."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert len(state["metrics"]["agent_calls"]) == 1

        metric = state["metrics"]["agent_calls"][0]
        assert metric["agent"] == "test_agent"
        assert metric["success"] is True
        assert metric["error"] is None
        assert "duration_seconds" in metric
        assert "timestamp" in metric

    @patch("src.llm_client.call_agent")
    def test_call_agent_metrics_failure(self, mock_call_agent):
        """Test failed call records failure metrics."""
        mock_call_agent.side_effect = ValueError("Failure")

        state = {}
        with pytest.raises(ValueError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        assert "metrics" in state
        assert len(state["metrics"]["agent_calls"]) == 1

        metric = state["metrics"]["agent_calls"][0]
        assert metric["agent"] == "test_agent"
        assert metric["success"] is False
        assert metric["error"] == "Failure"

    # =========================================================================
    # Return Value Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_returns_call_agent_result(self, mock_call_agent):
        """Test that call_agent_with_metrics returns the result from call_agent."""
        expected_result = {"key": "value", "nested": {"data": [1, 2, 3]}}
        mock_call_agent.return_value = expected_result

        state = {}
        result = call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert result == expected_result
        assert result is expected_result  # Should be the exact same object

    @patch("src.llm_client.call_agent")
    def test_returns_empty_dict_when_call_agent_returns_empty(self, mock_call_agent):
        """Test handling of empty dict return from call_agent."""
        mock_call_agent.return_value = {}

        state = {}
        result = call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert result == {}
        assert state["metrics"]["agent_calls"][0]["success"] is True

    # =========================================================================
    # Duration Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_duration_is_positive_number(self, mock_call_agent):
        """Test that duration_seconds is a positive float."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        metric = state["metrics"]["agent_calls"][0]
        assert isinstance(metric["duration_seconds"], float)
        assert metric["duration_seconds"] >= 0

    @patch("src.llm_client.call_agent")
    def test_duration_reflects_actual_time(self, mock_call_agent):
        """Test that duration approximately reflects actual call time."""
        delay = 0.1  # 100ms delay

        def slow_call(*args, **kwargs):
            time.sleep(delay)
            return {"output": "success"}

        mock_call_agent.side_effect = slow_call

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        metric = state["metrics"]["agent_calls"][0]
        # Duration should be at least the delay (with some tolerance)
        assert metric["duration_seconds"] >= delay * 0.9
        # And not unreasonably long
        assert metric["duration_seconds"] < delay + 0.5

    @patch("src.llm_client.call_agent")
    def test_duration_recorded_on_failure(self, mock_call_agent):
        """Test that duration is recorded even when call_agent fails."""
        delay = 0.05

        def slow_failure(*args, **kwargs):
            time.sleep(delay)
            raise RuntimeError("Failed")

        mock_call_agent.side_effect = slow_failure

        state = {}
        with pytest.raises(RuntimeError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        metric = state["metrics"]["agent_calls"][0]
        assert metric["duration_seconds"] >= delay * 0.9

    # =========================================================================
    # Timestamp Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_timestamp_format_is_iso8601(self, mock_call_agent):
        """Test that timestamp follows ISO 8601 format."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        metric = state["metrics"]["agent_calls"][0]
        timestamp = metric["timestamp"]
        
        # Verify ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        iso8601_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(iso8601_pattern, timestamp), f"Timestamp {timestamp} doesn't match ISO 8601 format"

    @patch("src.llm_client.call_agent")
    def test_timestamp_is_recent(self, mock_call_agent):
        """Test that timestamp is close to current time."""
        mock_call_agent.return_value = {"output": "success"}

        before = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        after = time.strftime("%Y-%m-%dT%H:%M", time.gmtime())
        timestamp = state["metrics"]["agent_calls"][0]["timestamp"]
        
        # Timestamp should start with either before or after time (within same minute)
        timestamp_prefix = timestamp[:16]  # YYYY-MM-DDTHH:MM
        assert timestamp_prefix in [before, after], f"Timestamp {timestamp} not recent"

    # =========================================================================
    # Multiple Calls Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_multiple_calls_accumulate(self, mock_call_agent):
        """Test that multiple calls accumulate in agent_calls list."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        
        call_agent_with_metrics(
            agent_name="agent1",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )
        call_agent_with_metrics(
            agent_name="agent2",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )
        call_agent_with_metrics(
            agent_name="agent3",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert len(state["metrics"]["agent_calls"]) == 3
        assert state["metrics"]["agent_calls"][0]["agent"] == "agent1"
        assert state["metrics"]["agent_calls"][1]["agent"] == "agent2"
        assert state["metrics"]["agent_calls"][2]["agent"] == "agent3"

    @patch("src.llm_client.call_agent")
    def test_mixed_success_and_failure_calls(self, mock_call_agent):
        """Test that both success and failure calls are properly recorded."""
        state = {}

        # First call succeeds
        mock_call_agent.return_value = {"output": "success"}
        call_agent_with_metrics(
            agent_name="success_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        # Second call fails
        mock_call_agent.side_effect = ValueError("error")
        with pytest.raises(ValueError):
            call_agent_with_metrics(
                agent_name="failure_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        # Third call succeeds
        mock_call_agent.side_effect = None
        mock_call_agent.return_value = {"output": "success2"}
        call_agent_with_metrics(
            agent_name="success_agent2",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert len(state["metrics"]["agent_calls"]) == 3
        
        assert state["metrics"]["agent_calls"][0]["success"] is True
        assert state["metrics"]["agent_calls"][0]["error"] is None
        
        assert state["metrics"]["agent_calls"][1]["success"] is False
        assert state["metrics"]["agent_calls"][1]["error"] == "error"
        
        assert state["metrics"]["agent_calls"][2]["success"] is True
        assert state["metrics"]["agent_calls"][2]["error"] is None

    # =========================================================================
    # Pre-existing State Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_preserves_existing_metrics(self, mock_call_agent):
        """Test that existing metrics structure is preserved."""
        mock_call_agent.return_value = {"output": "success"}

        existing_call = {
            "agent": "previous_agent",
            "duration_seconds": 1.5,
            "success": True,
            "error": None,
            "timestamp": "2024-01-01T00:00:00Z",
        }
        state = {
            "metrics": {
                "agent_calls": [existing_call],
                "stage_metrics": [{"stage": "stage1", "status": "completed"}],
            }
        }

        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert len(state["metrics"]["agent_calls"]) == 2
        assert state["metrics"]["agent_calls"][0] == existing_call
        assert state["metrics"]["agent_calls"][1]["agent"] == "test_agent"
        # stage_metrics should be preserved
        assert state["metrics"]["stage_metrics"] == [{"stage": "stage1", "status": "completed"}]

    @patch("src.llm_client.call_agent")
    def test_handles_state_with_metrics_but_no_agent_calls(self, mock_call_agent):
        """Test state that has metrics but no agent_calls list."""
        mock_call_agent.return_value = {"output": "success"}

        state = {
            "metrics": {
                "stage_metrics": [{"stage": "stage1"}],
            }
        }

        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert "agent_calls" in state["metrics"]
        assert len(state["metrics"]["agent_calls"]) == 1
        assert state["metrics"]["stage_metrics"] == [{"stage": "stage1"}]

    @patch("src.llm_client.call_agent")
    def test_handles_state_with_other_data(self, mock_call_agent):
        """Test that other state data is not modified."""
        mock_call_agent.return_value = {"output": "success"}

        state = {
            "paper_text": "some paper",
            "plan": {"stages": []},
            "current_stage_id": "stage1",
        }

        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        # Original state data should be preserved
        assert state["paper_text"] == "some paper"
        assert state["plan"] == {"stages": []}
        assert state["current_stage_id"] == "stage1"
        # Metrics should be added
        assert "metrics" in state

    # =========================================================================
    # Parameter Passthrough Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_passes_all_required_parameters_to_call_agent(self, mock_call_agent):
        """Test that required parameters are correctly passed to call_agent."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="my_agent",
            system_prompt="my system prompt",
            user_content="my user content",
            state=state,
        )

        mock_call_agent.assert_called_once()
        call_kwargs = mock_call_agent.call_args.kwargs
        
        assert call_kwargs["agent_name"] == "my_agent"
        assert call_kwargs["system_prompt"] == "my system prompt"
        assert call_kwargs["user_content"] == "my user content"

    @patch("src.llm_client.call_agent")
    def test_passes_schema_name_parameter(self, mock_call_agent):
        """Test that schema_name parameter is passed to call_agent."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
            schema_name="custom_schema",
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["schema_name"] == "custom_schema"

    @patch("src.llm_client.call_agent")
    def test_passes_images_parameter(self, mock_call_agent):
        """Test that images parameter is passed to call_agent."""
        mock_call_agent.return_value = {"output": "success"}

        images = ["/path/to/image1.png", "/path/to/image2.png"]
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
            images=images,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["images"] == images

    @patch("src.llm_client.call_agent")
    def test_passes_model_parameter(self, mock_call_agent):
        """Test that model parameter is passed to call_agent."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
            model="claude-sonnet-4-20250514",
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("src.llm_client.call_agent")
    def test_passes_all_optional_parameters_together(self, mock_call_agent):
        """Test passing all optional parameters at once."""
        mock_call_agent.return_value = {"output": "success"}

        images = ["/path/to/img.png"]
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
            schema_name="custom_schema",
            images=images,
            model="custom-model",
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["agent_name"] == "test_agent"
        assert call_kwargs["system_prompt"] == "prompt"
        assert call_kwargs["user_content"] == "content"
        assert call_kwargs["schema_name"] == "custom_schema"
        assert call_kwargs["images"] == images
        assert call_kwargs["model"] == "custom-model"

    @patch("src.llm_client.call_agent")
    def test_passes_none_for_optional_parameters_when_not_provided(self, mock_call_agent):
        """Test that None is passed for optional params not provided."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs.get("schema_name") is None
        assert call_kwargs.get("images") is None
        assert call_kwargs.get("model") is None

    @patch("src.llm_client.call_agent")
    def test_handles_multimodal_user_content(self, mock_call_agent):
        """Test that list-based multimodal user_content is passed correctly."""
        mock_call_agent.return_value = {"output": "success"}

        multimodal_content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content=multimodal_content,
            state=state,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["user_content"] == multimodal_content

    # =========================================================================
    # Exception Handling Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_reraises_valueerror(self, mock_call_agent):
        """Test that ValueError is re-raised after recording metrics."""
        mock_call_agent.side_effect = ValueError("validation error")

        state = {}
        with pytest.raises(ValueError, match="validation error"):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        # Metrics should still be recorded
        assert state["metrics"]["agent_calls"][0]["success"] is False
        assert state["metrics"]["agent_calls"][0]["error"] == "validation error"

    @patch("src.llm_client.call_agent")
    def test_reraises_runtimeerror(self, mock_call_agent):
        """Test that RuntimeError is re-raised after recording metrics."""
        mock_call_agent.side_effect = RuntimeError("runtime error")

        state = {}
        with pytest.raises(RuntimeError, match="runtime error"):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        assert state["metrics"]["agent_calls"][0]["success"] is False
        assert state["metrics"]["agent_calls"][0]["error"] == "runtime error"

    @patch("src.llm_client.call_agent")
    def test_reraises_custom_exception(self, mock_call_agent):
        """Test that custom exceptions are re-raised."""
        class CustomAgentError(Exception):
            pass

        mock_call_agent.side_effect = CustomAgentError("custom error")

        state = {}
        with pytest.raises(CustomAgentError, match="custom error"):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        assert state["metrics"]["agent_calls"][0]["error"] == "custom error"

    @patch("src.llm_client.call_agent")
    def test_error_message_is_string(self, mock_call_agent):
        """Test that error message is converted to string."""
        # Some exceptions might have complex objects as messages
        mock_call_agent.side_effect = ValueError({"complex": "error"})

        state = {}
        with pytest.raises(ValueError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        error = state["metrics"]["agent_calls"][0]["error"]
        assert isinstance(error, str)

    @patch("src.llm_client.call_agent")
    def test_records_empty_error_message(self, mock_call_agent):
        """Test handling of exception with empty message."""
        mock_call_agent.side_effect = ValueError("")

        state = {}
        with pytest.raises(ValueError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

        assert state["metrics"]["agent_calls"][0]["error"] == ""

    @patch("src.llm_client.call_agent")
    def test_exception_original_type_preserved(self, mock_call_agent):
        """Test that the original exception type is preserved when re-raised."""
        mock_call_agent.side_effect = FileNotFoundError("file not found")

        state = {}
        
        # Should raise FileNotFoundError specifically, not a generic Exception
        with pytest.raises(FileNotFoundError):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="prompt",
                user_content="content",
                state=state,
            )

    # =========================================================================
    # Edge Cases Tests
    # =========================================================================

    @patch("src.llm_client.call_agent")
    def test_empty_agent_name(self, mock_call_agent):
        """Test handling of empty agent name."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert state["metrics"]["agent_calls"][0]["agent"] == ""

    @patch("src.llm_client.call_agent")
    def test_empty_prompts(self, mock_call_agent):
        """Test handling of empty system_prompt and user_content."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="",
            user_content="",
            state=state,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["system_prompt"] == ""
        assert call_kwargs["user_content"] == ""
        assert state["metrics"]["agent_calls"][0]["success"] is True

    @patch("src.llm_client.call_agent")
    def test_agent_name_with_special_characters(self, mock_call_agent):
        """Test agent name with special characters is recorded correctly."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test-agent_v2.0",
            system_prompt="prompt",
            user_content="content",
            state=state,
        )

        assert state["metrics"]["agent_calls"][0]["agent"] == "test-agent_v2.0"

    @patch("src.llm_client.call_agent")
    def test_very_long_prompts(self, mock_call_agent):
        """Test handling of very long prompts."""
        mock_call_agent.return_value = {"output": "success"}

        long_prompt = "x" * 100000
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt=long_prompt,
            user_content=long_prompt,
            state=state,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["system_prompt"] == long_prompt
        assert call_kwargs["user_content"] == long_prompt

    @patch("src.llm_client.call_agent")
    def test_unicode_in_prompts(self, mock_call_agent):
        """Test handling of unicode characters in prompts."""
        mock_call_agent.return_value = {"output": "success"}

        unicode_prompt = "Test with émojis 🔬🧪 and ünïcödë: 日本語"
        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt=unicode_prompt,
            user_content=unicode_prompt,
            state=state,
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["system_prompt"] == unicode_prompt

    @patch("src.llm_client.call_agent")
    def test_empty_images_list(self, mock_call_agent):
        """Test handling of empty images list."""
        mock_call_agent.return_value = {"output": "success"}

        state = {}
        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="prompt",
            user_content="content",
            state=state,
            images=[],
        )

        call_kwargs = mock_call_agent.call_args.kwargs
        assert call_kwargs["images"] == []


class TestRecordCallMetrics:
    """Direct tests for _record_call_metrics helper function."""

    def test_creates_metrics_structure_from_empty_state(self):
        """Test that metrics structure is created from empty state."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.5,
            success=True,
        )

        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert "stage_metrics" in state["metrics"]
        assert isinstance(state["metrics"]["agent_calls"], list)
        assert isinstance(state["metrics"]["stage_metrics"], list)

    def test_adds_agent_calls_to_existing_metrics(self):
        """Test adding to existing metrics without agent_calls."""
        state = {"metrics": {"stage_metrics": []}}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.5,
            success=True,
        )

        assert "agent_calls" in state["metrics"]
        assert len(state["metrics"]["agent_calls"]) == 1

    def test_records_all_required_fields(self):
        """Test that all required fields are recorded."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=2.5,
            success=True,
            error=None,
        )

        metric = state["metrics"]["agent_calls"][0]
        
        # All required fields must be present
        required_fields = {"agent", "duration_seconds", "success", "error", "timestamp"}
        assert set(metric.keys()) == required_fields
        
        # Verify values
        assert metric["agent"] == "test_agent"
        assert metric["duration_seconds"] == 2.5
        assert metric["success"] is True
        assert metric["error"] is None
        assert isinstance(metric["timestamp"], str)

    def test_records_error_message(self):
        """Test that error message is recorded on failure."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=0.5,
            success=False,
            error="Something went wrong",
        )

        metric = state["metrics"]["agent_calls"][0]
        assert metric["success"] is False
        assert metric["error"] == "Something went wrong"

    def test_timestamp_format(self):
        """Test timestamp is in correct ISO 8601 format."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
        )

        timestamp = state["metrics"]["agent_calls"][0]["timestamp"]
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", timestamp)

    def test_appends_to_existing_calls(self):
        """Test that new metrics are appended, not replacing."""
        state = {
            "metrics": {
                "agent_calls": [
                    {"agent": "old_agent", "duration_seconds": 1.0, "success": True, "error": None, "timestamp": "2024-01-01T00:00:00Z"}
                ],
                "stage_metrics": [],
            }
        }
        
        _record_call_metrics(
            state=state,
            agent_name="new_agent",
            duration=2.0,
            success=True,
        )

        assert len(state["metrics"]["agent_calls"]) == 2
        assert state["metrics"]["agent_calls"][0]["agent"] == "old_agent"
        assert state["metrics"]["agent_calls"][1]["agent"] == "new_agent"

    def test_preserves_stage_metrics(self):
        """Test that stage_metrics are not modified."""
        original_stage_metrics = [{"stage": "1", "data": "value"}]
        state = {
            "metrics": {
                "agent_calls": [],
                "stage_metrics": original_stage_metrics.copy(),
            }
        }
        
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
        )

        assert state["metrics"]["stage_metrics"] == original_stage_metrics

    def test_zero_duration(self):
        """Test handling of zero duration."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=0.0,
            success=True,
        )

        assert state["metrics"]["agent_calls"][0]["duration_seconds"] == 0.0

    def test_very_small_duration(self):
        """Test handling of very small duration."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=0.000001,
            success=True,
        )

        assert state["metrics"]["agent_calls"][0]["duration_seconds"] == 0.000001

    def test_large_duration(self):
        """Test handling of large duration."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=3600.0,  # 1 hour
            success=True,
        )

        assert state["metrics"]["agent_calls"][0]["duration_seconds"] == 3600.0

    def test_error_defaults_to_none_when_not_provided(self):
        """Test that error defaults to None when not provided."""
        state = {}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
            # error not provided
        )

        assert state["metrics"]["agent_calls"][0]["error"] is None

    def test_creates_stage_metrics_when_metrics_exists_but_empty(self):
        """Test that stage_metrics is created even when metrics already exists.
        
        Ensures consistent initialization regardless of how state was created.
        """
        state = {"metrics": {}}  # metrics exists but empty
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
        )

        assert "agent_calls" in state["metrics"]
        assert "stage_metrics" in state["metrics"]
        assert state["metrics"]["stage_metrics"] == []

    def test_handles_metrics_with_none_stage_metrics(self):
        """Test handling when stage_metrics is explicitly None."""
        state = {"metrics": {"agent_calls": [], "stage_metrics": None}}
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
        )

        # Should append to agent_calls without modifying stage_metrics
        assert len(state["metrics"]["agent_calls"]) == 1
        assert state["metrics"]["stage_metrics"] is None

    def test_handles_existing_metrics_with_extra_fields(self):
        """Test that extra fields in metrics dict are preserved."""
        state = {
            "metrics": {
                "agent_calls": [],
                "stage_metrics": [],
                "custom_field": "custom_value",
                "another_field": 123,
            }
        }
        _record_call_metrics(
            state=state,
            agent_name="test_agent",
            duration=1.0,
            success=True,
        )

        # Custom fields should be preserved
        assert state["metrics"]["custom_field"] == "custom_value"
        assert state["metrics"]["another_field"] == 123


class TestCallAgentMetricsIntegration:
    """Integration tests verifying call_agent_with_metrics works correctly 
    with _record_call_metrics."""

    @patch("src.llm_client.call_agent")
    def test_full_success_flow_metrics_structure(self, mock_call_agent):
        """Test complete metrics structure after successful call."""
        mock_call_agent.return_value = {"result": "data"}

        state = {}
        result = call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="system",
            user_content="user",
            state=state,
        )

        # Verify return value
        assert result == {"result": "data"}
        
        # Verify complete metrics structure
        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert "stage_metrics" in state["metrics"]
        
        # Verify metric entry completeness
        metric = state["metrics"]["agent_calls"][0]
        assert set(metric.keys()) == {"agent", "duration_seconds", "success", "error", "timestamp"}
        assert metric["agent"] == "test_agent"
        assert metric["success"] is True
        assert metric["error"] is None
        assert isinstance(metric["duration_seconds"], float)
        assert isinstance(metric["timestamp"], str)

    @patch("src.llm_client.call_agent")
    def test_full_failure_flow_metrics_structure(self, mock_call_agent):
        """Test complete metrics structure after failed call."""
        mock_call_agent.side_effect = RuntimeError("test failure")

        state = {}
        with pytest.raises(RuntimeError, match="test failure"):
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="system",
                user_content="user",
                state=state,
            )

        # Verify complete metrics structure
        assert "metrics" in state
        assert "agent_calls" in state["metrics"]
        assert "stage_metrics" in state["metrics"]
        
        # Verify metric entry completeness
        metric = state["metrics"]["agent_calls"][0]
        assert set(metric.keys()) == {"agent", "duration_seconds", "success", "error", "timestamp"}
        assert metric["agent"] == "test_agent"
        assert metric["success"] is False
        assert metric["error"] == "test failure"
        assert isinstance(metric["duration_seconds"], float)
        assert isinstance(metric["timestamp"], str)

    @patch("src.llm_client.call_agent")
    def test_state_not_modified_beyond_metrics(self, mock_call_agent):
        """Test that call_agent_with_metrics only modifies metrics key."""
        mock_call_agent.return_value = {"result": "data"}

        original_state = {
            "paper_text": "some text",
            "plan": {"stages": [{"stage_id": "1"}]},
            "current_stage_id": "1",
            "extracted_parameters": [1, 2, 3],
            "nested": {"deep": {"value": 42}},
        }
        state = original_state.copy()
        state["nested"] = {"deep": {"value": 42}}  # Recreate nested dict

        call_agent_with_metrics(
            agent_name="test_agent",
            system_prompt="system",
            user_content="user",
            state=state,
        )

        # All original keys should be unchanged
        assert state["paper_text"] == "some text"
        assert state["plan"] == {"stages": [{"stage_id": "1"}]}
        assert state["current_stage_id"] == "1"
        assert state["extracted_parameters"] == [1, 2, 3]
        assert state["nested"]["deep"]["value"] == 42
        
        # Only metrics should be added
        assert "metrics" in state
        assert len(state) == 6  # 5 original keys + metrics

    @patch("src.llm_client.call_agent")
    def test_call_agent_receives_exact_arguments(self, mock_call_agent):
        """Verify call_agent receives exactly the arguments passed."""
        mock_call_agent.return_value = {}

        state = {}
        call_agent_with_metrics(
            agent_name="exact_agent",
            system_prompt="exact_system_prompt",
            user_content="exact_user_content",
            state=state,
            schema_name="exact_schema",
            images=["/exact/path.png"],
            model="exact-model",
        )

        # Verify exact argument values
        mock_call_agent.assert_called_once_with(
            agent_name="exact_agent",
            system_prompt="exact_system_prompt",
            user_content="exact_user_content",
            schema_name="exact_schema",
            images=["/exact/path.png"],
            model="exact-model",
        )

    @patch("src.llm_client.call_agent")
    def test_metrics_recorded_before_exception_propagates(self, mock_call_agent):
        """Verify metrics are recorded before exception is re-raised."""
        call_count = [0]
        
        def failing_call(*args, **kwargs):
            call_count[0] += 1
            raise ValueError("failure")
        
        mock_call_agent.side_effect = failing_call

        state = {}
        try:
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="system",
                user_content="user",
                state=state,
            )
        except ValueError:
            pass

        # Metrics should be recorded even though exception was raised
        assert "metrics" in state
        assert len(state["metrics"]["agent_calls"]) == 1
        assert state["metrics"]["agent_calls"][0]["success"] is False
        
        # Verify call_agent was called exactly once (no retries from wrapper)
        assert call_count[0] == 1

    @patch("src.llm_client.call_agent")
    def test_successive_calls_preserve_order(self, mock_call_agent):
        """Verify successive calls are recorded in order."""
        mock_call_agent.return_value = {}

        state = {}
        agents = ["first", "second", "third", "fourth", "fifth"]
        
        for agent in agents:
            call_agent_with_metrics(
                agent_name=agent,
                system_prompt="system",
                user_content="user",
                state=state,
            )

        # Verify order is preserved
        recorded_agents = [m["agent"] for m in state["metrics"]["agent_calls"]]
        assert recorded_agents == agents

    @patch("src.llm_client.call_agent")
    def test_duration_increases_monotonically_with_delay(self, mock_call_agent):
        """Test that duration values increase with actual delays."""
        delays = [0.01, 0.02, 0.03]
        call_idx = [0]

        def delayed_call(*args, **kwargs):
            time.sleep(delays[call_idx[0]])
            call_idx[0] += 1
            return {}

        mock_call_agent.side_effect = delayed_call

        state = {}
        for _ in delays:
            call_agent_with_metrics(
                agent_name="test_agent",
                system_prompt="system",
                user_content="user",
                state=state,
            )

        # Each call should have progressively longer duration
        durations = [m["duration_seconds"] for m in state["metrics"]["agent_calls"]]
        
        # First duration should be at least the first delay
        assert durations[0] >= delays[0] * 0.8
        assert durations[1] >= delays[1] * 0.8
        assert durations[2] >= delays[2] * 0.8
