"""Integration tests that verify nodes mutate state as expected."""

from unittest.mock import patch, MagicMock
import pytest

from schemas.state import (
    create_initial_state, 
    MAX_REPLANS, 
    MAX_DESIGN_REVISIONS, 
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
)


class TestStateMutations:
    """Test that nodes mutate state correctly."""

    # ═══════════════════════════════════════════════════════════════════════
    # plan_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_plan_node_sets_plan_field(self):
        """plan_node must set state['plan'] with the LLM response."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "test",
            "title": "Gold Nanorod Study",
            "summary": "Study of gold nanorods",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
            "extracted_parameters": [{"name": "length", "value": 100, "unit": "nm"}],
            "paper_domain": "plasmonics",
            "planned_materials": [{"name": "gold", "source": "palik"}],
            "assumptions": {"global_assumptions": ["Assumption 1"]},
        }
        paper_text = (
            "We study the optical properties of gold nanorods with length 100nm and "
            "diameter 40nm. Using FDTD simulations, we calculate extinction spectra "
            "and near-field enhancement patterns. The localized surface plasmon "
            "resonance is observed at 650nm wavelength."
        )
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Verify plan is set with all fields
        assert "plan" in result, "plan_node must return 'plan' in result"
        assert result["plan"]["title"] == "Gold Nanorod Study"
        assert result["plan"]["paper_id"] == "test"
        assert result["plan"]["summary"] == "Study of gold nanorods"
        assert len(result["plan"]["stages"]) == 1
        assert result["plan"]["stages"][0]["stage_id"] == "s1"
        assert result["plan"]["stages"][0]["stage_type"] == "MATERIAL_VALIDATION"
        assert result["plan"]["stages"][0]["targets"] == ["Fig1"]
        assert result["plan"]["stages"][0]["dependencies"] == []
        assert len(result["plan"]["targets"]) == 1
        assert result["plan"]["targets"][0]["figure_id"] == "Fig1"
        
        # Verify other state fields are set
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "planning"
        assert "paper_domain" in result
        assert result["paper_domain"] == "plasmonics"
        assert "planned_materials" in result
        assert len(result["planned_materials"]) == 1
        assert result["planned_materials"][0]["name"] == "gold"
        assert result["planned_materials"][0]["source"] == "palik"
        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        assert "Assumption 1" in result["assumptions"]["global_assumptions"]
        
        # Verify progress is initialized
        assert "progress" in result
        assert "extracted_parameters" in result

    def test_plan_node_handles_missing_llm_fields(self):
        """plan_node must handle missing fields in LLM response gracefully."""
        from src.agents.planning import plan_node

        # LLM returns minimal response
        mock_plan = {
            "title": "Test",
            "stages": [],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should still set plan with defaults for missing fields
        assert "plan" in result
        assert result["plan"]["title"] == "Test"
        assert result["plan"].get("stages") == []
        assert result["plan"].get("targets") == []
        assert result["plan"].get("extracted_parameters") == []
        assert result["plan"].get("summary") == ""
        assert result["paper_domain"] == "other"  # Default when missing
        assert result["planned_materials"] == []
        assert result["assumptions"] == {}

    def test_plan_node_preserves_existing_paper_id(self):
        """plan_node must preserve paper_id from state if valid."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "different_id",
            "title": "Test",
            "stages": [],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("original_id", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should preserve original paper_id
        assert result["plan"]["paper_id"] == "original_id"

    def test_plan_node_uses_llm_paper_id_when_state_is_unknown(self):
        """plan_node must use LLM paper_id when state has 'unknown'."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "llm_detected_id",
            "title": "Test",
            "stages": [],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("unknown", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should use LLM's paper_id when state paper_id is "unknown"
        assert result["plan"]["paper_id"] == "llm_detected_id"

    def test_plan_node_rejects_short_paper_text(self):
        """plan_node must reject paper text that's too short."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "short", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert result["workflow_phase"] == "planning"

    def test_plan_node_rejects_empty_paper_text(self):
        """plan_node must reject empty paper text."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result
        assert "pending_user_questions" in result
        assert any("0 characters" in q or "missing" in q.lower() for q in result["pending_user_questions"])

    def test_plan_node_rejects_whitespace_only_paper_text(self):
        """plan_node must reject whitespace-only paper text."""
        from src.agents.planning import plan_node

        state = create_initial_state("test", "   \n\t   ", "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_node_rejects_boundary_length_paper_text(self):
        """plan_node must reject paper text exactly at boundary (99 chars)."""
        from src.agents.planning import plan_node

        # Exactly 99 characters
        short_text = "a" * 99
        state = create_initial_state("test", short_text, "plasmonics")
        result = plan_node(state)

        assert result.get("ask_user_trigger") == "missing_paper_text"
        assert result.get("awaiting_user_input") is True
        assert "plan" not in result

    def test_plan_node_accepts_boundary_length_paper_text(self):
        """plan_node must accept paper text at minimum boundary (100 chars)."""
        from src.agents.planning import plan_node

        # Exactly 100 characters
        valid_text = "a" * 100
        mock_plan = {
            "paper_id": "test",
            "title": "Test",
            "stages": [],
        }
        state = create_initial_state("test", valid_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        assert "plan" in result
        assert "ask_user_trigger" not in result

    def test_plan_node_handles_progress_initialization_failure(self):
        """plan_node must handle progress initialization failures."""
        from src.agents.planning import plan_node
        from schemas.state import initialize_progress_from_plan

        # Use a plan structure that will cause initialize_progress_from_plan to raise an exception
        # We'll patch initialize_progress_from_plan to raise an exception to test the error handling
        mock_plan = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION"}],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                # Patch initialize_progress_from_plan to raise an exception
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(state)

        # Should mark plan for revision when progress initialization fails
        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert "Progress initialization failed" in result["planner_feedback"]
        assert "Invalid plan structure" in result["planner_feedback"]
        assert "replan_count" in result
        assert result["replan_count"] == 1  # Should increment from 0 to 1

    def test_plan_node_handles_llm_exception(self):
        """plan_node must escalate to user when LLM call fails."""
        from src.agents.planning import plan_node

        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=Exception("API rate limit exceeded"),
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should escalate to user on LLM failure
        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        assert any("failed" in q.lower() for q in result["pending_user_questions"])

    def test_plan_node_handles_context_escalation(self):
        """plan_node must handle context escalation."""
        from src.agents.planning import plan_node

        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        escalation_result = {
            "ask_user_trigger": "context_issue",
            "pending_user_questions": ["Context check failed"],
            "awaiting_user_input": True,
        }

        with patch(
            "src.agents.planning.check_context_or_escalate",
            return_value=escalation_result,
        ):
            result = plan_node(state)

        # Should return the escalation result
        assert result.get("ask_user_trigger") == "context_issue"
        assert result.get("awaiting_user_input") is True

    def test_plan_node_adds_replan_context_to_prompt(self):
        """plan_node must add replan context when replan_count > 0."""
        from src.agents.planning import plan_node

        mock_plan = {"title": "Test", "stages": []}
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")
        state["replan_count"] = 2

        captured_prompt = {}
        
        def capture_call(*args, **kwargs):
            captured_prompt["system_prompt"] = kwargs.get("system_prompt", args[1] if len(args) > 1 else "")
            return mock_plan

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        # Should mention replan attempt in system prompt
        assert "Replan Attempt #2" in captured_prompt["system_prompt"]

    def test_plan_node_handles_none_paper_domain_from_llm(self):
        """plan_node must use 'other' when LLM returns None paper_domain."""
        from src.agents.planning import plan_node

        mock_plan = {
            "title": "Test",
            "stages": [],
            "paper_domain": None,  # Explicitly None
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                result = plan_node(state)

        assert result["paper_domain"] == "other"
        assert result["plan"]["paper_domain"] == "other"

    def test_plan_node_progress_initialization_respects_max_replans(self):
        """plan_node must not increment replan_count beyond max when progress init fails."""
        from src.agents.planning import plan_node

        mock_plan = {
            "paper_id": "test",
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION"}],
        }
        paper_text = "paper text " * 20
        state = create_initial_state("test", paper_text, "plasmonics")
        state["replan_count"] = MAX_REPLANS  # Already at max

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_plan,
        ):
            with patch("src.agents.planning.check_context_or_escalate", return_value=None):
                with patch(
                    "src.agents.planning.initialize_progress_from_plan",
                    side_effect=ValueError("Invalid plan structure"),
                ):
                    result = plan_node(state)

        # Should not exceed max
        assert result["replan_count"] == MAX_REPLANS

    # ═══════════════════════════════════════════════════════════════════════
    # plan_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_plan_reviewer_sets_verdict_field(self):
        """plan_reviewer_node must set last_plan_review_verdict."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
        }
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good", "feedback": "Looks good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert "last_plan_review_verdict" in result
        assert result["last_plan_review_verdict"] == "approve"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "plan_review"
        # Should not increment replan_count on approve
        assert "replan_count" not in result or result.get("replan_count") == state.get("replan_count", 0)
        # Should not set planner_feedback on approve
        assert "planner_feedback" not in result

    def test_plan_reviewer_increments_replan_count_on_needs_revision(self):
        """plan_reviewer_node must increment replan_count when verdict is needs_revision."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
        }
        state["replan_count"] = 1
        # Set max_replans higher than current count to allow incrementing
        state["runtime_config"] = {"max_replans": 5}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs work",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "replan_count" in result
        # Should increment from 1 to 2 for LLM rejections
        assert result["replan_count"] == 2, (
            f"BUG: replan_count should be 2 (was 1), but got {result['replan_count']}. "
            "plan_reviewer_node is not incrementing replan_count for LLM rejections."
        )
        assert "planner_feedback" in result
        assert result["planner_feedback"] == "Fix issues"

    def test_plan_reviewer_respects_max_replans(self):
        """plan_reviewer_node must not increment replan_count beyond max_replans."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Validate gold optical constants",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
            "targets": [{"figure_id": "Fig1", "description": "Test"}],
        }
        state["replan_count"] = MAX_REPLANS
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs work",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert result["replan_count"] == MAX_REPLANS  # Should not exceed max

    def test_plan_reviewer_escalates_at_max_replans(self):
        """plan_reviewer_node must escalate to user when max_replans is reached."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test Plan",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                }
            ],
        }
        state["replan_count"] = MAX_REPLANS - 1  # One below max
        state["runtime_config"] = {"max_replans": MAX_REPLANS}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs work",
            "feedback": "Fix the structure issue",
        }

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        # After incrementing, we hit the max - should escalate
        assert result["replan_count"] == MAX_REPLANS
        assert result.get("ask_user_trigger") == "replan_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        # Should include feedback and options in the question
        assert any("APPROVE_PLAN" in q for q in result["pending_user_questions"])
        assert any("GUIDANCE" in q for q in result["pending_user_questions"])
        assert any("STOP" in q for q in result["pending_user_questions"])

    def test_plan_reviewer_rejects_empty_plan(self):
        """plan_reviewer_node must reject plans with no stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": [], "targets": []}

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result
        assert len(result.get("planner_feedback", "")) > 0
        assert "at least one stage" in result["planner_feedback"].lower()

    def test_plan_reviewer_rejects_none_plan(self):
        """plan_reviewer_node must handle None plan."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = None

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result

    def test_plan_reviewer_rejects_plan_with_none_stages(self):
        """plan_reviewer_node must reject plans with None stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {"title": "Test", "stages": None, "targets": []}

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "planner_feedback" in result

    def test_plan_reviewer_rejects_plan_with_missing_stage_id(self):
        """plan_reviewer_node must reject plans with stages missing stage_id."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_type": "MATERIAL_VALIDATION",
                    "description": "Test",
                    "targets": ["Fig1"],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "PLAN_ISSUE" in result.get("planner_feedback", "")
        assert "stage_id" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_duplicate_stage_ids(self):
        """plan_reviewer_node must reject plans with duplicate stage IDs."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_0",  # Duplicate
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": [],
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "duplicate" in result.get("planner_feedback", "").lower()
        assert "stage_0" in result.get("planner_feedback", "")

    def test_plan_reviewer_rejects_plan_with_missing_dependencies(self):
        """plan_reviewer_node must reject plans with dependencies on non-existent stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],  # stage_0 doesn't exist
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "missing stage" in result.get("planner_feedback", "").lower()
        assert "stage_0" in result.get("planner_feedback", "")

    def test_plan_reviewer_rejects_plan_with_circular_dependencies(self):
        """plan_reviewer_node must reject plans with circular dependencies."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_1"],  # Circular
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],  # Circular
                },
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_three_stage_circular_dependency(self):
        """plan_reviewer_node must detect circular dependencies involving 3+ stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {"stage_id": "A", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": ["C"]},
                {"stage_id": "B", "stage_type": "SINGLE_STRUCTURE", "targets": ["Fig2"], "dependencies": ["A"]},
                {"stage_id": "C", "stage_type": "PARAMETER_SWEEP", "targets": ["Fig3"], "dependencies": ["B"]},
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "circular" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_self_dependency(self):
        """plan_reviewer_node must reject plans with self-dependencies."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": ["stage_0"],  # Self-dependency
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "depends on itself" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_rejects_plan_with_stages_missing_targets(self):
        """plan_reviewer_node must reject plans with stages that have no targets."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],  # No targets
                    "dependencies": [],
                }
            ],
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "no targets" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_accepts_stage_with_target_details(self):
        """plan_reviewer_node must accept stages with target_details instead of targets."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": [],  # Empty targets
                    "target_details": [{"figure_id": "Fig1", "description": "Test"}],  # But has target_details
                    "dependencies": [],
                }
            ],
            "targets": [],
        }
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        # Should call LLM since there's no structural issue
        assert result["last_plan_review_verdict"] == "approve"

    def test_plan_reviewer_rejects_plan_with_invalid_stage_structure(self):
        """plan_reviewer_node must reject plans with invalid stage structures."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": ["not_a_dict", {"stage_id": "stage_0"}],  # Invalid structure
            "targets": [],
        }

        result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"
        assert "invalid stage structure" in result.get("planner_feedback", "").lower()

    def test_plan_reviewer_normalizes_verdict_pass_to_approve(self):
        """plan_reviewer_node must normalize 'pass' verdict to 'approve'."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        mock_response = {"verdict": "pass", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "approve"

    def test_plan_reviewer_normalizes_verdict_reject_to_needs_revision(self):
        """plan_reviewer_node must normalize 'reject' verdict to 'needs_revision'."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        state["runtime_config"] = {"max_replans": 5}
        mock_response = {"verdict": "reject", "issues": [], "summary": "Bad", "feedback": "Issues"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_handles_unknown_verdict(self):
        """plan_reviewer_node must default to needs_revision for unknown verdicts."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        state["runtime_config"] = {"max_replans": 5}
        mock_response = {"verdict": "maybe_good", "issues": [], "summary": "Uncertain"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        # Unknown verdict should default to needs_revision (safer)
        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_handles_missing_verdict(self):
        """plan_reviewer_node must handle missing verdict field."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        state["runtime_config"] = {"max_replans": 5}
        mock_response = {"issues": [], "summary": "No verdict given"}  # No verdict field

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        # Missing verdict should default to needs_revision
        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_handles_llm_exception(self):
        """plan_reviewer_node must auto-reject when LLM fails."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        state["runtime_config"] = {"max_replans": 5}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            result = plan_reviewer_node(state)

        # Should auto-reject with needs_revision on LLM failure
        assert result["last_plan_review_verdict"] == "needs_revision"

    def test_plan_reviewer_includes_assumptions_in_user_content(self):
        """plan_reviewer_node must include assumptions in the review content."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [{"stage_id": "s1", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"], "dependencies": []}],
        }
        state["assumptions"] = {"global_assumptions": ["Assume plane wave incidence"]}
        
        captured_content = {}
        
        def capture_call(*args, **kwargs):
            captured_content["user_content"] = kwargs.get("user_content", args[2] if len(args) > 2 else "")
            return {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = plan_reviewer_node(state)

        assert "ASSUMPTIONS" in captured_content["user_content"]
        assert "plane wave incidence" in captured_content["user_content"]

    def test_plan_reviewer_handles_none_dependencies(self):
        """plan_reviewer_node must handle None dependencies in stages."""
        from src.agents.planning import plan_reviewer_node

        state = create_initial_state("test", "paper about nanorods" * 20, "plasmonics")
        state["plan"] = {
            "title": "Test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": None,  # None instead of empty list
                }
            ],
            "targets": [],
        }
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.planning.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = plan_reviewer_node(state)

        # Should handle None dependencies gracefully
        assert result["last_plan_review_verdict"] == "approve"

    # ═══════════════════════════════════════════════════════════════════════
    # design_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_counter_increments_on_revision(self):
        """Revision counters must increment when verdict is needs_revision."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["design_revision_count"] = 0
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "problem"}],
            "summary": "Fix",
            "feedback": "Needs improvement",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] == 1, "Counter should increment on needs_revision"
        assert "last_design_review_verdict" in result
        assert result["last_design_review_verdict"] == "needs_revision"
        assert "reviewer_feedback" in result
        assert result["reviewer_feedback"] == "Needs improvement"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design_review"

    def test_design_reviewer_counter_does_not_increment_on_approve(self):
        """design_reviewer_node must not increment counter when verdict is approve."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["design_revision_count"] = 5
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Good",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["design_revision_count"] == 5, "Counter should not increment on approve"
        assert result["last_design_review_verdict"] == "approve"
        assert "reviewer_feedback" not in result

    def test_design_reviewer_respects_max_revisions(self):
        """design_reviewer_node must not increment counter beyond max_design_revisions."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "problem"}],
            "summary": "Fix",
            "feedback": "Still needs work",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS, "Counter should not exceed max"

    def test_design_reviewer_escalates_at_max_revisions(self):
        """design_reviewer_node must escalate to user when max_design_revisions is reached."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["design_revision_count"] = MAX_DESIGN_REVISIONS - 1  # One below max
        state["runtime_config"] = {"max_design_revisions": MAX_DESIGN_REVISIONS}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "problem"}],
            "summary": "Fix",
            "feedback": "Needs more work",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        # After incrementing, we hit the max - should escalate
        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS
        assert result.get("ask_user_trigger") == "design_review_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0
        # Should include options in the question
        assert any("PROVIDE_HINT" in q for q in result["pending_user_questions"])
        assert any("SKIP" in q for q in result["pending_user_questions"])
        assert any("STOP" in q for q in result["pending_user_questions"])

    def test_design_reviewer_handles_missing_verdict(self):
        """design_reviewer_node must handle missing verdict gracefully."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["runtime_config"] = {"max_design_revisions": 5}
        mock_response = {
            "issues": [],
            "summary": "No verdict",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        # Missing verdict should default to needs_revision (safer)
        assert result["last_design_review_verdict"] == "needs_revision"

    def test_design_reviewer_includes_reviewer_issues(self):
        """design_reviewer_node must include reviewer_issues in result."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        mock_response = {
            "verdict": "approve",
            "issues": [{"severity": "minor", "description": "Minor issue"}],
            "summary": "Good",
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) == 1
        assert result["reviewer_issues"][0]["severity"] == "minor"
        assert result["reviewer_issues"][0]["description"] == "Minor issue"

    def test_design_reviewer_normalizes_verdict_pass_to_approve(self):
        """design_reviewer_node must normalize 'pass' verdict to 'approve'."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        mock_response = {"verdict": "pass", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["last_design_review_verdict"] == "approve"

    def test_design_reviewer_normalizes_verdict_reject_to_needs_revision(self):
        """design_reviewer_node must normalize 'reject' verdict to 'needs_revision'."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["runtime_config"] = {"max_design_revisions": 5}
        mock_response = {"verdict": "reject", "issues": [], "summary": "Bad", "feedback": "Fix it"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["last_design_review_verdict"] == "needs_revision"
        assert result["reviewer_feedback"] == "Fix it"

    def test_design_reviewer_handles_unknown_verdict(self):
        """design_reviewer_node must default to needs_revision for unknown verdicts."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["runtime_config"] = {"max_design_revisions": 5}
        mock_response = {"verdict": "maybe_okay", "issues": [], "summary": "Uncertain"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        # Unknown verdict should default to needs_revision (safer)
        assert result["last_design_review_verdict"] == "needs_revision"

    def test_design_reviewer_handles_llm_exception(self):
        """design_reviewer_node must auto-reject when LLM fails."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"stage_id": "stage_0", "geometry": "nanorod"}
        state["runtime_config"] = {"max_design_revisions": 5}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            result = design_reviewer_node(state)

        # Should auto-reject with needs_revision on LLM failure
        assert result["last_design_review_verdict"] == "needs_revision"

    def test_design_reviewer_handles_string_design(self):
        """design_reviewer_node must handle string design_description."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = "A nanorod with length 100nm and diameter 40nm"
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["last_design_review_verdict"] == "approve"

    def test_design_reviewer_includes_plan_stage_in_review(self):
        """design_reviewer_node must include plan stage spec in user content."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"geometry": "nanorod"}
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        
        captured_content = {}
        
        def capture_call(*args, **kwargs):
            captured_content["user_content"] = kwargs.get("user_content", "")
            return {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = design_reviewer_node(state)

        assert "PLAN STAGE SPEC" in captured_content["user_content"]
        assert "MATERIAL_VALIDATION" in captured_content["user_content"]

    def test_design_reviewer_uses_summary_as_fallback_feedback(self):
        """design_reviewer_node must use summary as feedback when feedback is missing."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_design_revisions": 5}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [],
            "summary": "Summary message here",
            # No feedback field
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["reviewer_feedback"] == "Summary message here"

    def test_design_reviewer_handles_empty_issues(self):
        """design_reviewer_node must handle empty or None issues list."""
        from src.agents.design import design_reviewer_node

        state = create_initial_state("test", "paper text", "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {"verdict": "approve", "issues": None, "summary": "Good"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = design_reviewer_node(state)

        assert result["reviewer_issues"] == []

    # ═══════════════════════════════════════════════════════════════════════
    # simulation_designer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_simulation_designer_sets_design_description(self):
        """simulation_designer_node must set design_description."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ]
        }
        mock_response = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40},
            "materials": ["gold"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "design_description" in result
        assert result["design_description"] == mock_response
        assert result["design_description"]["geometry"] == "nanorod"
        assert result["design_description"]["parameters"]["length"] == 100
        assert result["design_description"]["parameters"]["diameter"] == 40
        assert result["design_description"]["materials"] == ["gold"]
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "design"

    def test_simulation_designer_rejects_missing_stage_id(self):
        """simulation_designer_node must reject when current_stage_id is missing."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = None

        result = simulation_designer_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "design_description" not in result
        assert result["workflow_phase"] == "design"
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0

    def test_simulation_designer_rejects_empty_stage_id(self):
        """simulation_designer_node must reject when current_stage_id is empty string."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = ""

        result = simulation_designer_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "design_description" not in result

    def test_simulation_designer_handles_new_assumptions(self):
        """simulation_designer_node must handle new_assumptions from LLM."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = {"global_assumptions": ["Existing assumption"]}
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": ["New assumption 1", "New assumption 2"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "assumptions" in result
        assert "global_assumptions" in result["assumptions"]
        assert len(result["assumptions"]["global_assumptions"]) == 3
        assert "Existing assumption" in result["assumptions"]["global_assumptions"]
        assert "New assumption 1" in result["assumptions"]["global_assumptions"]
        assert "New assumption 2" in result["assumptions"]["global_assumptions"]

    def test_simulation_designer_handles_empty_existing_assumptions(self):
        """simulation_designer_node must handle empty existing assumptions."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = {}  # Empty assumptions
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": ["New assumption"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "assumptions" in result
        assert result["assumptions"]["global_assumptions"] == ["New assumption"]

    def test_simulation_designer_handles_none_existing_assumptions(self):
        """simulation_designer_node must handle None existing assumptions."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = None  # None assumptions
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": ["New assumption"],
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "assumptions" in result
        assert result["assumptions"]["global_assumptions"] == ["New assumption"]

    def test_simulation_designer_ignores_invalid_new_assumptions(self):
        """simulation_designer_node must ignore non-list new_assumptions."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = {"global_assumptions": ["Existing"]}
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": "This is not a list",  # Invalid: string instead of list
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        # Should not have updated assumptions since new_assumptions was invalid
        assert "assumptions" not in result or result.get("assumptions") is None

    def test_simulation_designer_handles_llm_exception(self):
        """simulation_designer_node must escalate to user when LLM fails."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=Exception("API timeout"),
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result

    def test_simulation_designer_handles_context_escalation(self):
        """simulation_designer_node must handle context escalation."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }

        escalation_result = {
            "ask_user_trigger": "context_issue",
            "pending_user_questions": ["Context check failed"],
            "awaiting_user_input": True,
        }

        with patch(
            "src.agents.design.check_context_or_escalate",
            return_value=escalation_result,
        ):
            result = simulation_designer_node(state)

        assert result.get("ask_user_trigger") == "context_issue"
        assert result.get("awaiting_user_input") is True

    def test_simulation_designer_handles_non_dict_llm_output(self):
        """simulation_designer_node must handle non-dict LLM output."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        # LLM returns a string instead of dict
        mock_response = "This is a plain text design description"

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        # Should still store the design_description even if it's a string
        assert "design_description" in result
        assert result["design_description"] == mock_response
        # Should not try to add assumptions from non-dict output
        assert "assumptions" not in result

    def test_simulation_designer_adds_revision_feedback_to_prompt(self):
        """simulation_designer_node must include revision feedback in system prompt."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        state["reviewer_feedback"] = "Previous design had mesh resolution issues"
        
        captured_prompt = {}
        
        def capture_call(*args, **kwargs):
            captured_prompt["system_prompt"] = kwargs.get("system_prompt", "")
            return {"geometry": "nanorod"}

        with patch(
            "src.agents.design.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        assert "REVISION FEEDBACK" in captured_prompt["system_prompt"]
        assert "mesh resolution issues" in captured_prompt["system_prompt"]

    def test_simulation_designer_empty_new_assumptions(self):
        """simulation_designer_node must handle empty new_assumptions list."""
        from src.agents.design import simulation_designer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["assumptions"] = {"global_assumptions": ["Existing"]}
        state["plan"] = {
            "stages": [{"stage_id": "stage_0", "stage_type": "MATERIAL_VALIDATION", "targets": ["Fig1"]}]
        }
        mock_response = {
            "geometry": "nanorod",
            "new_assumptions": [],  # Empty list
        }

        with patch(
            "src.agents.design.call_agent_with_metrics",
            return_value=mock_response,
        ):
            with patch("src.agents.design.check_context_or_escalate", return_value=None):
                result = simulation_designer_node(state)

        # Should not update assumptions with empty list
        assert "assumptions" not in result

    # ═══════════════════════════════════════════════════════════════════════
    # code_generator_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_code_generator_sets_code_field(self):
        """code_generator_node must set code field."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100},
        }
        mock_code = "import meep as mp\nimport numpy as np\n\n# Simulation setup\nresolution = 10\nsize = mp.Vector3(1, 1, 0)\n# ... more code"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": mock_code, "expected_outputs": ["extinction.csv"]},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code" in result
        assert result["code"] == mock_code
        assert "import meep as mp" in result["code"]
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "code_generation"
        assert "expected_outputs" in result
        assert "extinction.csv" in result["expected_outputs"]

    def test_code_generator_rejects_missing_stage_id(self):
        """code_generator_node must reject when current_stage_id is missing."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = None

        result = code_generator_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True
        assert "code" not in result
        assert result["workflow_phase"] == "code_generation"
        assert "pending_user_questions" in result

    def test_code_generator_rejects_empty_stage_id(self):
        """code_generator_node must reject when current_stage_id is empty string."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = ""

        result = code_generator_node(state)

        assert result.get("ask_user_trigger") == "missing_stage_id"
        assert result.get("awaiting_user_input") is True

    def test_code_generator_rejects_stub_design(self):
        """code_generator_node must reject stub design descriptions."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = "STUB: This would be generated"

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] > 0
        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower() or "missing" in result["reviewer_feedback"].lower()

    def test_code_generator_rejects_design_with_todo_marker(self):
        """code_generator_node must reject design with TODO marker."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = "TODO: Fill in design parameters"

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert result["design_revision_count"] > 0
        assert "reviewer_feedback" in result

    def test_code_generator_rejects_empty_design(self):
        """code_generator_node must reject empty design descriptions."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = ""

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert "reviewer_feedback" in result

    def test_code_generator_rejects_short_design(self):
        """code_generator_node must reject design descriptions shorter than 50 chars."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = "nanorod geometry"  # Too short

        result = code_generator_node(state)

        assert "design_revision_count" in result
        assert "reviewer_feedback" in result

    def test_code_generator_rejects_missing_validated_materials_for_stage_1_plus(self):
        """code_generator_node must reject missing validated_materials for Stage 1+."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_1"
        state["current_stage_type"] = "SINGLE_STRUCTURE"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        state["validated_materials"] = []  # Empty

        result = code_generator_node(state)

        assert "run_error" in result
        assert "validated_materials" in result["run_error"].lower()
        assert "code_revision_count" in result

    def test_code_generator_rejects_none_validated_materials_for_stage_1_plus(self):
        """code_generator_node must reject None validated_materials for Stage 1+."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_1"
        state["current_stage_type"] = "SINGLE_STRUCTURE"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        state["validated_materials"] = None  # None

        result = code_generator_node(state)

        assert "run_error" in result
        assert "validated_materials" in result["run_error"].lower()

    def test_code_generator_allows_empty_validated_materials_for_stage_0(self):
        """code_generator_node must allow empty validated_materials for Stage 0."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"  # Stage 0 doesn't need validated_materials
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        state["validated_materials"] = []  # Empty but allowed for Stage 0
        mock_code = "import meep as mp\nimport numpy as np\n\n# Full simulation code here with proper implementation"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": mock_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code" in result
        assert "run_error" not in result

    def test_code_generator_rejects_stub_code(self):
        """code_generator_node must reject stub code from LLM."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        stub_code = "STUB: Code would be generated here"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": stub_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code_revision_count" in result
        assert result["code_revision_count"] > 0
        assert "reviewer_feedback" in result
        assert "stub" in result["reviewer_feedback"].lower() or "empty" in result["reviewer_feedback"].lower()

    def test_code_generator_rejects_code_with_todo_at_start(self):
        """code_generator_node must reject code starting with TODO marker."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        stub_code = "# TODO: Implement this simulation"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": stub_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code_revision_count" in result
        assert result["code_revision_count"] > 0

    def test_code_generator_accepts_code_with_todo_in_comments(self):
        """code_generator_node must accept valid code that has TODO in comments."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        # Valid code with TODO in a comment (not at the start, code is long enough)
        valid_code = """import meep as mp
import numpy as np

resolution = 10
geometry = []

# TODO: Consider adding more features later
# This is a valid simulation setup

def run_simulation():
    sim = mp.Simulation(resolution=resolution)
    sim.run()
"""

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": valid_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        # Should accept valid code with TODO in comments
        assert "code" in result
        assert "import meep" in result["code"]
        # code_revision_count should not be incremented for valid code
        assert result.get("code_revision_count", 0) == 0 or "reviewer_feedback" not in result

    def test_code_generator_rejects_short_code(self):
        """code_generator_node must reject code shorter than 50 chars."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        short_code = "import meep"  # Too short

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": short_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code_revision_count" in result
        assert result["code_revision_count"] > 0
        assert "reviewer_feedback" in result

    def test_code_generator_handles_llm_exception(self):
        """code_generator_node must escalate to user when LLM fails."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert result.get("ask_user_trigger") == "llm_error"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result

    def test_code_generator_handles_context_escalation(self):
        """code_generator_node must handle context escalation."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"

        escalation_result = {
            "ask_user_trigger": "context_issue",
            "pending_user_questions": ["Context check failed"],
            "awaiting_user_input": True,
        }

        with patch(
            "src.agents.code.check_context_or_escalate",
            return_value=escalation_result,
        ):
            result = code_generator_node(state)

        assert result.get("ask_user_trigger") == "context_issue"
        assert result.get("awaiting_user_input") is True

    def test_code_generator_uses_simulation_code_key(self):
        """code_generator_node must handle simulation_code key from LLM."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        mock_code = "import meep as mp\nimport numpy as np\n\n# Full simulation code with sufficient length for validation"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"simulation_code": mock_code},  # Different key
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code" in result
        assert result["code"] == mock_code

    def test_code_generator_handles_non_dict_llm_output(self):
        """code_generator_node must handle non-dict LLM output."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        # LLM returns string directly
        mock_code = "import meep as mp\nimport numpy as np\n\n# Full simulation with proper length for validation checks"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_code,  # String instead of dict
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "code" in result
        assert result["code"] == mock_code

    def test_code_generator_adds_revision_feedback_to_prompt(self):
        """code_generator_node must include revision feedback in system prompt."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        state["reviewer_feedback"] = "Previous code had import errors"
        
        captured_prompt = {}
        mock_code = "import meep as mp\nimport numpy as np\n\n# Full simulation with proper implementation"
        
        def capture_call(*args, **kwargs):
            captured_prompt["system_prompt"] = kwargs.get("system_prompt", "")
            return {"code": mock_code}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        assert "REVISION FEEDBACK" in captured_prompt["system_prompt"]
        assert "import errors" in captured_prompt["system_prompt"]

    def test_code_generator_respects_max_code_revisions(self):
        """code_generator_node must respect max_code_revisions when rejecting stub."""
        from src.agents.code import code_generator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["current_stage_type"] = "MATERIAL_VALIDATION"
        state["design_description"] = {
            "geometry": "nanorod",
            "parameters": {"length": 100, "diameter": 40, "mesh_resolution": 10},
        }
        state["code_revision_count"] = MAX_CODE_REVISIONS  # Already at max
        stub_code = "STUB: Code goes here"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value={"code": stub_code},
        ):
            with patch("src.agents.code.check_context_or_escalate", return_value=None):
                result = code_generator_node(state)

        # Should not exceed max
        assert result["code_revision_count"] == MAX_CODE_REVISIONS

    # ═══════════════════════════════════════════════════════════════════════
    # code_reviewer_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_code_reviewer_sets_verdict_field(self):
        """code_reviewer_node must set last_code_review_verdict."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Good code",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert "last_code_review_verdict" in result
        assert result["last_code_review_verdict"] == "approve"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "code_review"
        # Should include reviewer_issues (empty list on approve)
        assert "reviewer_issues" in result
        assert result["reviewer_issues"] == []
        # Should not set reviewer_feedback on approve
        assert "reviewer_feedback" not in result

    def test_code_reviewer_increments_counter_on_needs_revision(self):
        """code_reviewer_node must increment code_revision_count on needs_revision."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = 1
        state["runtime_config"] = {"max_code_revisions": 5}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs fixes",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["code_revision_count"] == 2
        assert "reviewer_feedback" in result
        assert result["reviewer_feedback"] == "Fix issues"

    def test_code_reviewer_respects_max_revisions(self):
        """code_reviewer_node must not increment counter beyond max_code_revisions."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = MAX_CODE_REVISIONS
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs fixes",
            "feedback": "Fix issues",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result.get("ask_user_trigger") == "code_review_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result

    def test_code_reviewer_escalates_at_max_revisions(self):
        """code_reviewer_node must escalate to user when max_code_revisions is reached."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = MAX_CODE_REVISIONS - 1  # One below max
        state["runtime_config"] = {"max_code_revisions": MAX_CODE_REVISIONS}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Problem"}],
            "summary": "Needs fixes",
            "feedback": "Fix the import issues",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        # After incrementing, we hit the max - should escalate
        assert result["code_revision_count"] == MAX_CODE_REVISIONS
        assert result.get("ask_user_trigger") == "code_review_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        # Should include options in the question
        assert any("PROVIDE_HINT" in q for q in result["pending_user_questions"])
        assert any("SKIP" in q for q in result["pending_user_questions"])
        assert any("STOP" in q for q in result["pending_user_questions"])

    def test_code_reviewer_defaults_to_needs_revision_on_missing_verdict(self):
        """code_reviewer_node must default to needs_revision when verdict is missing."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_code_revisions": 5}
        mock_response = {
            "issues": [],
            "summary": "No verdict",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "needs_revision"

    def test_code_reviewer_normalizes_verdict_pass_to_approve(self):
        """code_reviewer_node must normalize 'pass' verdict to 'approve'."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {"verdict": "pass", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "approve"

    def test_code_reviewer_normalizes_verdict_reject_to_needs_revision(self):
        """code_reviewer_node must normalize 'reject' verdict to 'needs_revision'."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_code_revisions": 5}
        mock_response = {"verdict": "reject", "issues": [], "summary": "Bad", "feedback": "Fix it"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "needs_revision"
        assert result["reviewer_feedback"] == "Fix it"

    def test_code_reviewer_handles_unknown_verdict(self):
        """code_reviewer_node must default to needs_revision for unknown verdicts."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_code_revisions": 5}
        mock_response = {"verdict": "maybe_okay", "issues": [], "summary": "Uncertain"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        # Unknown verdict should default to needs_revision (safer for code)
        assert result["last_code_review_verdict"] == "needs_revision"

    def test_code_reviewer_handles_llm_exception(self):
        """code_reviewer_node must auto-reject when LLM fails."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_code_revisions": 5}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            result = code_reviewer_node(state)

        # Should auto-reject with needs_revision on LLM failure
        assert result["last_code_review_verdict"] == "needs_revision"

    def test_code_reviewer_includes_design_in_user_content(self):
        """code_reviewer_node must include design spec in user content."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod", "parameters": {"length": 100}}
        
        captured_content = {}
        
        def capture_call(*args, **kwargs):
            captured_content["user_content"] = kwargs.get("user_content", "")
            return {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = code_reviewer_node(state)

        assert "DESIGN SPEC" in captured_content["user_content"]
        assert "nanorod" in captured_content["user_content"]

    def test_code_reviewer_uses_summary_as_fallback_feedback(self):
        """code_reviewer_node must use summary as feedback when feedback is missing."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["runtime_config"] = {"max_code_revisions": 5}
        mock_response = {
            "verdict": "needs_revision",
            "issues": [],
            "summary": "Summary message here",
            # No feedback field
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["reviewer_feedback"] == "Summary message here"

    def test_code_reviewer_handles_string_design(self):
        """code_reviewer_node must handle string design_description."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = "A nanorod with length 100nm"
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "approve"

    def test_code_reviewer_counter_does_not_increment_on_approve(self):
        """code_reviewer_node must not increment counter when verdict is approve."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        state["code_revision_count"] = 2
        mock_response = {"verdict": "approve", "issues": [], "summary": "Good"}

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["code_revision_count"] == 2, "Counter should not increment on approve"
        assert result["last_code_review_verdict"] == "approve"

    def test_code_reviewer_includes_issues_on_approve(self):
        """code_reviewer_node must include reviewer_issues even on approve."""
        from src.agents.code import code_reviewer_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["code"] = "import meep as mp\n# Code here"
        state["design_description"] = {"geometry": "nanorod"}
        mock_response = {
            "verdict": "approve",
            "issues": [{"severity": "minor", "description": "Minor style issue"}],
            "summary": "Good overall",
        }

        with patch(
            "src.agents.code.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = code_reviewer_node(state)

        assert result["last_code_review_verdict"] == "approve"
        assert "reviewer_issues" in result
        assert len(result["reviewer_issues"]) == 1
        assert result["reviewer_issues"][0]["severity"] == "minor"

    # ═══════════════════════════════════════════════════════════════════════
    # execution_validator_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_execution_validator_sets_verdict_field(self):
        """execution_validator_node must set execution_verdict."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "Execution successful",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert "execution_verdict" in result
        assert result["execution_verdict"] == "pass"
        assert "execution_feedback" in result
        assert result["execution_feedback"] == "Execution successful"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "execution_validation"
        # Should not increment failure count on pass
        assert "execution_failure_count" not in result or result.get("execution_failure_count") == state.get("execution_failure_count", 0)

    def test_execution_validator_increments_counter_on_fail(self):
        """execution_validator_node must increment execution_failure_count on fail."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "Simulation failed"
        state["execution_failure_count"] = 0
        state["runtime_config"] = {"max_execution_failures": 5}
        mock_response = {
            "verdict": "fail",
            "summary": "Execution failed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert result["execution_verdict"] == "fail"
        assert result["execution_failure_count"] == 1
        assert result["total_execution_failures"] == 1

    def test_execution_validator_respects_max_failures(self):
        """execution_validator_node must not increment counter beyond max_execution_failures."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "Simulation failed"
        state["execution_failure_count"] = MAX_EXECUTION_FAILURES
        state["total_execution_failures"] = MAX_EXECUTION_FAILURES
        mock_response = {
            "verdict": "fail",
            "summary": "Execution failed again",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert result["execution_failure_count"] == MAX_EXECUTION_FAILURES

    def test_execution_validator_escalates_at_max_failures(self):
        """execution_validator_node must escalate to user when max_execution_failures is reached."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "Simulation crashed"
        state["execution_failure_count"] = MAX_EXECUTION_FAILURES - 1  # One below max
        state["total_execution_failures"] = MAX_EXECUTION_FAILURES - 1
        state["runtime_config"] = {"max_execution_failures": MAX_EXECUTION_FAILURES}
        mock_response = {
            "verdict": "fail",
            "summary": "Execution failed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        # After incrementing, we hit the max - should escalate
        assert result["execution_failure_count"] == MAX_EXECUTION_FAILURES
        assert result.get("ask_user_trigger") == "execution_failure_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        # Should include run error in question
        assert any("crashed" in q.lower() for q in result["pending_user_questions"])

    def test_execution_validator_handles_timeout_with_skip_warning(self):
        """execution_validator_node must handle timeout with skip_with_warning fallback."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"timeout_exceeded": True, "files": []}
        state["run_error"] = "Execution exceeded timeout"
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "fallback_strategy": "skip_with_warning",
                }
            ]
        }

        result = execution_validator_node(state)

        assert result["execution_verdict"] == "pass"
        assert "timeout" in result["execution_feedback"].lower()

    def test_execution_validator_handles_timeout_without_fallback(self):
        """execution_validator_node must fail on timeout without skip_with_warning fallback."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"timeout_exceeded": True, "files": []}
        state["run_error"] = "Execution exceeded timeout"
        state["execution_failure_count"] = 0
        state["runtime_config"] = {"max_execution_failures": 5}
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "fallback_strategy": "ask_user",  # Not skip_with_warning
                }
            ]
        }

        result = execution_validator_node(state)

        assert result["execution_verdict"] == "fail"
        assert "timeout" in result["execution_feedback"].lower()
        assert result["execution_failure_count"] == 1

    def test_execution_validator_handles_timeout_string_pattern(self):
        """execution_validator_node must detect timeout from run_error string pattern."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}  # No timeout_exceeded flag
        state["run_error"] = "Process exceeded timeout limit"  # String pattern
        state["execution_failure_count"] = 0
        state["runtime_config"] = {"max_execution_failures": 5}
        state["plan"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "fallback_strategy": "skip_with_warning",
                }
            ]
        }

        result = execution_validator_node(state)

        # Should detect timeout from string pattern and use fallback
        assert result["execution_verdict"] == "pass"
        assert "timeout" in result["execution_feedback"].lower()

    def test_execution_validator_handles_missing_verdict(self):
        """execution_validator_node must handle missing verdict field."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "summary": "No verdict given",
            # No verdict field
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        # Missing verdict should default to pass
        assert result["execution_verdict"] == "pass"
        # Should note the missing verdict in feedback
        assert "missing verdict" in result["execution_feedback"].lower()

    def test_execution_validator_handles_llm_exception(self):
        """execution_validator_node must auto-pass when LLM fails."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            result = execution_validator_node(state)

        # Should auto-pass on LLM failure (to not block execution)
        assert result["execution_verdict"] == "pass"
        assert "auto-approved" in result["execution_feedback"].lower() or "llm" in result["execution_feedback"].lower()

    def test_execution_validator_includes_run_error_in_prompt(self):
        """execution_validator_node must include run_error in system prompt."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "ImportError: No module named 'meep'"
        
        captured_prompt = {}
        
        def capture_call(*args, **kwargs):
            captured_prompt["system_prompt"] = kwargs.get("system_prompt", "")
            return {"verdict": "fail", "summary": "Import error"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = execution_validator_node(state)

        assert "ImportError" in captured_prompt["system_prompt"]
        assert "meep" in captured_prompt["system_prompt"]

    def test_execution_validator_handles_none_stage_outputs(self):
        """execution_validator_node must handle None stage_outputs."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = None
        mock_response = {"verdict": "pass", "summary": "OK"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert result["execution_verdict"] == "pass"

    def test_execution_validator_handles_none_total_execution_failures(self):
        """execution_validator_node must handle None total_execution_failures."""
        from src.agents.execution import execution_validator_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": []}
        state["run_error"] = "Simulation failed"
        state["execution_failure_count"] = 0
        state["total_execution_failures"] = None  # None instead of 0
        state["runtime_config"] = {"max_execution_failures": 5}
        mock_response = {"verdict": "fail", "summary": "Failed"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = execution_validator_node(state)

        assert result["total_execution_failures"] == 1  # Should handle None as 0

    # ═══════════════════════════════════════════════════════════════════════
    # physics_sanity_node tests
    # ═══════════════════════════════════════════════════════════════════════

    def test_physics_sanity_sets_verdict_field(self):
        """physics_sanity_node must set physics_verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "Physics looks good",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert "physics_verdict" in result
        assert result["physics_verdict"] == "pass"
        assert "physics_feedback" in result
        assert result["physics_feedback"] == "Physics looks good"
        assert "workflow_phase" in result
        assert result["workflow_phase"] == "physics_validation"
        # Should not increment failure counters on pass
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result

    def test_physics_sanity_increments_physics_failure_count_on_fail(self):
        """physics_sanity_node must increment physics_failure_count on fail verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["physics_failure_count"] = 0
        state["runtime_config"] = {"max_physics_failures": 5}
        mock_response = {
            "verdict": "fail",
            "summary": "Physics check failed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "fail"
        assert result["physics_failure_count"] == 1

    def test_physics_sanity_respects_max_physics_failures(self):
        """physics_sanity_node must not increment counter beyond max_physics_failures."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        mock_response = {
            "verdict": "fail",
            "summary": "Physics check failed again",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_failure_count"] == MAX_PHYSICS_FAILURES

    def test_physics_sanity_escalates_at_max_physics_failures(self):
        """physics_sanity_node must escalate to user when max_physics_failures is reached."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["physics_failure_count"] = MAX_PHYSICS_FAILURES - 1  # One below max
        state["runtime_config"] = {"max_physics_failures": MAX_PHYSICS_FAILURES}
        mock_response = {
            "verdict": "fail",
            "summary": "Resonance frequency is off by 100%",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        # After incrementing, we hit the max - should escalate
        assert result["physics_failure_count"] == MAX_PHYSICS_FAILURES
        assert result.get("ask_user_trigger") == "physics_failure_limit"
        assert result.get("awaiting_user_input") is True
        assert "pending_user_questions" in result
        # Should include options
        assert any("RETRY" in q for q in result["pending_user_questions"])
        assert any("ACCEPT" in q for q in result["pending_user_questions"])
        assert any("SKIP" in q or "STOP" in q for q in result["pending_user_questions"])

    def test_physics_sanity_increments_design_revision_count_on_design_flaw(self):
        """physics_sanity_node must increment design_revision_count on design_flaw verdict."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_revision_count"] = 2
        state["runtime_config"] = {"max_design_revisions": 5}
        mock_response = {
            "verdict": "design_flaw",
            "summary": "Design flaw detected: wrong geometry",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "design_flaw"
        assert result["design_revision_count"] == 3
        assert "design_feedback" in result
        assert result["design_feedback"] == "Design flaw detected: wrong geometry"

    def test_physics_sanity_respects_max_design_revisions_on_design_flaw(self):
        """physics_sanity_node must not increment design_revision_count beyond max."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_revision_count"] = MAX_DESIGN_REVISIONS
        mock_response = {
            "verdict": "design_flaw",
            "summary": "Design flaw detected",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["design_revision_count"] == MAX_DESIGN_REVISIONS

    def test_physics_sanity_handles_warning_verdict(self):
        """physics_sanity_node must handle warning verdict without incrementing failure counters."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["physics_failure_count"] = 1
        state["design_revision_count"] = 1
        mock_response = {
            "verdict": "warning",
            "summary": "Minor concerns but proceed",
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "warning"
        # Warning should not increment failure counters
        assert "physics_failure_count" not in result
        assert "design_revision_count" not in result

    def test_physics_sanity_handles_missing_verdict(self):
        """physics_sanity_node must handle missing verdict field."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "summary": "No verdict given",
            # No verdict field
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        # Missing verdict should default to pass
        assert result["physics_verdict"] == "pass"
        assert "missing verdict" in result["physics_feedback"].lower()

    def test_physics_sanity_handles_llm_exception(self):
        """physics_sanity_node must auto-pass when LLM fails."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=Exception("API error"),
        ):
            result = physics_sanity_node(state)

        # Should auto-pass on LLM failure
        assert result["physics_verdict"] == "pass"
        assert "auto-approved" in result["physics_feedback"].lower() or "llm" in result["physics_feedback"].lower()

    def test_physics_sanity_handles_backtrack_suggestion(self):
        """physics_sanity_node must include backtrack_suggestion when agent suggests it."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "fail",
            "summary": "Results show fundamental issue",
            "backtrack_suggestion": {
                "suggest_backtrack": True,
                "target_stage": "stage_0",
                "reason": "Need to re-validate materials",
            },
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert "backtrack_suggestion" in result
        assert result["backtrack_suggestion"]["suggest_backtrack"] is True
        assert result["backtrack_suggestion"]["target_stage"] == "stage_0"

    def test_physics_sanity_ignores_backtrack_suggestion_when_false(self):
        """physics_sanity_node must not include backtrack_suggestion when suggest_backtrack is False."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "All good",
            "backtrack_suggestion": {
                "suggest_backtrack": False,
            },
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        # Should not include backtrack_suggestion when suggest_backtrack is False
        assert "backtrack_suggestion" not in result

    def test_physics_sanity_includes_design_in_user_content(self):
        """physics_sanity_node must include design_description in user content."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_description"] = {"geometry": "nanorod", "parameters": {"length": 100}}
        
        captured_content = {}
        
        def capture_call(*args, **kwargs):
            captured_content["user_content"] = kwargs.get("user_content", "")
            return {"verdict": "pass", "summary": "Good"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = physics_sanity_node(state)

        assert "Design Spec" in captured_content["user_content"]
        assert "nanorod" in captured_content["user_content"]

    def test_physics_sanity_handles_none_stage_outputs(self):
        """physics_sanity_node must handle None stage_outputs."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = None
        mock_response = {"verdict": "pass", "summary": "OK"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "pass"

    def test_physics_sanity_handles_empty_design_description(self):
        """physics_sanity_node must handle empty design_description."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_description"] = {}  # Empty dict
        mock_response = {"verdict": "pass", "summary": "OK"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        assert result["physics_verdict"] == "pass"

    def test_physics_sanity_handles_string_design_description(self):
        """physics_sanity_node must handle string design_description."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        state["design_description"] = "A nanorod with length 100nm"  # String instead of dict
        
        captured_content = {}
        
        def capture_call(*args, **kwargs):
            captured_content["user_content"] = kwargs.get("user_content", "")
            return {"verdict": "pass", "summary": "OK"}

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            side_effect=capture_call,
        ):
            result = physics_sanity_node(state)

        assert "A nanorod with length 100nm" in captured_content["user_content"]

    def test_physics_sanity_default_backtrack_suggestion(self):
        """physics_sanity_node must add default backtrack_suggestion when missing from response."""
        from src.agents.execution import physics_sanity_node

        state = create_initial_state("test", "paper text " * 20, "plasmonics")
        state["current_stage_id"] = "stage_0"
        state["stage_outputs"] = {"files": ["output.csv"]}
        mock_response = {
            "verdict": "pass",
            "summary": "OK",
            # No backtrack_suggestion
        }

        with patch(
            "src.agents.execution.call_agent_with_metrics",
            return_value=mock_response,
        ):
            result = physics_sanity_node(state)

        # Should not add backtrack_suggestion when not suggested
        assert "backtrack_suggestion" not in result


