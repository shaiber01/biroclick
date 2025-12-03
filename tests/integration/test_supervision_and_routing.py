from unittest.mock import patch

import pytest


class TestSupervisorMaterialCheckpoint:
    """Supervisor ask_user trigger handling for material checkpoints."""

    def test_supervisor_handles_material_checkpoint(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        pending_materials = [{"name": "gold", "path": "/materials/Au.csv"}]
        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = pending_materials
        base_state["pending_user_questions"] = ["Approve materials?"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {"stages": [{"stage_id": "stage_0", "status": "in_progress"}]}

        result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"
        assert result["validated_materials"] == pending_materials
        assert result.get("pending_validated_materials") == []
        assert result.get("ask_user_trigger") is None

    def test_supervisor_material_checkpoint_without_pending_materials_escalates(
        self, base_state
    ):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "material_checkpoint"
        base_state["user_responses"] = {"Q1": "APPROVE"}
        base_state["pending_validated_materials"] = []
        base_state["pending_user_questions"] = ["Approve materials?"]

        result = supervisor_node(base_state)
        assert result.get("supervisor_verdict") == "ask_user"
        questions = result.get("pending_user_questions", [])
        assert questions and "No materials were extracted" in questions[0]
        assert result.get("ask_user_trigger") is None

    def test_supervisor_defaults_on_llm_error(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["current_stage_id"] = "stage_0"
        with patch(
            "src.agents.supervision.supervisor.call_agent_with_metrics",
            side_effect=RuntimeError("LLM API error"),
        ):
            result = supervisor_node(base_state)

        assert result["supervisor_verdict"] == "ok_continue"

    def test_supervisor_with_unknown_trigger(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "unknown_trigger_xyz"
        base_state["user_responses"] = {"Q1": "yes"}
        base_state["pending_user_questions"] = ["Unknown question"]

        result = supervisor_node(base_state)
        assert "supervisor_verdict" in result


class TestSupervisorTriggerHandlers:
    """Test supervisor handles various ask_user_trigger types correctly."""

    def test_supervisor_handles_code_review_limit_with_hint(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "PROVIDE_HINT: Try using mp.Medium instead"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3

        result = supervisor_node(base_state)

        assert result.get("code_revision_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "User hint" in result.get("reviewer_feedback", "")
        assert result.get("ask_user_trigger") is None
        assert result.get("workflow_phase") == "supervision"
        assert result.get("pending_user_questions") in (None, [])

    def test_supervisor_handles_design_review_limit_skip(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "design_review_limit"
        base_state["user_responses"] = {"Q1": "SKIP"}
        base_state["pending_user_questions"] = ["Design review limit"]
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}]
        }

        result = supervisor_node(base_state)
        assert result.get("supervisor_verdict") == "ok_continue"

    def test_supervisor_handles_execution_failure_with_retry(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "execution_failure_limit"
        base_state["user_responses"] = {
            "Q1": "RETRY_WITH_GUIDANCE: Increase memory allocation"
        }
        base_state["pending_user_questions"] = ["Execution failed"]
        base_state["execution_failure_count"] = 2

        result = supervisor_node(base_state)
        assert result.get("execution_failure_count") == 0
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "User guidance" in result.get("supervisor_feedback", "")
        assert result.get("ask_user_trigger") is None

    def test_supervisor_handles_llm_error_with_retry(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "llm_error"
        base_state["user_responses"] = {"Q1": "RETRY"}
        base_state["pending_user_questions"] = ["LLM API failed"]

        result = supervisor_node(base_state)
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "retry" in result.get("supervisor_feedback", "").lower()

    def test_supervisor_handles_context_overflow_with_truncate(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["paper_text"] = "x" * 25000
        base_state["ask_user_trigger"] = "context_overflow"
        base_state["user_responses"] = {"Q1": "TRUNCATE"}
        base_state["pending_user_questions"] = ["Context too long"]

        result = supervisor_node(base_state)
        if "paper_text" in result:
            assert len(result["paper_text"]) < 25000
            assert "[TRUNCATED" in result["paper_text"]
        assert result.get("supervisor_verdict") == "ok_continue"
        assert "Truncating" in result.get("supervisor_feedback", "")
        assert result.get("ask_user_trigger") is None

    def test_supervisor_logs_user_interactions(self, base_state):
        from src.agents.supervision.supervisor import supervisor_node

        base_state["ask_user_trigger"] = "code_review_limit"
        base_state["user_responses"] = {"Q1": "PROVIDE_HINT: Focus on resonance monitors"}
        base_state["pending_user_questions"] = ["Code review limit reached"]
        base_state["code_revision_count"] = 3
        base_state["current_stage_id"] = "stage_0"
        base_state["progress"] = {
            "stages": [{"stage_id": "stage_0", "status": "in_progress"}],
            "user_interactions": [],
        }

        result = supervisor_node(base_state)

        interactions = result.get("progress", {}).get("user_interactions", [])
        assert len(interactions) == 1
        entry = interactions[0]
        assert entry.get("interaction_type") == "code_review_limit"
        assert entry.get("context", {}).get("stage_id") == "stage_0"
        assert "PROVIDE_HINT" in entry.get("user_response", "")
        assert entry.get("question")


class TestRoutingFunctions:
    """Routing decisions following reviewer/verifier verdicts."""

    def test_route_after_plan_review_approve_goes_to_select_stage(
        self, base_state, valid_plan
    ):
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "approve"
        base_state["replan_count"] = 0
        assert route_after_plan_review(base_state) == "select_stage"

    def test_route_after_plan_review_needs_revision_goes_to_plan(
        self, base_state, valid_plan
    ):
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 0
        assert route_after_plan_review(base_state) == "plan"

    def test_route_after_design_review_approve_goes_to_generate_code(self, base_state):
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "approve"
        assert route_after_design_review(base_state) == "generate_code"

    def test_route_after_design_review_needs_revision_goes_to_design(self, base_state):
        from src.routing import route_after_design_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 0
        assert route_after_design_review(base_state) == "design"

    def test_route_after_code_review_approve_goes_to_run_code(self, base_state):
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "approve"
        assert route_after_code_review(base_state) == "run_code"

    def test_route_after_code_review_needs_revision_goes_to_generate_code(
        self, base_state
    ):
        from src.routing import route_after_code_review

        base_state["current_stage_id"] = "stage_0"
        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 0
        assert route_after_code_review(base_state) == "generate_code"

    def test_route_after_execution_check_pass_goes_to_physics_check(self, base_state):
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "pass"
        assert route_after_execution_check(base_state) == "physics_check"

    def test_route_after_execution_check_fail_goes_to_generate_code(self, base_state):
        from src.routing import route_after_execution_check

        base_state["current_stage_id"] = "stage_0"
        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 0
        assert route_after_execution_check(base_state) == "generate_code"

    def test_route_after_physics_check_pass_goes_to_analyze(self, base_state):
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "pass"
        assert route_after_physics_check(base_state) == "analyze"

    def test_route_after_physics_check_design_flaw_goes_to_design(self, base_state):
        from src.routing import route_after_physics_check

        base_state["current_stage_id"] = "stage_0"
        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        assert route_after_physics_check(base_state) == "design"


class TestRoutingCountLimits:
    """Verify routing functions respect count limits and escalate correctly."""

    def test_code_review_escalates_at_limit(self, base_state):
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 3
        assert route_after_code_review(base_state) == "ask_user"

    def test_code_review_allows_under_limit(self, base_state):
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = "needs_revision"
        base_state["code_revision_count"] = 2
        assert route_after_code_review(base_state) == "generate_code"

    def test_design_review_escalates_at_limit(self, base_state):
        from src.routing import route_after_design_review

        base_state["last_design_review_verdict"] = "needs_revision"
        base_state["design_revision_count"] = 3
        assert route_after_design_review(base_state) == "ask_user"

    def test_execution_check_escalates_at_limit(self, base_state):
        from src.routing import route_after_execution_check

        base_state["execution_verdict"] = "fail"
        base_state["execution_failure_count"] = 2
        assert route_after_execution_check(base_state) == "ask_user"

    def test_physics_check_routes_to_design_on_design_flaw(self, base_state):
        from src.routing import route_after_physics_check

        base_state["physics_verdict"] = "design_flaw"
        base_state["design_revision_count"] = 0
        assert route_after_physics_check(base_state) == "design"

    def test_routing_with_none_verdict_escalates(self, base_state):
        from src.routing import route_after_code_review

        base_state["last_code_review_verdict"] = None
        assert route_after_code_review(base_state) == "ask_user"

    def test_plan_review_escalates_at_replan_limit(self, base_state, valid_plan):
        from src.routing import route_after_plan_review

        base_state["plan"] = valid_plan
        base_state["last_plan_review_verdict"] = "needs_revision"
        base_state["replan_count"] = 2
        assert route_after_plan_review(base_state) == "ask_user"


class TestStageSelectionEdgeCases:
    """Stage selection edge-case handling."""

    def test_select_stage_respects_validation_hierarchy(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],
                },
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "not_started",
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "not_started",
                    "dependencies": ["stage_0"],
                },
            ]
        }

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"

    def test_select_stage_skips_completed_stages(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],
                },
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_success",
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "not_started",
                    "dependencies": ["stage_0"],
                },
            ]
        }

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_1"

    def test_select_stage_detects_deadlock(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "completed_failed",
                    "dependencies": [],
                }
            ]
        }

        result = select_stage_node(base_state)
        assert result.get("ask_user_trigger") == "deadlock_detected"

    def test_blocked_stage_with_unsatisfied_deps_stays_blocked(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "targets": ["Fig2"],
                    "dependencies": ["stage_0"],
                },
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "not_started",
                    "dependencies": [],
                },
                {
                    "stage_id": "stage_1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "status": "blocked",
                    "dependencies": ["stage_0"],
                },
            ]
        }

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"

    def test_select_stage_resets_counters_on_new_stage(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "not_started",
                    "dependencies": [],
                }
            ]
        }
        base_state["current_stage_id"] = None
        base_state["design_revision_count"] = 5
        base_state["code_revision_count"] = 5

        result = select_stage_node(base_state)
        assert result.get("design_revision_count") == 0
        assert result.get("code_revision_count") == 0

    def test_select_stage_sets_stage_start_time_and_clears_outputs(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
        }
        base_state["progress"] = {
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "status": "not_started",
                }
            ]
        }
        base_state["stage_outputs"] = {"files": ["/tmp/old.csv"]}
        base_state["run_error"] = "stale failure"

        result = select_stage_node(base_state)
        assert result["current_stage_id"] == "stage_0"
        assert result.get("stage_outputs") == {}
        assert result.get("run_error") is None
        start_time = result.get("stage_start_time")
        assert isinstance(start_time, str) and "T" in start_time

    def test_select_stage_reports_progress_init_failure(self, base_state):
        from src.agents.stage_selection import select_stage_node

        base_state["plan"] = {
            "paper_id": "test",
            "stages": [
                {
                    "stage_id": "stage_0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "targets": ["Fig1"],
                }
            ],
        }
        base_state["progress"] = {}

        with patch(
            "src.agents.stage_selection.initialize_progress_from_plan",
            side_effect=RuntimeError("boom"),
        ):
            result = select_stage_node(base_state)

        assert result.get("ask_user_trigger") == "progress_init_failed"
        assert result.get("awaiting_user_input") is True
        questions = result.get("pending_user_questions", [])
        assert questions and "Failed to initialize progress" in questions[0]

