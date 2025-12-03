"""Routing decisions following reviewer/verifier verdicts."""


class TestRoutingFunctions:
    """Happy-path routing decisions that advance the workflow."""

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

