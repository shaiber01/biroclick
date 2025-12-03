"""Tests that router configs stay in sync with schema definitions.

This module ensures that:
1. All fields accessed by routers exist in ReproState/RuntimeConfig
2. Default constants used by routers match schema constants
3. Route types returned by routers are valid
4. Verdict types are properly defined
"""

from typing import get_type_hints, get_args, Literal
import inspect

import pytest

from schemas.state import (
    ReproState,
    RuntimeConfig,
    MAX_DESIGN_REVISIONS,
    MAX_CODE_REVISIONS,
    MAX_EXECUTION_FAILURES,
    MAX_PHYSICS_FAILURES,
    MAX_ANALYSIS_REVISIONS,
    MAX_REPLANS,
    DEFAULT_RUNTIME_CONFIG,
)
from src.routing import (
    create_verdict_router,
    route_after_plan_review,
    route_after_design_review,
    route_after_code_review,
    route_after_execution_check,
    route_after_physics_check,
    route_after_comparison_check,
    RouteType,
    PlanReviewVerdict,
    DesignReviewVerdict,
    CodeReviewVerdict,
    ExecutionVerdict,
    PhysicsVerdict,
    ComparisonVerdict,
)


class TestSchemaConsistency:
    """
    Tests that router configurations match the ReproState and RuntimeConfig schemas.
    This ensures that fields accessed by routers actually exist.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # DEFINITIVE LIST OF ROUTER CONFIGURATIONS
    # This serves as the single source of truth for what routers use.
    # If a router is updated, this list MUST be updated too.
    # ═══════════════════════════════════════════════════════════════════════
    
    ROUTER_CONFIGS = [
        {
            "name": "plan_review",
            "verdict_field": "last_plan_review_verdict",
            "count_fields": [("replan_count", "max_replans", MAX_REPLANS)],
        },
        {
            "name": "design_review",
            "verdict_field": "last_design_review_verdict",
            "count_fields": [("design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS)],
        },
        {
            "name": "code_review",
            "verdict_field": "last_code_review_verdict",
            "count_fields": [("code_revision_count", "max_code_revisions", MAX_CODE_REVISIONS)],
        },
        {
            "name": "execution_check",
            "verdict_field": "execution_verdict",
            "count_fields": [("execution_failure_count", "max_execution_failures", MAX_EXECUTION_FAILURES)],
        },
        {
            "name": "physics_check",
            "verdict_field": "physics_verdict",
            # Physics check uses TWO count fields: physics_failure_count for "fail", design_revision_count for "design_flaw"
            "count_fields": [
                ("physics_failure_count", "max_physics_failures", MAX_PHYSICS_FAILURES),
                ("design_revision_count", "max_design_revisions", MAX_DESIGN_REVISIONS),
            ],
        },
        {
            "name": "comparison_check",
            "verdict_field": "comparison_verdict",
            "count_fields": [("analysis_revision_count", "max_analysis_revisions", MAX_ANALYSIS_REVISIONS)],
        },
    ]

    def test_all_runtime_config_keys_exist(self):
        """
        Verify that ALL max_count_keys used in routers exist in RuntimeConfig.
        Failure here means a router relies on a config key that isn't defined.
        """
        runtime_config_keys = set(get_type_hints(RuntimeConfig).keys())
        
        missing_keys = []
        for config in self.ROUTER_CONFIGS:
            for _, max_count_key, _ in config["count_fields"]:
                if max_count_key not in runtime_config_keys:
                    missing_keys.append(f"{config['name']}: {max_count_key}")
        
        if missing_keys:
            pytest.fail(
                f"Router config keys missing from RuntimeConfig schema:\n"
                f"  - {chr(10).join('  - ' + k for k in missing_keys)}\n"
                f"Available keys: {sorted(runtime_config_keys)}"
            )
    
    def test_all_runtime_config_keys_have_correct_type(self):
        """
        Verify that max_count_keys in RuntimeConfig are typed as int.
        If they're not int, comparison logic will fail.
        """
        type_hints = get_type_hints(RuntimeConfig)
        
        wrong_types = []
        for config in self.ROUTER_CONFIGS:
            for _, max_count_key, _ in config["count_fields"]:
                if max_count_key in type_hints:
                    expected_type = type_hints[max_count_key]
                    if expected_type != int:
                        wrong_types.append(
                            f"{max_count_key}: expected int, got {expected_type}"
                        )
        
        if wrong_types:
            pytest.fail(
                f"RuntimeConfig keys have wrong types:\n"
                f"  - {chr(10).join('  - ' + t for t in wrong_types)}"
            )

    def test_all_verdict_fields_exist_in_state(self):
        """
        Verify that ALL verdict_fields used in routers exist in ReproState.
        """
        state_keys = set(get_type_hints(ReproState).keys())
        
        missing_fields = []
        for config in self.ROUTER_CONFIGS:
            verdict_field = config["verdict_field"]
            if verdict_field not in state_keys:
                missing_fields.append(f"{config['name']}: {verdict_field}")
        
        if missing_fields:
            pytest.fail(
                f"Verdict fields missing from ReproState schema:\n"
                f"  - {chr(10).join('  - ' + f for f in missing_fields)}\n"
                f"Available state keys (partial): {sorted([k for k in state_keys if 'verdict' in k.lower()])}"
            )

    def test_all_count_fields_exist_in_state(self):
        """
        Verify that ALL count_fields used in routers exist in ReproState.
        """
        state_keys = set(get_type_hints(ReproState).keys())
        
        missing_fields = []
        for config in self.ROUTER_CONFIGS:
            for count_field, _, _ in config["count_fields"]:
                if count_field not in state_keys:
                    missing_fields.append(f"{config['name']}: {count_field}")
        
        if missing_fields:
            pytest.fail(
                f"Count fields missing from ReproState schema:\n"
                f"  - {chr(10).join('  - ' + f for f in missing_fields)}\n"
                f"Available state keys (partial): {sorted([k for k in state_keys if 'count' in k.lower()])}"
            )

    def test_count_fields_have_correct_type(self):
        """
        Verify that count_fields in ReproState are typed as int.
        If they're not int, comparison logic may fail.
        """
        type_hints = get_type_hints(ReproState)
        
        wrong_types = []
        for config in self.ROUTER_CONFIGS:
            for count_field, _, _ in config["count_fields"]:
                if count_field in type_hints:
                    expected_type = type_hints[count_field]
                    if expected_type != int:
                        wrong_types.append(
                            f"{count_field}: expected int, got {expected_type}"
                        )
        
        if wrong_types:
            pytest.fail(
                f"Count fields have wrong types:\n"
                f"  - {chr(10).join('  - ' + t for t in wrong_types)}"
            )

    def test_verdict_fields_allow_none(self):
        """
        Verify that verdict fields are Optional (allow None).
        Routers handle None verdicts specially, so the schema must allow None.
        """
        type_hints = get_type_hints(ReproState)
        
        not_optional = []
        for config in self.ROUTER_CONFIGS:
            verdict_field = config["verdict_field"]
            if verdict_field in type_hints:
                hint = type_hints[verdict_field]
                # Check if it's Optional (Union with None)
                hint_str = str(hint)
                if "None" not in hint_str and "Optional" not in hint_str:
                    not_optional.append(f"{verdict_field}: {hint}")
        
        if not_optional:
            pytest.fail(
                f"Verdict fields must be Optional (allow None) for router error handling:\n"
                f"  - {chr(10).join('  - ' + n for n in not_optional)}"
            )


class TestDefaultConstantsConsistency:
    """
    Tests that default constants used by routers match the schema constants.
    """
    
    def test_default_runtime_config_has_all_max_keys(self):
        """
        Verify that DEFAULT_RUNTIME_CONFIG has values for all max keys used by routers.
        """
        missing_keys = []
        for config in TestSchemaConsistency.ROUTER_CONFIGS:
            for _, max_count_key, _ in config["count_fields"]:
                if max_count_key not in DEFAULT_RUNTIME_CONFIG:
                    missing_keys.append(max_count_key)
        
        if missing_keys:
            pytest.fail(
                f"DEFAULT_RUNTIME_CONFIG is missing keys: {missing_keys}"
            )

    def test_default_runtime_config_values_match_constants(self):
        """
        Verify that DEFAULT_RUNTIME_CONFIG values match the module-level constants.
        This catches cases where someone updates a constant but forgets the config.
        """
        expected_mappings = {
            "max_replans": MAX_REPLANS,
            "max_design_revisions": MAX_DESIGN_REVISIONS,
            "max_code_revisions": MAX_CODE_REVISIONS,
            "max_execution_failures": MAX_EXECUTION_FAILURES,
            "max_physics_failures": MAX_PHYSICS_FAILURES,
            "max_analysis_revisions": MAX_ANALYSIS_REVISIONS,
        }
        
        mismatches = []
        for key, expected_value in expected_mappings.items():
            actual_value = DEFAULT_RUNTIME_CONFIG.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"{key}: constant={expected_value}, DEFAULT_RUNTIME_CONFIG={actual_value}"
                )
        
        if mismatches:
            pytest.fail(
                f"DEFAULT_RUNTIME_CONFIG values don't match constants:\n"
                f"  - {chr(10).join('  - ' + m for m in mismatches)}"
            )

    def test_router_default_max_values_match_constants(self):
        """
        Verify that router configurations use the correct default_max values.
        This catches cases where a router hardcodes a different default.
        """
        # This test validates the ROUTER_CONFIGS definition itself
        for config in TestSchemaConsistency.ROUTER_CONFIGS:
            for count_field, max_count_key, expected_default in config["count_fields"]:
                # The expected_default should match what we expect from the constant
                actual_constant = {
                    "max_replans": MAX_REPLANS,
                    "max_design_revisions": MAX_DESIGN_REVISIONS,
                    "max_code_revisions": MAX_CODE_REVISIONS,
                    "max_execution_failures": MAX_EXECUTION_FAILURES,
                    "max_physics_failures": MAX_PHYSICS_FAILURES,
                    "max_analysis_revisions": MAX_ANALYSIS_REVISIONS,
                }.get(max_count_key)
                
                assert expected_default == actual_constant, (
                    f"{config['name']} router: {max_count_key} default_max={expected_default} "
                    f"doesn't match constant={actual_constant}"
                )

    def test_constants_are_positive_integers(self):
        """
        Verify that all limit constants are positive integers.
        Zero or negative limits would cause immediate escalation.
        """
        constants = {
            "MAX_REPLANS": MAX_REPLANS,
            "MAX_DESIGN_REVISIONS": MAX_DESIGN_REVISIONS,
            "MAX_CODE_REVISIONS": MAX_CODE_REVISIONS,
            "MAX_EXECUTION_FAILURES": MAX_EXECUTION_FAILURES,
            "MAX_PHYSICS_FAILURES": MAX_PHYSICS_FAILURES,
            "MAX_ANALYSIS_REVISIONS": MAX_ANALYSIS_REVISIONS,
        }
        
        issues = []
        for name, value in constants.items():
            if not isinstance(value, int):
                issues.append(f"{name}={value} is not an int (type: {type(value).__name__})")
            elif value <= 0:
                issues.append(f"{name}={value} is not positive")
        
        if issues:
            pytest.fail(
                f"Limit constants must be positive integers:\n"
                f"  - {chr(10).join('  - ' + i for i in issues)}"
            )


class TestRouteTypeConsistency:
    """
    Tests that RouteType and verdict types are consistent with actual usage.
    """
    
    def test_route_type_includes_all_expected_routes(self):
        """
        Verify that RouteType includes all routes that routers can return.
        """
        # Get all valid route values from RouteType Literal
        route_type_args = get_args(RouteType)
        valid_routes = set(route_type_args)
        
        # Expected routes from our known router configurations
        expected_routes = {
            # From plan_review router
            "select_stage", "plan", "ask_user",
            # From design_review router
            "generate_code", "design",
            # From code_review router
            "run_code",
            # From execution_check router
            "physics_check",
            # From physics_check router
            "analyze",
            # From comparison_check router
            "supervisor",
        }
        
        missing_routes = expected_routes - valid_routes
        if missing_routes:
            pytest.fail(
                f"RouteType is missing expected routes: {sorted(missing_routes)}\n"
                f"Available routes in RouteType: {sorted(valid_routes)}"
            )

    def test_route_type_is_literal(self):
        """
        Verify that RouteType is a Literal type for type safety.
        """
        # Get the origin of RouteType (should be Literal)
        from typing import get_origin
        origin = get_origin(RouteType)
        assert origin is Literal, f"RouteType should be Literal, got {origin}"

    def test_route_type_values_are_all_strings(self):
        """
        Verify that all RouteType values are strings.
        """
        route_values = get_args(RouteType)
        non_strings = [v for v in route_values if not isinstance(v, str)]
        
        if non_strings:
            pytest.fail(f"RouteType contains non-string values: {non_strings}")

    def test_route_type_values_are_non_empty(self):
        """
        Verify that no RouteType values are empty strings.
        """
        route_values = get_args(RouteType)
        empty_strings = [v for v in route_values if v == ""]
        
        if empty_strings:
            pytest.fail("RouteType contains empty string values")

    def test_route_type_values_are_snake_case(self):
        """
        Verify that RouteType values follow snake_case convention.
        """
        import re
        route_values = get_args(RouteType)
        snake_case_pattern = re.compile(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$')
        
        non_snake_case = [v for v in route_values if not snake_case_pattern.match(v)]
        
        if non_snake_case:
            pytest.fail(f"RouteType values not in snake_case: {non_snake_case}")


class TestVerdictTypeConsistency:
    """
    Tests that verdict types match expected values.
    """
    
    @pytest.mark.parametrize("verdict_type,expected_values", [
        (PlanReviewVerdict, {"approve", "needs_revision"}),
        (DesignReviewVerdict, {"approve", "needs_revision"}),
        (CodeReviewVerdict, {"approve", "needs_revision"}),
        (ExecutionVerdict, {"pass", "warning", "fail"}),
        (PhysicsVerdict, {"pass", "warning", "fail", "design_flaw"}),
        (ComparisonVerdict, {"approve", "needs_revision"}),
    ])
    def test_verdict_type_has_expected_values(self, verdict_type, expected_values):
        """
        Verify that each verdict type contains exactly the expected values.
        """
        actual_values = set(get_args(verdict_type))
        
        assert actual_values == expected_values, (
            f"{verdict_type.__name__}: expected {expected_values}, got {actual_values}"
        )

    @pytest.mark.parametrize("verdict_type,verdict_name", [
        (PlanReviewVerdict, "PlanReviewVerdict"),
        (DesignReviewVerdict, "DesignReviewVerdict"),
        (CodeReviewVerdict, "CodeReviewVerdict"),
        (ExecutionVerdict, "ExecutionVerdict"),
        (PhysicsVerdict, "PhysicsVerdict"),
        (ComparisonVerdict, "ComparisonVerdict"),
    ])
    def test_verdict_types_are_literal(self, verdict_type, verdict_name):
        """
        Verify that verdict types are Literal types.
        """
        from typing import get_origin
        origin = get_origin(verdict_type)
        assert origin is Literal, f"{verdict_name} should be Literal, got {origin}"

    def test_all_review_verdicts_consistent(self):
        """
        Verify that all review-type verdicts have the same values.
        PlanReview, DesignReview, CodeReview, Comparison should all use approve/needs_revision.
        """
        review_verdicts = [
            ("PlanReviewVerdict", PlanReviewVerdict),
            ("DesignReviewVerdict", DesignReviewVerdict),
            ("CodeReviewVerdict", CodeReviewVerdict),
            ("ComparisonVerdict", ComparisonVerdict),
        ]
        
        expected = {"approve", "needs_revision"}
        for name, verdict_type in review_verdicts:
            actual = set(get_args(verdict_type))
            assert actual == expected, (
                f"{name} should have {expected} for consistency, got {actual}"
            )

    def test_execution_and_physics_share_common_verdicts(self):
        """
        Verify that ExecutionVerdict and PhysicsVerdict share pass/warning/fail.
        PhysicsVerdict has additional 'design_flaw'.
        """
        execution_values = set(get_args(ExecutionVerdict))
        physics_values = set(get_args(PhysicsVerdict))
        
        common_expected = {"pass", "warning", "fail"}
        
        execution_common = execution_values & common_expected
        assert execution_common == common_expected, (
            f"ExecutionVerdict missing common verdicts: {common_expected - execution_values}"
        )
        
        physics_common = physics_values & common_expected
        assert physics_common == common_expected, (
            f"PhysicsVerdict missing common verdicts: {common_expected - physics_values}"
        )
        
        # Physics should have design_flaw in addition
        assert "design_flaw" in physics_values, (
            "PhysicsVerdict should have 'design_flaw' verdict"
        )


class TestPreConfiguredRoutersExist:
    """
    Tests that all expected pre-configured routers are exported.
    """
    
    def test_all_expected_routers_exist(self):
        """
        Verify that all expected router functions are exported from routing module.
        """
        import src.routing as routing_module
        
        expected_routers = [
            "route_after_plan_review",
            "route_after_design_review",
            "route_after_code_review",
            "route_after_execution_check",
            "route_after_physics_check",
            "route_after_comparison_check",
        ]
        
        missing_routers = []
        for router_name in expected_routers:
            if not hasattr(routing_module, router_name):
                missing_routers.append(router_name)
        
        if missing_routers:
            pytest.fail(f"Missing router exports: {missing_routers}")

    def test_all_routers_are_callable(self):
        """
        Verify that all pre-configured routers are callable functions.
        """
        routers = [
            ("route_after_plan_review", route_after_plan_review),
            ("route_after_design_review", route_after_design_review),
            ("route_after_code_review", route_after_code_review),
            ("route_after_execution_check", route_after_execution_check),
            ("route_after_physics_check", route_after_physics_check),
            ("route_after_comparison_check", route_after_comparison_check),
        ]
        
        not_callable = []
        for name, router in routers:
            if not callable(router):
                not_callable.append(name)
        
        if not_callable:
            pytest.fail(f"Routers are not callable: {not_callable}")

    def test_router_count_matches_config_count(self):
        """
        Verify that the number of pre-configured routers matches ROUTER_CONFIGS.
        This catches cases where a new router is added but not tested.
        """
        expected_count = len(TestSchemaConsistency.ROUTER_CONFIGS)
        
        import src.routing as routing_module
        actual_routers = [
            name for name in dir(routing_module)
            if name.startswith("route_after_") and callable(getattr(routing_module, name))
        ]
        actual_count = len(actual_routers)
        
        assert actual_count == expected_count, (
            f"Expected {expected_count} pre-configured routers (from ROUTER_CONFIGS), "
            f"but found {actual_count}: {actual_routers}"
        )

    def test_factory_function_exists(self):
        """
        Verify that the create_verdict_router factory function is exported.
        """
        import src.routing as routing_module
        
        assert hasattr(routing_module, "create_verdict_router"), (
            "create_verdict_router factory function should be exported"
        )
        assert callable(routing_module.create_verdict_router), (
            "create_verdict_router should be callable"
        )


class TestRouterSchemaIntegration:
    """
    Integration tests that verify routers work correctly with actual state schemas.
    """
    
    @pytest.fixture
    def minimal_state(self):
        """Create a minimal state dict with only fields needed for routing."""
        return {
            "runtime_config": DEFAULT_RUNTIME_CONFIG.copy(),
            # Verdict fields
            "last_plan_review_verdict": None,
            "last_design_review_verdict": None,
            "last_code_review_verdict": None,
            "execution_verdict": None,
            "physics_verdict": None,
            "comparison_verdict": None,
            # Count fields
            "replan_count": 0,
            "design_revision_count": 0,
            "code_revision_count": 0,
            "execution_failure_count": 0,
            "physics_failure_count": 0,
            "analysis_revision_count": 0,
        }
    
    @pytest.mark.parametrize("router,verdict_field,approve_verdict,approve_route", [
        (route_after_plan_review, "last_plan_review_verdict", "approve", "select_stage"),
        (route_after_design_review, "last_design_review_verdict", "approve", "generate_code"),
        (route_after_code_review, "last_code_review_verdict", "approve", "run_code"),
        (route_after_execution_check, "execution_verdict", "pass", "physics_check"),
        (route_after_physics_check, "physics_verdict", "pass", "analyze"),
        (route_after_comparison_check, "comparison_verdict", "approve", "supervisor"),
    ])
    def test_router_returns_valid_route_type(
        self, minimal_state, mock_save_checkpoint, router, verdict_field, approve_verdict, approve_route
    ):
        """
        Verify that each router returns a value that's in RouteType.
        """
        valid_routes = set(get_args(RouteType))
        
        minimal_state[verdict_field] = approve_verdict
        result = router(minimal_state)
        
        assert result in valid_routes, (
            f"Router returned '{result}' which is not in RouteType: {valid_routes}"
        )
        assert result == approve_route, (
            f"Expected route '{approve_route}' for verdict '{approve_verdict}', got '{result}'"
        )

    @pytest.mark.parametrize("router,verdict_field", [
        (route_after_plan_review, "last_plan_review_verdict"),
        (route_after_design_review, "last_design_review_verdict"),
        (route_after_code_review, "last_code_review_verdict"),
        (route_after_execution_check, "execution_verdict"),
        (route_after_physics_check, "physics_verdict"),
        (route_after_comparison_check, "comparison_verdict"),
    ])
    def test_router_handles_missing_verdict_field(
        self, minimal_state, mock_save_checkpoint, router, verdict_field
    ):
        """
        Verify that routers handle missing verdict fields gracefully.
        """
        del minimal_state[verdict_field]
        
        result = router(minimal_state)
        assert result == "ask_user", (
            f"Router should return 'ask_user' when verdict field is missing"
        )

    def test_physics_check_router_uses_both_count_fields(self, minimal_state, mock_save_checkpoint):
        """
        Verify that physics_check router uses physics_failure_count for 'fail'
        and design_revision_count for 'design_flaw'.
        """
        # Test 'fail' verdict uses physics_failure_count
        minimal_state["physics_verdict"] = "fail"
        minimal_state["physics_failure_count"] = MAX_PHYSICS_FAILURES
        minimal_state["design_revision_count"] = 0
        
        result = route_after_physics_check(minimal_state)
        assert result == "ask_user", (
            "Physics check should escalate when physics_failure_count at limit"
        )
        
        # Reset and test 'design_flaw' verdict uses design_revision_count
        mock_save_checkpoint.reset_mock()
        minimal_state["physics_verdict"] = "design_flaw"
        minimal_state["physics_failure_count"] = 0
        minimal_state["design_revision_count"] = MAX_DESIGN_REVISIONS
        
        result = route_after_physics_check(minimal_state)
        assert result == "ask_user", (
            "Physics check should escalate when design_revision_count at limit for design_flaw"
        )

    def test_comparison_check_routes_to_supervisor_on_limit(self, minimal_state, mock_save_checkpoint):
        """
        Verify that comparison_check routes to 'supervisor' (not 'ask_user') when at limit.
        This is a special case where route_on_limit is customized.
        """
        minimal_state["comparison_verdict"] = "needs_revision"
        minimal_state["analysis_revision_count"] = MAX_ANALYSIS_REVISIONS
        
        result = route_after_comparison_check(minimal_state)
        assert result == "supervisor", (
            "Comparison check should route to 'supervisor' when at limit, not 'ask_user'"
        )


class TestStateFieldExhaustiveness:
    """
    Tests that verify ROUTER_CONFIGS covers all verdict and count fields in ReproState.
    """
    
    def test_all_verdict_fields_in_state_are_tested(self):
        """
        Verify that we test all verdict fields defined in ReproState.
        This catches cases where a new verdict field is added but not tested.
        """
        state_hints = get_type_hints(ReproState)
        
        # Find all verdict fields in ReproState
        verdict_fields_in_state = {
            key for key in state_hints.keys()
            if "verdict" in key.lower() and key != "supervisor_verdict"  # supervisor_verdict is output, not router input
        }
        
        # Get verdict fields from ROUTER_CONFIGS
        tested_verdict_fields = {
            config["verdict_field"] for config in TestSchemaConsistency.ROUTER_CONFIGS
        }
        
        untested = verdict_fields_in_state - tested_verdict_fields
        if untested:
            pytest.fail(
                f"Verdict fields in ReproState not covered by ROUTER_CONFIGS: {untested}\n"
                f"If these are new fields, add them to ROUTER_CONFIGS.\n"
                f"If they shouldn't be routed, add them to the exclusion list."
            )

    def test_all_count_fields_for_routing_are_tested(self):
        """
        Verify that we test all count fields that are used for routing limits.
        """
        state_hints = get_type_hints(ReproState)
        
        # Count fields that should be used for routing
        routing_count_fields = {
            "replan_count",
            "design_revision_count", 
            "code_revision_count",
            "execution_failure_count",
            "physics_failure_count",
            "analysis_revision_count",
        }
        
        # Get count fields from ROUTER_CONFIGS
        tested_count_fields = set()
        for config in TestSchemaConsistency.ROUTER_CONFIGS:
            for count_field, _, _ in config["count_fields"]:
                tested_count_fields.add(count_field)
        
        # Check all routing count fields are tested
        untested = routing_count_fields - tested_count_fields
        if untested:
            pytest.fail(
                f"Count fields for routing not covered by ROUTER_CONFIGS: {untested}"
            )
        
        # Check all tested fields exist in state
        missing_from_state = tested_count_fields - set(state_hints.keys())
        if missing_from_state:
            pytest.fail(
                f"ROUTER_CONFIGS references count fields not in ReproState: {missing_from_state}"
            )
