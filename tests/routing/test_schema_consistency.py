"""Tests that router configs stay in sync with schema definitions."""

from typing import get_type_hints

import pytest

from schemas.state import ReproState, RuntimeConfig


class TestSchemaConsistency:
    """
    Tests that router configurations match the ReproState and RuntimeConfig schemas.
    This ensures that fields accessed by routers actually exist.
    """

    def test_runtime_config_keys_exist(self):
        """
        Verify that max_count_keys used in routers exist in RuntimeConfig.
        Failure here means the router relies on a config key that isn't defined in the schema.
        """
        router_configs = [
            ("plan_review", "max_replans"),
            ("design_review", "max_design_revisions"),
            ("code_review", "max_code_revisions"),
            ("execution_check", "max_execution_failures"),
            ("physics_check", "max_physics_failures"),
            ("comparison_check", "max_analysis_revisions"),
        ]

        runtime_config_keys = get_type_hints(RuntimeConfig).keys()

        missing_keys = []
        for router_name, key in router_configs:
            if key not in runtime_config_keys:
                missing_keys.append(f"{router_name}: {key}")

        if missing_keys:
            pytest.fail(
                "Router config keys missing from RuntimeConfig schema: "
                + ", ".join(missing_keys)
            )

    def test_state_fields_exist(self):
        """
        Verify that verdict_field and count_field used in routers exist in ReproState.
        """
        router_configs = [
            ("plan_review", "last_plan_review_verdict", "replan_count"),
            ("design_review", "last_design_review_verdict", "design_revision_count"),
            ("code_review", "last_code_review_verdict", "code_revision_count"),
            ("execution_check", "execution_verdict", "execution_failure_count"),
            ("physics_check", "physics_verdict", "physics_failure_count"),
            ("comparison_check", "comparison_verdict", "analysis_revision_count"),
        ]

        state_keys = get_type_hints(ReproState).keys()

        missing_fields = []
        for router_name, verdict_field, count_field in router_configs:
            if verdict_field not in state_keys:
                missing_fields.append(f"{router_name} verdict: {verdict_field}")
            if count_field not in state_keys:
                missing_fields.append(f"{router_name} count: {count_field}")

        if missing_fields:
            pytest.fail(
                "Router fields missing from ReproState schema: "
                + ", ".join(missing_fields)
            )

