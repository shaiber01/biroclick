"""Cross-field logic checks for LLM mock responses."""

import pytest

from .helpers import load_mock_response


class TestStrictCrossFieldConstraints:
    """Test logical dependencies between fields not captured by JSON Schema."""

    def test_reviewer_rejection_logic(self):
        """If a reviewer rejects (needs_revision), there MUST be issues listed."""
        reviewers = ["plan_reviewer", "design_reviewer", "code_reviewer"]

        for agent in reviewers:
            try:
                response = load_mock_response(agent)
            except FileNotFoundError:
                continue

            verdict = response.get("verdict")
            issues = response.get("issues", [])

            if verdict == "needs_revision":
                assert issues, f"{agent}: Verdict is 'needs_revision' but 'issues' list is empty."
                for issue in issues:
                    assert issue.get("description"), f"{agent}: Issue missing description."
            elif verdict == "approve":
                critical_issues = [i for i in issues if i.get("severity") == "critical"]
                assert not critical_issues, f"{agent}: Verdict is 'approve' but has critical issues."

    def test_supervisor_decision_consistency(self):
        """Supervisor decision fields must match the verdict."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock not found")

        verdict = response.get("verdict")

        if verdict == "backtrack_to_stage":
            assert "backtrack_decision" in response, "Missing 'backtrack_decision' for backtrack verdict."
            assert response["backtrack_decision"].get("target_stage_id"), "Backtrack target stage ID missing."
        elif verdict == "ask_user":
            assert response.get("user_question"), "Verdict is 'ask_user' but 'user_question' is empty/missing."
        elif verdict == "all_complete":
            assert response.get("should_stop") is True, "Verdict is 'all_complete' but 'should_stop' is not True."

    def test_planner_stages_integrity(self):
        """Planner stages must form a coherent plan."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock not found")

        stages = response.get("stages", [])

        if not stages:
            return

        stage_ids = {stage["stage_id"] for stage in stages}
        for stage in stages:
            for dep in stage.get("dependencies", []):
                assert dep in stage_ids, f"Stage {stage['stage_id']} depends on unknown stage {dep}"

        seen_ids = set()
        for stage in stages:
            missing_deps = [dep for dep in stage.get("dependencies", []) if dep not in seen_ids]
            assert not missing_deps, f"Stage {stage['stage_id']} appears before its dependencies: {missing_deps}"
            seen_ids.add(stage["stage_id"])

    def test_code_generator_safety_compliance(self):
        """Code generator must confirm safety checks are passed."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock not found")

        safety = response.get("safety_checks", {})
        for check, passed in safety.items():
            assert passed is True, f"Code generator failed safety check: {check}"

        runtime = response.get("estimated_runtime_minutes", 0)
        assert runtime > 0, "Estimated runtime must be positive"

    def test_simulation_designer_content(self):
        """Simulation design must be non-empty."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock not found")

        geometry = response.get("geometry", {})
        structures = geometry.get("structures", [])
        assert structures, "Simulation design has no structures"

        unit_system = response.get("unit_system", {})
        assert unit_system.get("characteristic_length_m", 0) > 0, "Characteristic length must be positive"

    def test_execution_validator_logic(self):
        """Execution validator consistency check."""
        try:
            response = load_mock_response("execution_validator")
        except FileNotFoundError:
            pytest.skip("Execution validator mock not found")

        exec_status = response.get("execution_status", {})
        verdict = response.get("verdict")

        if exec_status.get("completed") is False:
            assert verdict == "fail", "Verdict should be fail if execution not completed"

        files_check = response.get("files_check", {})
        expected = set(files_check.get("expected_files", []))
        found = set(files_check.get("found_files", []))
        if expected and expected.issubset(found):
            assert files_check.get("all_present") is True, "all_present should be True if all files found"

    def test_physics_sanity_logic(self):
        """Physics sanity logic check."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock not found")

        verdict = response.get("verdict")

        if verdict in ["fail", "design_flaw"]:
            concerns = response.get("concerns", [])
            failed_conservation = [c for c in response.get("conservation_checks", []) if c.get("status") == "fail"]
            failed_ranges = [c for c in response.get("value_range_checks", []) if c.get("status") == "fail"]
            assert (
                concerns or failed_conservation or failed_ranges
            ), "Failed physics verdict requires concerns or failed checks"

        if verdict == "pass":
            concerns = response.get("concerns", [])
            critical = [c for c in concerns if c.get("severity") == "critical"]
            assert not critical, "Pass verdict cannot have critical concerns"

