"""Contract shape tests for reviewer and supervisor agents."""


class TestReviewerContract:
    """Test reviewer agent contract logic."""

    def test_approve_verdict_structure(self):
        """Approve verdict should have clean structure."""
        response = {"verdict": "approve", "issues": [], "summary": "LGTM"}
        assert response["verdict"] == "approve"
        assert not any(issue.get("severity") == "critical" for issue in response["issues"])

    def test_needs_revision_structure(self):
        """needs_revision verdict should include actionable feedback."""
        response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Fix X"}],
            "summary": "Needs work",
            "feedback": "Please address X",
        }
        assert response["verdict"] == "needs_revision"
        assert response["issues"] or response.get("feedback")


class TestSupervisorContract:
    """Test supervisor agent contract logic."""

    def test_ok_continue_allows_progression(self):
        response = {
            "verdict": "ok_continue",
            "summary": "Stage completed",
            "validation_hierarchy_status": {},
            "main_physics_assessment": {},
        }
        assert response["verdict"] == "ok_continue"
        assert not response.get("should_stop", False)

    def test_all_complete_stops_workflow(self):
        response = {
            "verdict": "all_complete",
            "should_stop": True,
            "stop_reason": "Done",
            "summary": "Complete",
        }
        assert response["verdict"] == "all_complete"
        assert response["should_stop"] is True

