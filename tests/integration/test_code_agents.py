"""
Integration tests for code.py - code_generator_node and code_reviewer_node.

These tests verify the complete behavior of code agents including:
- Counter management (bounded increments, max limits)
- Verdict handling (approve vs needs_revision)
- User escalation when limits are reached
- LLM error handling paths
- Stub/empty code detection
- Design validation
- Context checking and decorators
"""

import json
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - COUNTER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeRevisionCounters:
    """Verify code reviewer counters respect bounds and increments."""

    def test_code_revision_counter_bounded_at_max(self, base_state):
        """Counter should NOT exceed max when already at limit."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Syntax error"}],
            "summary": "Fix the syntax errors",
            "feedback": "Please fix the identified issues",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        max_revisions = 5
        base_state["code_revision_count"] = max_revisions
        base_state["runtime_config"] = {"max_code_revisions": max_revisions}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Counter must be included and must NOT exceed max
        assert "code_revision_count" in result, "code_revision_count must be in result"
        assert result["code_revision_count"] == max_revisions, (
            f"Counter should stay at max ({max_revisions}), not increment further. "
            f"Got: {result['code_revision_count']}"
        )
        # Should escalate to user when at max and needs_revision
        assert result.get("ask_user_trigger") is not None, (
            "Should escalate to user when at max revisions"
        )
        assert result.get("ask_user_trigger") == "code_review_limit"

    def test_code_revision_counter_increments_under_max(self, base_state):
        """Counter should increment when below max."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "major", "description": "Missing import"}],
            "summary": "Add the missing import",
            "feedback": "Add import meep",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 2
        base_state["runtime_config"] = {"max_code_revisions": 5}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["code_revision_count"] == 3, (
            f"Counter should increment from 2 to 3. Got: {result['code_revision_count']}"
        )
        # Should NOT escalate when under max
        assert result.get("ask_user_trigger") is None, (
            "Should not escalate when under max revisions"
        )

    def test_code_revision_counter_starts_at_zero(self, base_state):
        """Counter should start at 0 and increment to 1 on first rejection."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "Code style issue"}],
            "summary": "Fix code style",
            "feedback": "Use consistent indentation",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        # Explicitly start with count NOT set (defaults to 0)
        assert "code_revision_count" not in base_state or base_state.get("code_revision_count", 0) == 0

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["code_revision_count"] == 1, (
            f"Counter should be 1 after first rejection. Got: {result['code_revision_count']}"
        )

    def test_code_revision_counter_uses_default_max_when_config_missing(self, base_state):
        """Counter should use default MAX_CODE_REVISIONS when not in config."""
        from src.agents.code import code_reviewer_node
        from schemas.state import MAX_CODE_REVISIONS

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Bug"}],
            "summary": "Fix bug",
            "feedback": "Critical bug found",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = MAX_CODE_REVISIONS  # At default max
        base_state["runtime_config"] = {}  # No max_code_revisions specified

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should escalate because at default max
        assert result.get("ask_user_trigger") is not None, (
            f"Should escalate at default max ({MAX_CODE_REVISIONS})"
        )
        assert result["code_revision_count"] == MAX_CODE_REVISIONS


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - VERDICT HANDLING
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewerVerdictHandling:
    """Verify correct behavior based on verdict."""

    def test_code_reviewer_approve_does_not_increment_counter(self, base_state):
        """Counter should NOT increment when verdict is approve."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('valid code')"
        base_state["code_revision_count"] = 2

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Counter should stay the same, not increment
        assert result["code_revision_count"] == 2, (
            f"Counter should NOT increment on approval. Expected 2, got: {result['code_revision_count']}"
        )
        assert result["last_code_review_verdict"] == "approve"
        # Should NOT trigger user escalation
        assert result.get("ask_user_trigger") is None

    def test_code_reviewer_approve_sets_correct_verdict(self, base_state):
        """Verdict field should match the LLM response."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "All checks passed",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nsim = mp.Simulation()"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "approve", (
            f"Expected 'approve', got: {result.get('last_code_review_verdict')}"
        )

    def test_code_reviewer_needs_revision_sets_correct_verdict(self, base_state):
        """Verdict field should be 'needs_revision' when LLM returns that."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Missing simulation setup"}],
            "summary": "Code is incomplete",
            "feedback": "Add proper Meep simulation setup",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('incomplete')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision"

    def test_code_reviewer_missing_verdict_defaults_to_needs_revision(self, base_state):
        """If LLM doesn't return verdict, should default to needs_revision."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "issues": [{"severity": "minor", "description": "Style issue"}],
            "summary": "Some issues found",
            # No "verdict" key
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision", (
            "Missing verdict should default to needs_revision"
        )

    def test_code_reviewer_normalizes_pass_to_approve(self, base_state):
        """Verdict 'pass' should be normalized to 'approve'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "pass",  # Alternative spelling
            "issues": [],
            "summary": "Code passes review",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "approve", (
            "'pass' should be normalized to 'approve'"
        )

    def test_code_reviewer_normalizes_reject_to_needs_revision(self, base_state):
        """Verdict 'reject' should be normalized to 'needs_revision'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "reject",  # Alternative spelling
            "issues": [{"severity": "major", "description": "Bug"}],
            "summary": "Code rejected",
            "feedback": "Fix the bug",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('broken')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision", (
            "'reject' should be normalized to 'needs_revision'"
        )

    def test_code_reviewer_normalizes_approved_to_approve(self, base_state):
        """Verdict 'approved' should be normalized to 'approve'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approved",  # Past tense
            "issues": [],
            "summary": "Approved",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "approve", (
            "'approved' should be normalized to 'approve'"
        )

    def test_code_reviewer_normalizes_accept_to_approve(self, base_state):
        """Verdict 'accept' should be normalized to 'approve'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "accept",
            "issues": [],
            "summary": "Accepted",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "approve", (
            "'accept' should be normalized to 'approve'"
        )

    def test_code_reviewer_normalizes_unknown_verdict_to_needs_revision(self, base_state):
        """Unknown verdict values should default to 'needs_revision' (safer)."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "maybe_ok",  # Unknown verdict
            "issues": [],
            "summary": "Unclear",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision", (
            "Unknown verdict should default to 'needs_revision' for safety"
        )

    def test_code_reviewer_normalizes_revision_needed_to_needs_revision(self, base_state):
        """Verdict 'revision_needed' should be normalized to 'needs_revision'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "revision_needed",  # Alternative spelling
            "issues": [{"severity": "minor", "description": "Issue"}],
            "summary": "Needs work",
            "feedback": "Fix this",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision", (
            "'revision_needed' should be normalized to 'needs_revision'"
        )

    def test_code_reviewer_normalizes_needs_work_to_needs_revision(self, base_state):
        """Verdict 'needs_work' should be normalized to 'needs_revision'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_work",
            "issues": [{"severity": "minor", "description": "Issue"}],
            "summary": "Work needed",
            "feedback": "More work required",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["last_code_review_verdict"] == "needs_revision", (
            "'needs_work' should be normalized to 'needs_revision'"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - FEEDBACK AND OUTPUT FIELDS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewerFeedback:
    """Verify code reviewer feedback fields are preserved correctly."""

    def test_code_reviewer_populates_feedback_on_rejection(self, base_state):
        """Feedback should be populated with meaningful content on rejection."""
        from src.agents.code import code_reviewer_node

        expected_feedback = "The code is missing the import statement for numpy"
        mock_response = {
            "verdict": "needs_revision",
            "issues": [
                {"severity": "critical", "description": "Missing import statement"},
                {"severity": "major", "description": "Incorrect parameter value"},
            ],
            "summary": "Code has critical issues",
            "feedback": expected_feedback,
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('incomplete code')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert "reviewer_feedback" in result, "reviewer_feedback should be in result"
        feedback = result["reviewer_feedback"]
        # Should use the feedback field from response
        assert expected_feedback in feedback or len(feedback) > 20, (
            f"Feedback should contain meaningful content. Got: {feedback}"
        )

    def test_code_reviewer_uses_summary_when_feedback_missing(self, base_state):
        """Should fall back to summary when feedback key is missing."""
        from src.agents.code import code_reviewer_node

        expected_summary = "Critical issues found in the code"
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Bug"}],
            "summary": expected_summary,
            # No "feedback" key
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert "reviewer_feedback" in result
        # Should use summary as fallback
        assert expected_summary in result["reviewer_feedback"], (
            f"Should use summary as fallback. Got: {result['reviewer_feedback']}"
        )

    def test_code_reviewer_issues_passed_through(self, base_state):
        """Issues array should be passed through to result."""
        from src.agents.code import code_reviewer_node

        expected_issues = [
            {"severity": "critical", "description": "Missing import"},
            {"severity": "major", "description": "Wrong parameter"},
        ]
        mock_response = {
            "verdict": "needs_revision",
            "issues": expected_issues,
            "summary": "Issues found",
            "feedback": "Fix these issues",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert "reviewer_issues" in result, "reviewer_issues should be in result"
        assert result["reviewer_issues"] == expected_issues, (
            f"Issues should be passed through. Expected: {expected_issues}, Got: {result['reviewer_issues']}"
        )


class TestCodeReviewerOutputFields:
    """Verify code reviewer sets all required output fields."""

    def test_code_reviewer_sets_all_fields_on_approve(self, base_state):
        """All required fields should be set when code is approved."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code looks good",
        }

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('good code')"
        base_state["code_revision_count"] = 3

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Check all required fields
        assert result.get("last_code_review_verdict") == "approve"
        assert result.get("workflow_phase") == "code_review", (
            f"workflow_phase should be 'code_review'. Got: {result.get('workflow_phase')}"
        )
        assert "code_revision_count" in result
        assert "reviewer_issues" in result
        assert result["reviewer_issues"] == [], "Issues should be empty on approval"

    def test_code_reviewer_sets_workflow_phase(self, base_state):
        """workflow_phase should always be set to 'code_review'."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "OK",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result["workflow_phase"] == "code_review"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - USER ESCALATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewerEscalation:
    """Verify user escalation when max revisions reached."""

    def test_code_reviewer_escalates_at_max_revisions(self, base_state):
        """Should trigger ask_user when at max revisions and needs_revision."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Cannot fix automatically"}],
            "summary": "Persistent issues",
            "feedback": "This issue requires manual intervention",
        }
        max_revs = 3
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('broken')"
        base_state["code_revision_count"] = max_revs
        base_state["runtime_config"] = {"max_code_revisions": max_revs}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should escalate to user
        assert result.get("ask_user_trigger") is not None, "Should be awaiting user input"
        assert result.get("ask_user_trigger") == "code_review_limit", (
            f"ask_user_trigger should be 'code_review_limit'. Got: {result.get('ask_user_trigger')}"
        )
        assert "pending_user_questions" in result
        assert len(result["pending_user_questions"]) > 0, "Should have at least one question"
        # Question should mention the stage and attempts
        question = result["pending_user_questions"][0]
        assert "stage_0" in question, "Question should mention the stage"
        assert str(max_revs) in question, "Question should mention attempt count"

    def test_code_reviewer_escalation_includes_feedback(self, base_state):
        """Escalation question should include reviewer feedback."""
        from src.agents.code import code_reviewer_node

        expected_feedback = "The simulation setup is fundamentally incorrect"
        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Wrong setup"}],
            "summary": "Critical errors",
            "feedback": expected_feedback,
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('wrong')"
        base_state["code_revision_count"] = 3
        base_state["runtime_config"] = {"max_code_revisions": 3}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        question = result["pending_user_questions"][0]
        # Feedback should be included in the question
        assert expected_feedback in question or "feedback" in question.lower(), (
            f"Question should include reviewer feedback. Question: {question}"
        )

    def test_code_reviewer_sets_last_node_on_escalation(self, base_state):
        """Should set last_node_before_ask_user when escalating."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Bug"}],
            "summary": "Issues",
            "feedback": "Fix this",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 5
        base_state["runtime_config"] = {"max_code_revisions": 5}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        assert result.get("last_node_before_ask_user") == "code_review", (
            f"Should set last_node_before_ask_user to 'code_review'. "
            f"Got: {result.get('last_node_before_ask_user')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - LLM ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewerLLMErrors:
    """Verify LLM error handling in code reviewer."""

    def test_code_reviewer_defaults_to_needs_revision_on_llm_error(self, base_state):
        """Should default to needs_revision (fail-closed) when LLM call fails."""
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nsim = mp.Simulation()"
        initial_count = base_state.get("code_revision_count", 0)

        # Make the LLM call raise an exception
        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("API rate limit exceeded")
        ):
            result = code_reviewer_node(base_state)

        # Fail-closed: LLM error should trigger needs_revision (safer than auto-approve)
        assert result.get("last_code_review_verdict") == "needs_revision", (
            f"Should default to needs_revision on LLM error. Got: {result.get('last_code_review_verdict')}"
        )
        # Counter should be incremented for needs_revision
        assert result["code_revision_count"] == initial_count + 1

    def test_code_reviewer_llm_error_includes_issue_note(self, base_state):
        """Auto-approval on LLM error should include note in issues."""
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("Connection timeout")
        ):
            result = code_reviewer_node(base_state)

        # Should have issues with LLM unavailability note
        issues = result.get("reviewer_issues", [])
        assert len(issues) > 0, "Should have at least one issue noting LLM unavailability"
        # Check that the issue mentions LLM unavailability
        issue_str = str(issues)
        assert "LLM" in issue_str or "unavailable" in issue_str.lower(), (
            f"Issue should mention LLM unavailability. Got: {issues}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEWER - USER CONTENT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeReviewerUserContent:
    """Verify code reviewer receives the required design context."""

    def test_code_reviewer_receives_code_in_user_content(self, base_state):
        """User content should include the code to review."""
        from src.agents.code import code_reviewer_node

        code_to_review = "import meep as mp\nsim = mp.Simulation(cell_size=mp.Vector3(10,10,10))"
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = code_to_review

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        assert code_to_review in user_content, (
            f"User content should include the code to review. "
            f"Code: {code_to_review[:50]}..., User content: {user_content[:100]}..."
        )

    def test_code_reviewer_receives_design_spec(self, base_state):
        """User content should include design specification."""
        from src.agents.code import code_reviewer_node

        design = {
            "stage_id": "stage_0",
            "design_description": "FDTD for gold nanorod",
            "geometry": [{"type": "cylinder", "radius": 20, "height": 100}],
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('test')"
        base_state["design_description"] = design

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        # Design spec should be in user content
        assert "DESIGN" in user_content.upper() or "design" in user_content.lower(), (
            "User content should mention design"
        )
        # Should include the geometry details
        assert "cylinder" in user_content or "geometry" in user_content.lower()

    def test_code_reviewer_includes_previous_feedback(self, base_state):
        """User content should include previous revision feedback."""
        from src.agents.code import code_reviewer_node

        previous_feedback = "Previous issue: Missing import statement for numpy"
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nimport numpy as np"
        base_state["reviewer_feedback"] = previous_feedback

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        assert previous_feedback in user_content, (
            f"User content should include previous feedback: {previous_feedback}"
        )

    def test_code_reviewer_uses_correct_agent_name(self, base_state):
        """Should call LLM with agent_name='code_reviewer'."""
        from src.agents.code import code_reviewer_node

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "code_reviewer", (
            f"agent_name should be 'code_reviewer'. Got: {call_kwargs.get('agent_name')}"
        )

    def test_code_reviewer_handles_string_design(self, base_state):
        """Should handle design_description as plain string."""
        from src.agents.code import code_reviewer_node

        string_design = "This is a plain string design description for FDTD simulation"
        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"
        base_state["design_description"] = string_design

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_reviewer_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        # String design should be included directly
        assert string_design in user_content, (
            "String design should be included in user content"
        )

    def test_code_reviewer_handles_empty_code(self, base_state):
        """Should handle empty code string."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "No code provided"}],
            "summary": "Empty code",
            "feedback": "Please provide actual code",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = ""  # Empty

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should still process and return valid result
        assert "last_code_review_verdict" in result
        assert "workflow_phase" in result

    def test_code_reviewer_handles_none_design(self, base_state):
        """Should handle None design_description gracefully."""
        from src.agents.code import code_reviewer_node

        mock_response = {"verdict": "approve", "issues": [], "summary": "OK"}
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"
        base_state["design_description"] = None

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            result = code_reviewer_node(base_state)

        # Should succeed without crashing
        assert "last_code_review_verdict" in result
        # User content should not include design section
        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        # Design should not be added if None
        assert "```json" not in user_content or "null" in user_content.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - BASIC FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorBehavior:
    """Verify code generator basic functionality."""

    def test_code_generator_node_creates_code(self, base_state, valid_plan):
        """Code generator should produce code and set all required fields."""
        from src.agents.code import code_generator_node

        expected_code = "import meep as mp\nimport numpy as np\nprint('Simulation started')"
        expected_outputs = ["output.csv", "spectrum.png"]
        mock_response = {
            "code": expected_code,
            "expected_outputs": expected_outputs,
            "explanation": "Simple FDTD test simulation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod extinction spectrum measurement",
            "geometry": [{"type": "cylinder", "radius": 20, "height": 100}],
            "sources": [{"type": "gaussian", "wavelength_range": [400, 900]}],
            "monitors": [{"type": "flux", "name": "transmission"}],
        }
        base_state["validated_materials"] = [
            {"material_id": "gold", "path": "/materials/Au.csv"}
        ]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Check code is in result and matches
        assert "code" in result, "Result must include 'code'"
        assert result["code"] == expected_code, (
            f"Code should match mock response. Expected: {expected_code}, Got: {result['code']}"
        )
        # Check workflow_phase
        assert result["workflow_phase"] == "code_generation", (
            f"workflow_phase should be 'code_generation'. Got: {result.get('workflow_phase')}"
        )
        # Check expected_outputs
        assert result["expected_outputs"] == expected_outputs, (
            f"expected_outputs should be {expected_outputs}. Got: {result.get('expected_outputs')}"
        )

    def test_code_generator_requires_validated_materials_for_stage1(
        self, base_state, valid_plan
    ):
        """Stage 1+ should require validated_materials."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_1"
        base_state["current_stage_type"] = "SINGLE_STRUCTURE"
        base_state["design_description"] = {
            "stage_id": "stage_1",
            "design_description": "FDTD simulation for the main structure analysis",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = []  # Empty materials

        result = code_generator_node(base_state)

        # Should return error or not produce code
        assert "code" not in result or result.get("run_error"), (
            "Should fail or set run_error when validated_materials is empty for Stage 1+"
        )
        if "run_error" in result:
            assert "validated_materials" in result["run_error"].lower() or "material" in result["run_error"].lower(), (
                f"Error should mention missing materials. Got: {result['run_error']}"
            )

    def test_code_generator_allows_empty_materials_for_stage0(self, base_state, valid_plan):
        """Stage 0 (MATERIAL_VALIDATION) should work without validated_materials."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nprint('material validation')",
            "expected_outputs": [],
            "explanation": "Material validation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Material validation simulation for gold optical properties",
            "geometry": [{"type": "box"}],
        }
        base_state["validated_materials"] = []  # Empty is OK for Stage 0

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should succeed
        assert "code" in result, "Stage 0 should produce code without validated_materials"
        assert result.get("run_error") is None, "Should not have run_error"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - EXPECTED OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorExpectedOutputs:
    """Verify code_generator expected outputs handling."""

    def test_expected_outputs_passed_through(self, base_state, valid_plan):
        """Expected outputs from LLM should be passed through."""
        from src.agents.code import code_generator_node

        expected = ["spectrum.csv", "field_map.png", "resonance_data.json"]
        mock_response = {
            "code": "import meep as mp\nimport numpy as np\n\nprint('Running simulation')",
            "expected_outputs": expected,
            "explanation": "Test simulation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod optical properties",
            "geometry": [{"type": "cylinder", "radius": 20}],
            "sources": [{"type": "gaussian"}],
            "monitors": [{"type": "flux"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result["expected_outputs"] == expected, (
            f"expected_outputs should match. Expected: {expected}, Got: {result.get('expected_outputs')}"
        )

    def test_empty_expected_outputs_defaults_to_empty_list(self, base_state, valid_plan):
        """Missing expected_outputs should default to empty list."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nimport numpy as np\nprint('Simulation')",
            "explanation": "Test",
            # No "expected_outputs" key
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Test design description for simulation code generation",
            "geometry": [{"type": "box"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result["expected_outputs"] == [], (
            f"Missing expected_outputs should default to []. Got: {result.get('expected_outputs')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - VALIDATION AND EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorValidation:
    """Verify code generator input validation."""

    def test_code_generator_fails_without_stage_id(self, base_state, valid_plan):
        """Should fail/escalate when current_stage_id is None."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = None  # Missing!
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Test description for code generation process",
            "geometry": [{"type": "box"}],
        }

        result = code_generator_node(base_state)

        # Should escalate to user or indicate error
        assert result.get("ask_user_trigger") is not None or "code" not in result, (
            "Should escalate or fail when current_stage_id is None"
        )
        if result.get("awaiting_user_input"):
            assert result.get("ask_user_trigger") == "missing_stage_id"

    def test_code_generator_with_stub_design(self, base_state, valid_plan):
        """Should fail when design contains stub markers."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "TODO: Add design",  # Stub marker
            "geometry": [],
        }

        result = code_generator_node(base_state)

        # Should not produce code or should have error
        assert "code" not in result or result.get("run_error") or result.get("reviewer_feedback"), (
            "Should fail when design is a stub"
        )
        # Should increment design_revision_count
        if "design_revision_count" in result:
            assert result["design_revision_count"] >= 1

    def test_code_generator_with_empty_design(self, base_state, valid_plan):
        """Should fail when design_description is empty."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = ""  # Empty

        result = code_generator_node(base_state)

        assert "code" not in result or result.get("reviewer_feedback"), (
            "Should fail when design is empty"
        )

    def test_code_generator_with_placeholder_design(self, base_state, valid_plan):
        """Should fail when design contains PLACEHOLDER marker."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "PLACEHOLDER - design would be generated here",
            "geometry": [],
        }

        result = code_generator_node(base_state)

        assert "code" not in result or result.get("reviewer_feedback"), (
            "Should fail when design contains PLACEHOLDER"
        )

    def test_code_generator_with_short_design(self, base_state, valid_plan):
        """Should fail when design is too short (< 50 chars)."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {"design_description": "Short"}  # Too short

        result = code_generator_node(base_state)

        assert "code" not in result or result.get("reviewer_feedback"), (
            "Should fail when design is too short"
        )

    def test_code_generator_stub_design_increments_design_revision_count(self, base_state, valid_plan):
        """Stub design should increment design_revision_count."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "STUB: design would go here",
            "geometry": [],
        }
        base_state["design_revision_count"] = 0

        result = code_generator_node(base_state)

        # Should increment design_revision_count
        assert result.get("design_revision_count", 0) >= 1, (
            "Should increment design_revision_count for stub design"
        )
        # Should set supervisor_verdict to ok_continue
        assert result.get("supervisor_verdict") == "ok_continue", (
            "Should set supervisor_verdict to ok_continue for stub design"
        )

    def test_code_generator_stub_design_respects_max_design_revisions(self, base_state, valid_plan):
        """Design revision count should be bounded by max_design_revisions."""
        from src.agents.code import code_generator_node
        from schemas.state import MAX_DESIGN_REVISIONS

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "TODO: add design",
            "geometry": [],
        }
        base_state["design_revision_count"] = MAX_DESIGN_REVISIONS  # Already at max
        base_state["runtime_config"] = {"max_design_revisions": MAX_DESIGN_REVISIONS}

        result = code_generator_node(base_state)

        # Should not exceed max
        assert result.get("design_revision_count") == MAX_DESIGN_REVISIONS, (
            f"design_revision_count should not exceed max ({MAX_DESIGN_REVISIONS})"
        )

    def test_code_generator_with_none_design(self, base_state, valid_plan):
        """Should handle None design_description."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = None

        result = code_generator_node(base_state)

        # Should fail gracefully
        assert "code" not in result or result.get("reviewer_feedback"), (
            "Should fail when design is None"
        )


class TestCodeGeneratorStubDetection:
    """Verify stub/empty code detection in generated code."""

    def test_code_generator_detects_stub_code_starting_with_hash_stub(self, base_state, valid_plan):
        """Should detect and reject stub code starting with # STUB."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "# STUB: This would be the simulation code",
            "expected_outputs": [],
            "explanation": "Stub code",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should have error or feedback indicating stub detection
        assert result.get("reviewer_feedback") or result.get("code_revision_count", 0) > 0, (
            "Should detect and flag stub code"
        )
        if result.get("reviewer_feedback"):
            assert "stub" in result["reviewer_feedback"].lower() or "empty" in result["reviewer_feedback"].lower()

    def test_code_generator_detects_stub_code_starting_with_placeholder(self, base_state, valid_plan):
        """Should detect code starting with PLACEHOLDER as stub."""
        from src.agents.code import code_generator_node

        # Long code but starts with PLACEHOLDER
        mock_response = {
            "code": "# PLACEHOLDER: Replace this with actual simulation code\nimport meep\n" + "x = 1\n" * 50,
            "expected_outputs": [],
            "explanation": "Placeholder code",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should detect PLACEHOLDER at start even for long code
        assert result.get("reviewer_feedback") is not None, (
            "Should detect code starting with PLACEHOLDER marker"
        )

    def test_code_generator_detects_empty_code(self, base_state, valid_plan):
        """Should detect and reject empty code."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "",  # Empty
            "expected_outputs": [],
            "explanation": "Empty code",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result.get("reviewer_feedback"), "Should detect empty code"

    def test_code_generator_detects_short_stub_code(self, base_state, valid_plan):
        """Should detect short code with TODO marker as stub."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "# TODO: implement simulation",  # Short code with TODO
            "expected_outputs": [],
            "explanation": "Placeholder",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result.get("reviewer_feedback"), "Should detect short stub code"

    def test_code_generator_allows_todo_in_long_valid_code(self, base_state, valid_plan):
        """Should allow TODO comments in otherwise valid, long code."""
        from src.agents.code import code_generator_node

        valid_code = """import meep as mp
import numpy as np

# Set up simulation parameters
resolution = 10
cell_size = mp.Vector3(16, 8, 0)

# TODO: Consider increasing resolution for better accuracy
# This is a valid comment in production code

# Create geometry
geometry = [
    mp.Cylinder(radius=2, height=mp.inf, material=mp.metal)
]

# Set up simulation
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    resolution=resolution
)

# Run and save results
sim.run(until=200)
print("Simulation complete")
"""
        mock_response = {
            "code": valid_code,
            "expected_outputs": ["output.csv"],
            "explanation": "Valid simulation",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should accept valid code with TODO in comments
        assert result.get("code") == valid_code, (
            "Should accept valid code with TODO in comments"
        )
        assert result.get("reviewer_feedback") is None, "Should not flag valid code as stub"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - LLM ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorLLMErrors:
    """Verify LLM error handling in code generator."""

    def test_code_generator_escalates_on_llm_error(self, base_state, valid_plan):
        """Should escalate to user when LLM call fails."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("API connection failed")
        ):
            result = code_generator_node(base_state)

        # Should escalate to user
        assert result.get("ask_user_trigger") is not None, (
            "Should escalate to user on LLM error"
        )
        assert result.get("ask_user_trigger") == "llm_error", (
            f"ask_user_trigger should be 'llm_error'. Got: {result.get('ask_user_trigger')}"
        )
        assert "pending_user_questions" in result
        question = result["pending_user_questions"][0]
        assert "failed" in question.lower() or "error" in question.lower(), (
            f"Question should mention failure. Got: {question}"
        )

    def test_code_generator_llm_error_preserves_workflow_phase(self, base_state, valid_plan):
        """workflow_phase should be preserved when LLM error occurs."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch(
            "src.agents.code.call_agent_with_metrics",
            side_effect=Exception("Timeout")
        ):
            result = code_generator_node(base_state)

        assert result.get("workflow_phase") == "code_generation", (
            f"workflow_phase should be 'code_generation'. Got: {result.get('workflow_phase')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - ALTERNATIVE CODE KEYS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorAlternativeKeys:
    """Verify code generator handles alternative response keys."""

    def test_code_generator_uses_simulation_code_key(self, base_state, valid_plan):
        """Should use 'simulation_code' key if 'code' is missing."""
        from src.agents.code import code_generator_node

        expected_code = "import meep as mp\nsim = mp.Simulation()"
        mock_response = {
            "simulation_code": expected_code,  # Alternative key
            "expected_outputs": [],
            "explanation": "Using simulation_code key",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        assert result.get("code") == expected_code, (
            f"Should use simulation_code. Expected: {expected_code}, Got: {result.get('code')}"
        )

    def test_code_generator_falls_back_to_json_dump(self, base_state, valid_plan):
        """Should JSON dump response if neither code nor simulation_code exists."""
        from src.agents.code import code_generator_node

        mock_response = {
            "other_key": "some value",
            "explanation": "No code key",
            # Neither "code" nor "simulation_code"
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should fall back to JSON dump, then be detected as stub/invalid
        # Either the code is the JSON dump or there's an error
        if "code" in result:
            # JSON dump would be short and not valid code
            assert result.get("reviewer_feedback") or "other_key" in result["code"]

    def test_code_generator_handles_string_response(self, base_state, valid_plan):
        """Should handle non-dict (string) response from LLM."""
        from src.agents.code import code_generator_node

        # LLM returns raw string instead of dict
        string_response = "import meep as mp\nsim = mp.Simulation(cell_size=mp.Vector3(10,10,10), resolution=10)\nsim.run(until=100)"

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=string_response):
            result = code_generator_node(base_state)

        # Should convert string to code
        assert "code" in result, "Should include code from string response"
        assert result["code"] == string_response, (
            f"Code should match string response. Got: {result.get('code')}"
        )

    def test_code_generator_handles_empty_code_key(self, base_state, valid_plan):
        """Should handle empty string in 'code' key."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "",  # Empty string
            "expected_outputs": [],
            "explanation": "Empty code",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_generator_node(base_state)

        # Should detect empty code and flag it
        assert result.get("reviewer_feedback") is not None, (
            "Should detect and flag empty code"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR - PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorPromptBuilding:
    """Verify prompts and schema selection for code generator."""

    def test_code_generator_uses_correct_agent_name(self, base_state, valid_plan):
        """Should call LLM with agent_name='code_generator'."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nprint('test')",
            "expected_outputs": ["output.csv"],
            "explanation": "Test",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod optical properties",
            "geometry": [{"type": "cylinder", "radius": 20}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)

        assert mock.called
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("agent_name") == "code_generator", (
            f"agent_name should be 'code_generator'. Got: {call_kwargs.get('agent_name')}"
        )
        system_prompt = call_kwargs.get("system_prompt", "")
        assert len(system_prompt) > 100, (
            f"System prompt too short for code_generator. Length: {len(system_prompt)}"
        )

    def test_code_generator_includes_revision_feedback_in_prompt(self, base_state, valid_plan):
        """Revision feedback should be appended to system prompt."""
        from src.agents.code import code_generator_node

        revision_feedback = "Please add proper error handling for the flux calculation"
        mock_response = {
            "code": "import meep as mp\ntry:\n    sim = mp.Simulation()\nexcept:\n    pass",
            "expected_outputs": [],
            "explanation": "Added error handling",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod optical properties",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["reviewer_feedback"] = revision_feedback

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)

        call_kwargs = mock.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        # Feedback should be in system prompt
        assert revision_feedback in system_prompt, (
            f"Revision feedback should be in system prompt. Feedback: {revision_feedback}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR AND CONTEXT CHECK TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestWithContextCheckDecorator:
    """Verify @with_context_check decorator behavior."""

    def test_code_reviewer_returns_empty_when_awaiting_user_input(self, base_state):
        """Should return empty dict when awaiting_user_input is True."""
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"
        base_state["awaiting_user_input"] = True  # Already awaiting

        # Should NOT call LLM, should return empty
        with patch("src.agents.code.call_agent_with_metrics") as mock:
            result = code_reviewer_node(base_state)

        assert result == {}, (
            f"Should return empty dict when awaiting_user_input. Got: {result}"
        )
        mock.assert_not_called()

    def test_code_reviewer_handles_context_escalation(self, base_state):
        """Should return escalation when context check fails."""
        from src.agents.code import code_reviewer_node

        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"

        # Mock context check to return escalation
        escalation_result = {
            "pending_user_questions": ["Context overflow - how to proceed?"],
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "context_overflow",
        }

        with patch(
            "src.agents.base.check_context_or_escalate",
            return_value=escalation_result
        ):
            result = code_reviewer_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "context_overflow"


class TestCodeGeneratorContextCheck:
    """Verify context checking in code_generator_node."""

    def test_code_generator_returns_escalation_on_context_overflow(self, base_state, valid_plan):
        """Should return escalation dict when context check fails."""
        from src.agents.code import code_generator_node

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        # Mock context check to return escalation
        escalation_result = {
            "pending_user_questions": ["Context overflow in generate_code"],
            "ask_user_trigger": "context_overflow",
            "ask_user_trigger": "context_overflow",
        }

        with patch(
            "src.agents.code.check_context_or_escalate",
            return_value=escalation_result
        ):
            result = code_generator_node(base_state)

        assert result.get("ask_user_trigger") is not None
        assert result.get("ask_user_trigger") == "context_overflow"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS - FULL FLOW
# ═══════════════════════════════════════════════════════════════════════════════


class TestCodeGeneratorUserContent:
    """Verify user content building for code generator."""

    def test_code_generator_user_content_includes_stage(self, base_state, valid_plan):
        """User content should include current stage ID."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nprint('test')",
            "expected_outputs": [],
            "explanation": "Test",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_42"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_42",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        assert "stage_42" in user_content, (
            f"User content should include stage ID. Got: {user_content[:200]}"
        )

    def test_code_generator_user_content_includes_materials(self, base_state, valid_plan):
        """User content should include validated materials."""
        from src.agents.code import code_generator_node

        mock_response = {
            "code": "import meep as mp\nprint('test')",
            "expected_outputs": [],
            "explanation": "Test",
        }

        materials = [
            {"material_id": "gold", "path": "/materials/gold.csv"},
            {"material_id": "silver", "path": "/materials/silver.csv"},
        ]
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "Full design description for FDTD simulation of gold nanorod",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = materials

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response) as mock:
            code_generator_node(base_state)

        call_kwargs = mock.call_args.kwargs
        user_content = call_kwargs.get("user_content", "")
        # Materials should be in user content
        assert "gold" in user_content and "silver" in user_content, (
            f"User content should include material IDs. Got: {user_content[:300]}"
        )


class TestCodeReviewerBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_code_reviewer_counter_at_exactly_max_minus_one(self, base_state):
        """Counter at max-1 should increment to max AND escalate (budget exhausted)."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "Issue"}],
            "summary": "Issues",
            "feedback": "Fix",
        }
        max_revs = 5
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = max_revs - 1  # One below max
        base_state["runtime_config"] = {"max_code_revisions": max_revs}

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should increment to max
        assert result["code_revision_count"] == max_revs, (
            f"Should increment to max. Expected {max_revs}, got {result['code_revision_count']}"
        )
        # SHOULD escalate - when counter reaches max, revision budget is exhausted
        assert result.get("ask_user_trigger") is not None, (
            "Should escalate when counter reaches max (budget exhausted)"
        )
        assert result.get("ask_user_trigger") == "code_review_limit"

    def test_code_reviewer_with_zero_max_revisions(self, base_state):
        """Should handle zero max_revisions edge case."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "minor", "description": "Issue"}],
            "summary": "Issues",
            "feedback": "Fix",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "print('test')"
        base_state["code_revision_count"] = 0
        base_state["runtime_config"] = {"max_code_revisions": 0}  # Zero max

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should immediately escalate since max is 0
        assert result.get("ask_user_trigger") is not None, (
            "Should escalate immediately when max_revisions is 0"
        )

    def test_code_reviewer_preserves_other_state_fields(self, base_state):
        """Result should not accidentally include/exclude unexpected fields."""
        from src.agents.code import code_reviewer_node

        mock_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "OK",
        }
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp"
        base_state["some_other_field"] = "should not appear in result"

        with patch("src.agents.code.call_agent_with_metrics", return_value=mock_response):
            result = code_reviewer_node(base_state)

        # Should only include expected output fields
        expected_fields = {
            "workflow_phase", "last_code_review_verdict",
            "reviewer_issues", "code_revision_count"
        }
        for field in expected_fields:
            assert field in result, f"Missing expected field: {field}"

        # Should NOT include input state fields
        assert "some_other_field" not in result, (
            "Result should not include unrelated state fields"
        )


class TestCodeAgentsIntegration:
    """Integration tests for code agents working together."""

    def test_code_generator_then_reviewer_approve_flow(self, base_state, valid_plan):
        """Test the typical flow: generate code -> review -> approve."""
        from src.agents.code import code_generator_node, code_reviewer_node

        generated_code = """import meep as mp
import numpy as np

sim = mp.Simulation(
    cell_size=mp.Vector3(10, 10, 10),
    resolution=10
)

sim.run(until=100)
print("Done")
"""
        gen_response = {
            "code": generated_code,
            "expected_outputs": ["output.csv"],
            "explanation": "FDTD simulation",
        }
        review_response = {
            "verdict": "approve",
            "issues": [],
            "summary": "Code is correct",
        }

        # Setup state for code generation
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod optical properties analysis",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]

        # Generate code
        with patch("src.agents.code.call_agent_with_metrics", return_value=gen_response):
            gen_result = code_generator_node(base_state)

        assert gen_result.get("code") == generated_code
        assert gen_result.get("workflow_phase") == "code_generation"

        # Update state with generated code
        base_state["code"] = gen_result["code"]

        # Review code
        with patch("src.agents.code.call_agent_with_metrics", return_value=review_response):
            review_result = code_reviewer_node(base_state)

        assert review_result.get("last_code_review_verdict") == "approve"
        assert review_result.get("workflow_phase") == "code_review"
        assert review_result.get("ask_user_trigger") is None

    def test_code_generator_then_reviewer_revision_flow(self, base_state, valid_plan):
        """Test revision flow: generate -> review (reject) -> feedback."""
        from src.agents.code import code_generator_node, code_reviewer_node

        generated_code = "import meep as mp\nprint('incomplete')"
        gen_response = {
            "code": generated_code,
            "expected_outputs": [],
            "explanation": "Initial code",
        }
        review_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Missing simulation setup"}],
            "summary": "Code is incomplete",
            "feedback": "Add proper Meep Simulation setup with cell_size and resolution",
        }

        # Setup state
        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["current_stage_type"] = "MATERIAL_VALIDATION"
        base_state["design_description"] = {
            "stage_id": "stage_0",
            "design_description": "FDTD simulation for gold nanorod optical properties analysis",
            "geometry": [{"type": "cylinder"}],
        }
        base_state["validated_materials"] = [{"material_id": "gold"}]
        base_state["code_revision_count"] = 0

        # Generate code
        with patch("src.agents.code.call_agent_with_metrics", return_value=gen_response):
            gen_result = code_generator_node(base_state)

        base_state["code"] = gen_result["code"]

        # Review code (should reject)
        with patch("src.agents.code.call_agent_with_metrics", return_value=review_response):
            review_result = code_reviewer_node(base_state)

        assert review_result.get("last_code_review_verdict") == "needs_revision"
        assert review_result.get("code_revision_count") == 1
        assert "reviewer_feedback" in review_result
        assert len(review_result["reviewer_feedback"]) > 10

    def test_repeated_rejections_trigger_escalation(self, base_state, valid_plan):
        """Test that repeated rejections eventually escalate to user."""
        from src.agents.code import code_reviewer_node

        review_response = {
            "verdict": "needs_revision",
            "issues": [{"severity": "critical", "description": "Still broken"}],
            "summary": "Cannot fix automatically",
            "feedback": "This requires manual intervention",
        }

        base_state["plan"] = valid_plan
        base_state["current_stage_id"] = "stage_0"
        base_state["code"] = "import meep as mp\nprint('broken')"
        max_revs = 3
        base_state["runtime_config"] = {"max_code_revisions": max_revs}

        # Simulate multiple rejections
        # With max=3, escalation happens when counter REACHES max:
        # i=0: 0→1, no escalate (1 < 3)
        # i=1: 1→2, no escalate (2 < 3)
        # i=2: 2→3, ESCALATE (3 >= 3, budget exhausted)
        for i in range(max_revs):
            base_state["code_revision_count"] = i

            with patch("src.agents.code.call_agent_with_metrics", return_value=review_response):
                result = code_reviewer_node(base_state)

            # Counter increments by 1 each rejection
            expected_count = i + 1
            assert result["code_revision_count"] == expected_count

            if expected_count < max_revs:
                # Should increment counter and continue
                assert result.get("ask_user_trigger") is None, (
                    f"Should not escalate at count {expected_count} (max={max_revs})"
                )
            else:
                # Should escalate when reaching max
                assert result.get("ask_user_trigger") is not None, (
                    f"Should escalate at count {expected_count} (max={max_revs})"
                )
                assert result.get("ask_user_trigger") == "code_review_limit"
