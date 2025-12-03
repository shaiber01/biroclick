"""Integration tests ensuring routing verdicts match schema definitions."""

import json
from pathlib import Path


class TestRoutingReturnsValidValues:
    """Test that routing functions return values that exist in the graph."""

    SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "schemas"

    def test_supervisor_verdicts_match_routing(self):
        """Supervisor schema verdicts must match what routing expects."""
        schema_file = self.SCHEMAS_DIR / "supervisor_output_schema.json"
        with open(schema_file, encoding="utf-8") as file:
            schema = json.load(file)

        schema_verdicts = set(schema["properties"]["verdict"]["enum"])
        handled_verdicts = {
            "ok_continue",
            "change_priority",
            "replan_needed",
            "ask_user",
            "backtrack_to_stage",
            "all_complete",
        }

        unhandled = schema_verdicts - handled_verdicts
        assert not unhandled, (
            "Supervisor schema allows verdicts that routing doesn't handle: "
            f"{unhandled}\nThis could cause routing errors at runtime!"
        )

    def test_reviewer_verdicts_match_routing(self):
        """Reviewer verdicts must match what routing expects."""
        expected_verdicts = {"approve", "needs_revision"}
        for reviewer in ["plan_reviewer", "design_reviewer", "code_reviewer"]:
            schema_file = self.SCHEMAS_DIR / f"{reviewer}_output_schema.json"
            if schema_file.exists():
                with open(schema_file, encoding="utf-8") as file:
                    schema = json.load(file)
                verdict_property = schema.get("properties", {}).get("verdict", {})
                schema_verdicts = set(verdict_property.get("enum", []))
                missing = expected_verdicts - schema_verdicts
                assert not missing, f"{reviewer} schema missing verdicts: {missing}"


