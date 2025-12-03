"""Edge case handling tests for LLM responses."""

from jsonschema import validate

from .helpers import load_schema


class TestEdgeCaseResponses:
    """Test handling of edge cases in LLM responses."""

    def test_empty_stages_array(self):
        """Handle planner returning empty stages."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "No reproducible content",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }

        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)

    def test_planner_mixed_parameter_types(self):
        """extracted_parameters values can be number, string, or array."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "T",
            "summary": "S",
            "extracted_parameters": [
                {"name": "p1", "value": 1.5, "unit": "nm", "source": "text"},
                {"name": "p2", "value": "approx 5", "unit": "nm", "source": "text"},
                {"name": "p3", "value": [1.0, 2.0], "unit": "nm", "source": "text"},
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)

    def test_supervisor_partial_validation_status(self):
        """Validation hierarchy can be partially complete."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Continue despite failure",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)

