"""Contract shape tests for reviewer and supervisor agents.

These tests validate that responses conform to their JSON schemas and enforce
logical constraints that cannot be expressed in JSON Schema alone.
"""

import pytest
from jsonschema import ValidationError, validate

from .helpers import load_schema


class TestReviewerContract:
    """Test reviewer agent contract logic for all reviewer types."""

    REVIEWER_SCHEMAS = {
        "plan_reviewer": "plan_reviewer_output_schema.json",
        "design_reviewer": "design_reviewer_output_schema.json",
        "code_reviewer": "code_reviewer_output_schema.json",
    }

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_approve_verdict_full_schema_validation(self, reviewer_name, schema_file):
        """Approve verdict must fully conform to schema with all required fields."""
        schema = load_schema(schema_file)
        
        # Plan reviewer has different required fields than design/code reviewers
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "summary": "Plan looks good",
            }
        else:  # design_reviewer or code_reviewer
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],
                "summary": "Design/Code looks good",
            }
        
        # Must validate against full schema
        validate(instance=response, schema=schema)
        
        # Strict assertions
        assert response["verdict"] == "approve"
        assert isinstance(response["summary"], str)
        assert len(response["summary"]) > 0, "Summary must be non-empty"
        
        # Approve verdict should not have critical issues
        if "issues" in response:
            critical_issues = [i for i in response["issues"] if i.get("severity") == "blocking"]
            assert not critical_issues, f"{reviewer_name}: Approve verdict cannot have blocking issues"

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_needs_revision_verdict_full_schema_validation(self, reviewer_name, schema_file):
        """needs_revision verdict must fully conform to schema with actionable issues."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "needs_revision",
                "checklist_results": {
                    "coverage": {"status": "fail"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "issues": [
                    {
                        "severity": "major",
                        "category": "coverage",
                        "description": "Missing figure coverage",
                        "suggested_fix": "Add missing figures to stages",
                    }
                ],
                "summary": "Plan needs revision",
            }
        else:  # design_reviewer or code_reviewer
            response = {
                "stage_id": "stage_1",
                "verdict": "needs_revision",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [
                    {
                        "severity": "major",
                        "category": "geometry" if reviewer_name == "design_reviewer" else "unit_normalization",
                        "description": "Issue found",
                        "suggested_fix": "Fix the issue",
                    }
                ],
                "summary": "Needs revision",
            }
        
        # Must validate against full schema
        validate(instance=response, schema=schema)
        
        # Strict assertions
        assert response["verdict"] == "needs_revision"
        assert isinstance(response["issues"], list)
        assert len(response["issues"]) > 0, f"{reviewer_name}: needs_revision must have at least one issue"
        
        # All issues must have required fields
        for issue in response["issues"]:
            assert "severity" in issue, f"{reviewer_name}: Issue missing severity"
            assert "category" in issue, f"{reviewer_name}: Issue missing category"
            assert "description" in issue, f"{reviewer_name}: Issue missing description"
            assert "suggested_fix" in issue, f"{reviewer_name}: Issue missing suggested_fix"
            assert issue["severity"] in ["blocking", "major", "minor"], f"{reviewer_name}: Invalid severity value"
            assert len(issue["description"]) > 0, f"{reviewer_name}: Issue description must be non-empty"

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_all_required_fields_present(self, reviewer_name, schema_file):
        """All required fields must be present in response."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "summary": "Test",
            }
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],
                "summary": "Test",
            }
        
        # Must validate - this will fail if required fields are missing
        validate(instance=response, schema=schema)
        
        # Explicitly check required fields exist
        assert "verdict" in response
        assert "summary" in response
        assert "checklist_results" in response
        
        if reviewer_name != "plan_reviewer":
            assert "stage_id" in response, f"{reviewer_name}: stage_id is required"
            assert "issues" in response, f"{reviewer_name}: issues is required"

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_verdict_enum_values(self, reviewer_name, schema_file):
        """Verdict must be one of the allowed enum values."""
        schema = load_schema(schema_file)
        
        valid_verdicts = ["approve", "needs_revision"]
        
        for verdict in valid_verdicts:
            if reviewer_name == "plan_reviewer":
                response = {
                    "verdict": verdict,
                    "checklist_results": {
                        "coverage": {"status": "pass"},
                        "digitized_data": {"status": "pass"},
                        "staging": {"status": "pass"},
                        "parameter_extraction": {"status": "pass"},
                        "assumptions": {"status": "pass"},
                        "performance": {"status": "pass"},
                    },
                    "summary": "Test",
                }
            else:
                response = {
                    "stage_id": "stage_1",
                    "verdict": verdict,
                    "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                    "issues": [],
                    "summary": "Test",
                }
            
            # Must validate
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_invalid_verdict_rejected(self, reviewer_name, schema_file):
        """Invalid verdict values must be rejected."""
        schema = load_schema(schema_file)
        
        invalid_verdicts = ["approved", "reject", "maybe", "", None, 123]
        
        for invalid_verdict in invalid_verdicts:
            if reviewer_name == "plan_reviewer":
                response = {
                    "verdict": invalid_verdict,
                    "checklist_results": {
                        "coverage": {"status": "pass"},
                        "digitized_data": {"status": "pass"},
                        "staging": {"status": "pass"},
                        "parameter_extraction": {"status": "pass"},
                        "assumptions": {"status": "pass"},
                        "performance": {"status": "pass"},
                    },
                    "summary": "Test",
                }
            else:
                response = {
                    "stage_id": "stage_1",
                    "verdict": invalid_verdict,
                    "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                    "issues": [],
                    "summary": "Test",
                }
            
            # Must fail validation
            with pytest.raises(ValidationError) as exc_info:
                validate(instance=response, schema=schema)
            assert "verdict" in str(exc_info.value).lower() or "enum" in str(exc_info.value).lower()

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_checklist_results_structure(self, reviewer_name, schema_file):
        """Checklist results must have correct structure for each reviewer type."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "summary": "Test",
            }
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],
                "summary": "Test",
            }
        
        validate(instance=response, schema=schema)
        
        checklist = response["checklist_results"]
        assert isinstance(checklist, dict)
        
        # Check that all required checklist items have status
        for key, value in checklist.items():
            assert isinstance(value, dict), f"{reviewer_name}: Checklist item {key} must be a dict"
            assert "status" in value, f"{reviewer_name}: Checklist item {key} must have status"
            assert value["status"] in ["pass", "fail", "warning"], f"{reviewer_name}: Invalid status value"

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_issues_array_structure(self, reviewer_name, schema_file):
        """Issues array must have correct structure (if present)."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            # Plan reviewer issues are optional
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "issues": [
                    {
                        "severity": "minor",
                        "category": "coverage",
                        "description": "Minor issue",
                        "suggested_fix": "Fix it",
                    }
                ],
                "summary": "Test",
            }
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [
                    {
                        "severity": "minor",
                        "category": "geometry" if reviewer_name == "design_reviewer" else "unit_normalization",
                        "description": "Minor issue",
                        "suggested_fix": "Fix it",
                    }
                ],
                "summary": "Test",
            }
        
        validate(instance=response, schema=schema)
        
        if "issues" in response:
            assert isinstance(response["issues"], list)
            for issue in response["issues"]:
                assert isinstance(issue, dict)
                assert "severity" in issue
                assert "category" in issue
                assert "description" in issue
                assert "suggested_fix" in issue

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_empty_issues_array_allowed(self, reviewer_name, schema_file):
        """Empty issues array is allowed for approve verdict."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "issues": [],
                "summary": "Test",
            }
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],
                "summary": "Test",
            }
        
        validate(instance=response, schema=schema)
        assert response["issues"] == []

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_needs_revision_requires_issues(self, reviewer_name, schema_file):
        """needs_revision verdict should have issues (cross-field logic)."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            # Plan reviewer can have empty issues if checklist_results show failures
            response = {
                "verdict": "needs_revision",
                "checklist_results": {
                    "coverage": {"status": "fail"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "issues": [],
                "summary": "Test",
            }
            # This is technically valid per schema, but let's check the logic
            validate(instance=response, schema=schema)
            # If needs_revision, there should be some indication why (issues or failed checklist)
            has_failed_checklist = any(
                item.get("status") == "fail" for item in response["checklist_results"].values()
            )
            assert has_failed_checklist or response.get("issues"), \
                f"{reviewer_name}: needs_revision should have issues or failed checklist items"
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "needs_revision",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],  # Empty issues with needs_revision
                "summary": "Test",
            }
            validate(instance=response, schema=schema)
            # For design/code reviewers, needs_revision should have issues
            assert len(response["issues"]) > 0, \
                f"{reviewer_name}: needs_revision verdict must have at least one issue"

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_missing_required_field_fails(self, reviewer_name, schema_file):
        """Missing required fields must cause validation failure."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            # Missing verdict
            response = {
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"},
                },
                "summary": "Test",
            }
        else:
            # Missing verdict
            response = {
                "stage_id": "stage_1",
                "checklist_results": self._get_checklist_results_for_reviewer(reviewer_name),
                "issues": [],
                "summary": "Test",
            }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value).lower()

    @pytest.mark.parametrize("reviewer_name,schema_file", list(REVIEWER_SCHEMAS.items()))
    def test_wrong_type_fails(self, reviewer_name, schema_file):
        """Wrong field types must cause validation failure."""
        schema = load_schema(schema_file)
        
        if reviewer_name == "plan_reviewer":
            response = {
                "verdict": "approve",
                "checklist_results": "not a dict",  # Wrong type
                "summary": "Test",
            }
        else:
            response = {
                "stage_id": "stage_1",
                "verdict": "approve",
                "checklist_results": "not a dict",  # Wrong type
                "issues": [],
                "summary": "Test",
            }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "checklist_results" in str(exc_info.value).lower() or "object" in str(exc_info.value).lower()

    def _get_checklist_results_for_reviewer(self, reviewer_name: str) -> dict:
        """Get checklist results structure for design or code reviewer."""
        if reviewer_name == "design_reviewer":
            return {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"},
            }
        elif reviewer_name == "code_reviewer":
            return {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {"status": "pass", "all_outputs_accounted": True},
            }
        else:
            raise ValueError(f"Unknown reviewer: {reviewer_name}")


class TestSupervisorContract:
    """Test supervisor agent contract logic."""

    def test_ok_continue_verdict_full_schema_validation(self):
        """ok_continue verdict must fully conform to schema with all required fields."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Stage completed successfully",
        }
        
        # Must validate against full schema
        validate(instance=response, schema=schema)
        
        # Strict assertions
        assert response["verdict"] == "ok_continue"
        assert response.get("should_stop") is not True, "ok_continue should not stop workflow"
        assert isinstance(response["summary"], str)
        assert len(response["summary"]) > 0

    def test_all_complete_verdict_full_schema_validation(self):
        """all_complete verdict must fully conform to schema."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "all_complete",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "passed",
                "parameter_sweeps": "passed",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "should_stop": True,
            "stop_reason": "All stages completed",
            "summary": "Reproduction complete",
        }
        
        validate(instance=response, schema=schema)
        
        assert response["verdict"] == "all_complete"
        assert response["should_stop"] is True, "all_complete must set should_stop to True"
        assert "stop_reason" in response, "all_complete should have stop_reason"
        assert isinstance(response["stop_reason"], str)
        assert len(response["stop_reason"]) > 0

    def test_all_verdict_enum_values(self):
        """All valid verdict enum values must be accepted."""
        schema = load_schema("supervisor_output_schema.json")
        
        valid_verdicts = [
            "ok_continue",
            "replan_needed",
            "change_priority",
            "ask_user",
            "backtrack_to_stage",
            "all_complete",
        ]
        
        for verdict in valid_verdicts:
            response = {
                "verdict": verdict,
                "validation_hierarchy_status": {
                    "material_validation": "passed",
                    "single_structure": "passed",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "summary": f"Test for {verdict}",
            }
            
            # Add verdict-specific required fields
            if verdict == "backtrack_to_stage":
                response["backtrack_decision"] = {
                    "accepted": True,
                    "target_stage_id": "stage_0",
                    "stages_to_invalidate": ["stage_1"],
                    "reason": "Test backtrack",
                }
            elif verdict == "ask_user":
                response["user_question"] = "Test question"
            
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_invalid_verdict_rejected(self):
        """Invalid verdict values must be rejected."""
        schema = load_schema("supervisor_output_schema.json")
        
        invalid_verdicts = ["continue", "complete", "stop", "", None, 123]
        
        for invalid_verdict in invalid_verdicts:
            response = {
                "verdict": invalid_verdict,
                "validation_hierarchy_status": {
                    "material_validation": "passed",
                    "single_structure": "passed",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "summary": "Test",
            }
            
            with pytest.raises(ValidationError) as exc_info:
                validate(instance=response, schema=schema)
            assert "verdict" in str(exc_info.value).lower() or "enum" in str(exc_info.value).lower()

    def test_validation_hierarchy_status_structure(self):
        """validation_hierarchy_status must have all required fields with valid enum values."""
        schema = load_schema("supervisor_output_schema.json")
        
        valid_statuses = ["passed", "partial", "failed", "not_done"]
        
        for status in valid_statuses:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": status,
                    "single_structure": status,
                    "arrays_systems": status,
                    "parameter_sweeps": status,
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "summary": "Test",
            }
            
            validate(instance=response, schema=schema)
            assert response["validation_hierarchy_status"]["material_validation"] == status

    def test_validation_hierarchy_status_invalid_enum(self):
        """Invalid validation hierarchy status values must be rejected."""
        schema = load_schema("supervisor_output_schema.json")
        
        invalid_statuses = ["complete", "done", "pending", "", None, 123]
        
        for invalid_status in invalid_statuses:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": invalid_status,
                    "single_structure": "passed",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "summary": "Test",
            }
            
            with pytest.raises(ValidationError) as exc_info:
                validate(instance=response, schema=schema)
            assert "material_validation" in str(exc_info.value).lower() or "enum" in str(exc_info.value).lower()

    def test_main_physics_assessment_structure(self):
        """main_physics_assessment must have all required boolean fields."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": False,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        
        validate(instance=response, schema=schema)
        
        assessment = response["main_physics_assessment"]
        assert isinstance(assessment["physics_plausible"], bool)
        assert isinstance(assessment["conservation_satisfied"], bool)
        assert isinstance(assessment["value_ranges_reasonable"], bool)

    def test_main_physics_assessment_missing_required_field(self):
        """Missing required fields in main_physics_assessment must fail."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                # Missing value_ranges_reasonable
            },
            "summary": "Test",
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "value_ranges_reasonable" in str(exc_info.value).lower()

    def test_backtrack_decision_structure(self):
        """backtrack_decision must have all required fields when verdict is backtrack_to_stage."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": False,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage_0",
                "stages_to_invalidate": ["stage_1", "stage_2"],
                "reason": "Physics assessment failed",
            },
            "summary": "Need to backtrack",
        }
        
        validate(instance=response, schema=schema)
        
        decision = response["backtrack_decision"]
        assert isinstance(decision["accepted"], bool)
        assert isinstance(decision["target_stage_id"], str)
        assert len(decision["target_stage_id"]) > 0
        assert isinstance(decision["stages_to_invalidate"], list)
        assert isinstance(decision["reason"], str)
        assert len(decision["reason"]) > 0

    def test_backtrack_decision_missing_required_field(self):
        """Missing required fields in backtrack_decision must fail."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "failed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": False,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage_0",
                # Missing stages_to_invalidate
                "reason": "Test",
            },
            "summary": "Test",
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "stages_to_invalidate" in str(exc_info.value).lower()

    def test_ask_user_requires_user_question(self):
        """ask_user verdict should have user_question field (cross-field logic)."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ask_user",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "user_question": "Should we continue?",
            "summary": "Need user input",
        }
        
        validate(instance=response, schema=schema)
        
        # Cross-field logic: ask_user should have user_question
        assert "user_question" in response, "ask_user verdict should have user_question"
        assert isinstance(response["user_question"], str)
        assert len(response["user_question"]) > 0

    def test_all_complete_requires_should_stop_true(self):
        """all_complete verdict must have should_stop=True (cross-field logic)."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "all_complete",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "passed",
                "parameter_sweeps": "passed",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "should_stop": True,
            "stop_reason": "All done",
            "summary": "Complete",
        }
        
        validate(instance=response, schema=schema)
        
        # Cross-field logic: all_complete must stop
        assert response.get("should_stop") is True, "all_complete must set should_stop to True"

    def test_missing_required_fields_fails(self):
        """Missing any required field must cause validation failure."""
        schema = load_schema("supervisor_output_schema.json")
        
        # Missing verdict
        response = {
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value).lower()

    def test_wrong_type_fails(self):
        """Wrong field types must cause validation failure."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": "not a dict",  # Wrong type
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "validation_hierarchy_status" in str(exc_info.value).lower() or "object" in str(exc_info.value).lower()

    def test_empty_summary_rejected(self):
        """Summary must be non-empty string - schema enforces minLength."""
        schema = load_schema("supervisor_output_schema.json")
        
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "passed",
                "single_structure": "passed",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "",  # Empty string - should be rejected
        }
        
        # Schema should reject empty string due to minLength: 1
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "summary" in str(exc_info.value).lower() or "minLength" in str(exc_info.value).lower() or "non-empty" in str(exc_info.value).lower()
