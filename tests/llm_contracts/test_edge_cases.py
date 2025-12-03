"""Edge case handling tests for LLM responses.

This module tests edge cases for LLM response schemas:
1. Valid edge case values that SHOULD pass (empty arrays, nulls, boundaries)
2. Invalid edge case values that SHOULD fail (violations of constraints)

The goal is to ensure schemas are strict enough to catch bugs while
flexible enough to handle legitimate edge cases.
"""

import pytest
from jsonschema import validate, ValidationError

from .helpers import load_schema


class TestEdgeCaseResponses:
    """Test handling of edge cases in LLM responses."""

    # ========== PLANNER SCHEMA EDGE CASES ==========

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
        
        # Assertions: verify structure is valid and arrays are empty
        assert response["stages"] == []
        assert response["targets"] == []
        assert response["extracted_parameters"] == []
        assert isinstance(response["assumptions"], dict)
        assert response["progress"]["stages"] == []

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
        
        # Assertions: verify each parameter type is correctly handled
        assert isinstance(response["extracted_parameters"][0]["value"], (int, float))
        assert isinstance(response["extracted_parameters"][1]["value"], str)
        assert isinstance(response["extracted_parameters"][2]["value"], list)
        assert all(isinstance(x, (int, float)) for x in response["extracted_parameters"][2]["value"])

    def test_planner_all_paper_domain_enums(self):
        """Test all valid paper_domain enum values."""
        valid_domains = ["plasmonics", "photonic_crystal", "metamaterial", "thin_film", 
                        "waveguide", "strong_coupling", "nonlinear", "other"]
        
        for domain in valid_domains:
            response = {
                "paper_id": "test",
                "paper_domain": domain,
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["paper_domain"] == domain

    def test_planner_empty_strings(self):
        """Test that empty strings are allowed for string fields."""
        response = {
            "paper_id": "",
            "paper_domain": "other",
            "title": "",
            "summary": "",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["paper_id"] == ""
        assert response["title"] == ""
        assert response["summary"] == ""

    def test_planner_all_parameter_source_enums(self):
        """Test all valid source enum values for extracted_parameters."""
        valid_sources = ["text", "figure_caption", "figure_axis", "supplementary", 
                        "inferred", "user_correction"]
        
        for source in valid_sources:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [
                    {"name": "p1", "value": 1.0, "unit": "nm", "source": source}
                ],
                "targets": [],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["extracted_parameters"][0]["source"] == source

    def test_planner_parameter_with_null_discrepancy_notes(self):
        """Test that discrepancy_notes can be null."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {"name": "p1", "value": 1.0, "unit": "nm", "source": "text", 
                 "discrepancy_notes": None}
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["extracted_parameters"][0]["discrepancy_notes"] is None

    def test_planner_parameter_with_string_discrepancy_notes(self):
        """Test that discrepancy_notes can be a string."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {"name": "p1", "value": 1.0, "unit": "nm", "source": "text", 
                 "discrepancy_notes": "Some discrepancy"}
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert isinstance(response["extracted_parameters"][0]["discrepancy_notes"], str)

    def test_planner_parameter_with_optional_fields(self):
        """Test extracted_parameters with all optional fields."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {
                    "name": "p1",
                    "value": 1.0,
                    "unit": "nm",
                    "source": "text",
                    "location": "Section 2.1",
                    "cross_checked": True,
                    "discrepancy_notes": "Checked against figure"
                }
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        param = response["extracted_parameters"][0]
        assert param["location"] == "Section 2.1"
        assert param["cross_checked"] is True
        assert param["discrepancy_notes"] == "Checked against figure"

    def test_planner_all_target_type_enums(self):
        """Test all valid target type enum values."""
        valid_types = ["spectrum", "dispersion", "field_map", "parameter_sweep", "other"]
        
        for target_type in valid_types:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [
                    {
                        "figure_id": "fig1",
                        "description": "Test",
                        "type": target_type,
                        "simulation_class": "FDTD_DIRECT"
                    }
                ],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["targets"][0]["type"] == target_type

    def test_planner_all_simulation_class_enums(self):
        """Test all valid simulation_class enum values."""
        valid_classes = ["FDTD_DIRECT", "FDTD_DERIVED", "ANALYTICAL", 
                        "COMPLEX_PHYSICS", "NOT_REPRODUCIBLE"]
        
        for sim_class in valid_classes:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [
                    {
                        "figure_id": "fig1",
                        "description": "Test",
                        "type": "spectrum",
                        "simulation_class": sim_class
                    }
                ],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["targets"][0]["simulation_class"] == sim_class

    def test_planner_all_precision_requirement_enums(self):
        """Test all valid precision_requirement enum values."""
        valid_precisions = ["excellent", "good", "acceptable", "qualitative"]
        
        for precision in valid_precisions:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [
                    {
                        "figure_id": "fig1",
                        "description": "Test",
                        "type": "spectrum",
                        "simulation_class": "FDTD_DIRECT",
                        "precision_requirement": precision
                    }
                ],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["targets"][0]["precision_requirement"] == precision

    def test_planner_target_with_null_complexity_notes(self):
        """Test that complexity_notes can be null."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    "figure_id": "fig1",
                    "description": "Test",
                    "type": "spectrum",
                    "simulation_class": "FDTD_DIRECT",
                    "complexity_notes": None
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["targets"][0]["complexity_notes"] is None

    def test_planner_target_with_null_digitized_data_path(self):
        """Test that digitized_data_path can be null."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    "figure_id": "fig1",
                    "description": "Test",
                    "type": "spectrum",
                    "simulation_class": "FDTD_DIRECT",
                    "digitized_data_path": None
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["targets"][0]["digitized_data_path"] is None

    def test_planner_all_stage_type_enums(self):
        """Test all valid stage_type enum values."""
        valid_stage_types = ["MATERIAL_VALIDATION", "SINGLE_STRUCTURE", "ARRAY_SYSTEM", 
                            "PARAMETER_SWEEP", "COMPLEX_PHYSICS"]
        
        for stage_type in valid_stage_types:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [
                    {
                        "stage_id": "stage1",
                        "stage_type": stage_type,
                        "name": "Test Stage",
                        "description": "Test",
                        "targets": [],
                        "dependencies": []
                    }
                ],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["stages"][0]["stage_type"] == stage_type

    def test_planner_all_complexity_class_enums(self):
        """Test all valid complexity_class enum values."""
        valid_complexities = ["analytical", "2D_light", "2D_medium", "3D_light", 
                            "3D_medium", "3D_heavy"]
        
        for complexity in valid_complexities:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [
                    {
                        "stage_id": "stage1",
                        "stage_type": "SINGLE_STRUCTURE",
                        "name": "Test",
                        "description": "Test",
                        "targets": [],
                        "dependencies": [],
                        "complexity_class": complexity
                    }
                ],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["stages"][0]["complexity_class"] == complexity

    def test_planner_all_fallback_strategy_enums(self):
        """Test all valid fallback_strategy enum values."""
        valid_strategies = ["ask_user", "simplify", "skip_with_warning"]
        
        for strategy in valid_strategies:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [
                    {
                        "stage_id": "stage1",
                        "stage_type": "SINGLE_STRUCTURE",
                        "name": "Test",
                        "description": "Test",
                        "targets": [],
                        "dependencies": [],
                        "fallback_strategy": strategy
                    }
                ],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["stages"][0]["fallback_strategy"] == strategy

    def test_planner_all_artifact_type_enums(self):
        """Test all valid artifact_type enum values."""
        valid_artifacts = ["spectrum_csv", "field_data_npz", "field_plot_png", 
                          "spectrum_plot_png", "dispersion_csv", "raw_h5"]
        
        for artifact_type in valid_artifacts:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [
                    {
                        "stage_id": "stage1",
                        "stage_type": "SINGLE_STRUCTURE",
                        "name": "Test",
                        "description": "Test",
                        "targets": [],
                        "dependencies": [],
                        "expected_outputs": [
                            {
                                "artifact_type": artifact_type,
                                "filename_pattern": "test.*",
                                "description": "Test"
                            }
                        ]
                    }
                ],
                "assumptions": {},
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["stages"][0]["expected_outputs"][0]["artifact_type"] == artifact_type

    def test_planner_stage_with_null_reference_data_path(self):
        """Test that reference_data_path can be null."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Test",
                    "description": "Test",
                    "targets": [],
                    "dependencies": [],
                    "reference_data_path": None
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["stages"][0]["reference_data_path"] is None

    def test_planner_stage_with_numeric_boundaries(self):
        """Test numeric fields with boundary values."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    "stage_type": "SINGLE_STRUCTURE",
                    "name": "Test",
                    "description": "Test",
                    "targets": [],
                    "dependencies": [],
                    "runtime_estimate_minutes": 0.001,
                    "runtime_budget_minutes": 1000.0,
                    "max_revisions": 0
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        stage = response["stages"][0]
        assert stage["runtime_estimate_minutes"] == 0.001
        assert stage["runtime_budget_minutes"] == 1000.0
        assert stage["max_revisions"] == 0

    def test_planner_all_assumption_category_enums(self):
        """Test all valid assumption category enum values."""
        valid_categories = ["material", "geometry", "source", "boundary", "numerical"]
        
        for category in valid_categories:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {
                    "global_assumptions": [
                        {
                            "id": "a1",
                            "category": category,
                            "description": "Test",
                            "reason": "Test",
                            "source": "paper_stated"
                        }
                    ]
                },
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["assumptions"]["global_assumptions"][0]["category"] == category

    def test_planner_all_assumption_source_enums(self):
        """Test all valid assumption source enum values."""
        valid_sources = ["paper_stated", "paper_inferred", "literature_default", "user_provided"]
        
        for source in valid_sources:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {
                    "global_assumptions": [
                        {
                            "id": "a1",
                            "category": "material",
                            "description": "Test",
                            "reason": "Test",
                            "source": source
                        }
                    ]
                },
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["assumptions"]["global_assumptions"][0]["source"] == source

    def test_planner_assumption_with_null_validation_stage(self):
        """Test that validation_stage can be null."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {
                "global_assumptions": [
                    {
                        "id": "a1",
                        "category": "material",
                        "description": "Test",
                        "reason": "Test",
                        "source": "paper_stated",
                        "validation_stage": None
                    }
                ]
            },
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["assumptions"]["global_assumptions"][0]["validation_stage"] is None

    def test_planner_all_progress_status_enums(self):
        """Test all valid progress status enum values."""
        valid_statuses = ["not_started", "in_progress", "completed_success", 
                         "completed_partial", "completed_failed", "blocked", 
                         "needs_rerun", "invalidated"]
        
        for status in valid_statuses:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {},
                "progress": {
                    "stages": [
                        {
                            "stage_id": "stage1",
                            "status": status
                        }
                    ]
                },
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["progress"]["stages"][0]["status"] == status

    def test_planner_geometry_interpretation_confidence_enums(self):
        """Test all valid confidence enum values for geometry_interpretations."""
        valid_confidences = ["high", "medium", "low"]
        
        for confidence in valid_confidences:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {
                    "geometry_interpretations": [
                        {
                            "term": "nanoantenna",
                            "interpretation": "Test",
                            "confidence": confidence
                        }
                    ]
                },
                "progress": {"stages": []},
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["assumptions"]["geometry_interpretations"][0]["confidence"] == confidence

    def test_planner_reproduction_scope_edge_cases(self):
        """Test reproduction_scope with edge case values."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
            "reproduction_scope": {
                "total_figures": 0,
                "reproducible_figures": 0,
                "attempted_figures": [],
                "skipped_figures": [],
                "coverage_percent": 0.0
            }
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        scope = response["reproduction_scope"]
        assert scope["total_figures"] == 0
        assert scope["reproducible_figures"] == 0
        assert scope["attempted_figures"] == []
        assert scope["skipped_figures"] == []
        assert scope["coverage_percent"] == 0.0

    def test_planner_reproduction_scope_all_skipped_classifications(self):
        """Test all valid skipped_figures classification enum values."""
        valid_classifications = ["not_reproducible", "reproducible_other", 
                                "out_of_scope", "low_priority"]
        
        for classification in valid_classifications:
            response = {
                "paper_id": "test",
                "paper_domain": "other",
                "title": "Test",
                "summary": "Test",
                "extracted_parameters": [],
                "targets": [],
                "stages": [],
                "assumptions": {},
                "progress": {"stages": []},
                "reproduction_scope": {
                    "skipped_figures": [
                        {
                            "figure_id": "fig1",
                            "reason": "Test",
                            "classification": classification
                        }
                    ]
                }
            }
            schema = load_schema("planner_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["reproduction_scope"]["skipped_figures"][0]["classification"] == classification

    def test_planner_blocking_issues_edge_cases(self):
        """Test blocking_issues with edge cases."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
            "blocking_issues": [
                {
                    "description": "",
                    "question_for_user": ""
                }
            ]
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert len(response["blocking_issues"]) == 1
        assert response["blocking_issues"][0]["description"] == ""
        assert response["blocking_issues"][0]["question_for_user"] == ""

    def test_planner_planned_materials_with_null_path(self):
        """Test planned_materials with null path."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
            "planned_materials": [
                {
                    "material_id": "mat1",
                    "name": "Gold",
                    "suggested_source": "palik",
                    "path": None
                }
            ]
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["planned_materials"][0]["path"] is None

    def test_planner_complex_nested_structure(self):
        """Test complex nested structure with all optional fields populated."""
        response = {
            "paper_id": "complex_test",
            "paper_domain": "metamaterial",
            "title": "Complex Test Paper",
            "summary": "A complex test case",
            "main_system": "Metamaterial structure",
            "main_claims": ["Claim 1", "Claim 2"],
            "simulation_approach": "FDTD with Meep",
            "extracted_parameters": [
                {
                    "name": "wavelength",
                    "value": 500.0,
                    "unit": "nm",
                    "source": "text",
                    "location": "Section 2",
                    "cross_checked": True,
                    "discrepancy_notes": None
                },
                {
                    "name": "period",
                    "value": [100.0, 200.0],
                    "unit": "nm",
                    "source": "figure_axis",
                    "location": "Figure 3",
                    "cross_checked": False,
                    "discrepancy_notes": "Range from figure"
                }
            ],
            "targets": [
                {
                    "figure_id": "fig1",
                    "description": "Transmission spectrum",
                    "type": "spectrum",
                    "simulation_class": "FDTD_DIRECT",
                    "precision_requirement": "excellent",
                    "complexity_notes": "High resolution needed",
                    "digitized_data_path": "/path/to/data.csv"
                }
            ],
            "stages": [
                {
                    "stage_id": "stage0",
                    "stage_type": "MATERIAL_VALIDATION",
                    "name": "Material Validation",
                    "description": "Validate material properties",
                    "targets": [],
                    "dependencies": [],
                    "is_mandatory_validation": True,
                    "complexity_class": "2D_light",
                    "runtime_estimate_minutes": 5.0,
                    "runtime_budget_minutes": 10.0,
                    "max_revisions": 3,
                    "fallback_strategy": "ask_user",
                    "validation_criteria": ["Material data exists"],
                    "expected_outputs": [
                        {
                            "artifact_type": "spectrum_csv",
                            "filename_pattern": "material_*.csv",
                            "description": "Material spectrum",
                            "columns": ["wavelength", "n", "k"],
                            "target_figure": "fig1"
                        }
                    ],
                    "reference_data_path": "/path/to/reference.csv"
                }
            ],
            "assumptions": {
                "global_assumptions": [
                    {
                        "id": "assum1",
                        "category": "material",
                        "description": "Gold permittivity",
                        "reason": "Standard model",
                        "source": "literature_default",
                        "alternatives_considered": ["Drude model"],
                        "critical": True,
                        "validated": False,
                        "validation_stage": None
                    }
                ],
                "geometry_interpretations": [
                    {
                        "term": "unit cell",
                        "interpretation": "Square lattice",
                        "alternatives": ["Hexagonal"],
                        "confidence": "high"
                    }
                ]
            },
            "progress": {
                "stages": [
                    {
                        "stage_id": "stage0",
                        "status": "not_started",
                        "summary": "Not started yet"
                    }
                ]
            },
            "staging_rationale": "Material validation first, then structure",
            "blocking_issues": [],
            "planned_materials": [
                {
                    "material_id": "gold",
                    "name": "Gold",
                    "suggested_source": "palik",
                    "path": "/materials/palik_gold.csv"
                }
            ]
        }
        schema = load_schema("planner_output_schema.json")
        validate(instance=response, schema=schema)
        
        # Verify complex structure
        assert len(response["extracted_parameters"]) == 2
        assert len(response["targets"]) == 1
        assert len(response["stages"]) == 1
        assert len(response["assumptions"]["global_assumptions"]) == 1
        assert len(response["assumptions"]["geometry_interpretations"]) == 1
        assert response["stages"][0]["expected_outputs"][0]["columns"] == ["wavelength", "n", "k"]

    # ========== SUPERVISOR SCHEMA EDGE CASES ==========

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
        
        # Assertions: verify partial status is valid
        status = response["validation_hierarchy_status"]
        assert status["material_validation"] == "passed"
        assert status["single_structure"] == "failed"
        assert status["arrays_systems"] == "not_done"
        assert status["parameter_sweeps"] == "not_done"

    def test_supervisor_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["ok_continue", "replan_needed", "change_priority", 
                         "ask_user", "backtrack_to_stage", "all_complete"]
        
        for verdict in valid_verdicts:
            response = {
                "verdict": verdict,
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
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
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_supervisor_all_validation_status_enums(self):
        """Test all valid validation status enum values."""
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
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["validation_hierarchy_status"]["material_validation"] == status

    def test_supervisor_physics_assessment_all_false(self):
        """Test physics assessment with all boolean fields False."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": False,
                "conservation_satisfied": False,
                "value_ranges_reasonable": False,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assessment = response["main_physics_assessment"]
        assert assessment["physics_plausible"] is False
        assert assessment["conservation_satisfied"] is False
        assert assessment["value_ranges_reasonable"] is False

    def test_supervisor_physics_assessment_with_optional_fields(self):
        """Test physics assessment with optional fields."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
                "systematic_issues": ["Issue 1", "Issue 2"],
                "notes": "Some notes"
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assessment = response["main_physics_assessment"]
        assert assessment["systematic_issues"] == ["Issue 1", "Issue 2"]
        assert assessment["notes"] == "Some notes"

    def test_supervisor_all_error_type_enums(self):
        """Test all valid error_type enum values."""
        valid_types = ["systematic", "random", "none"]
        
        for error_type in valid_types:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "error_analysis": {
                    "error_type": error_type
                },
                "summary": "Test",
            }
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["error_analysis"]["error_type"] == error_type

    def test_supervisor_all_error_persistence_enums(self):
        """Test all valid error_persistence enum values."""
        valid_persistences = ["improving", "stable", "worsening", "not_applicable"]
        
        for persistence in valid_persistences:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "error_analysis": {
                    "error_persistence": persistence
                },
                "summary": "Test",
            }
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["error_analysis"]["error_persistence"] == persistence

    def test_supervisor_all_confidence_enums(self):
        """Test all valid confidence enum values."""
        valid_confidences = ["high", "medium", "low"]
        
        for confidence in valid_confidences:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "error_analysis": {
                    "confidence": confidence
                },
                "summary": "Test",
            }
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["error_analysis"]["confidence"] == confidence

    def test_supervisor_all_priority_enums(self):
        """Test all valid priority enum values."""
        valid_priorities = ["critical", "high", "medium", "low"]
        
        for priority in valid_priorities:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "recommendations": [
                    {
                        "action": "Test action",
                        "priority": priority
                    }
                ],
                "summary": "Test",
            }
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["recommendations"][0]["priority"] == priority

    def test_supervisor_backtrack_decision_complete(self):
        """Test backtrack_decision with all required fields."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage0",
                "stages_to_invalidate": ["stage1", "stage2"],
                "reason": "Material validation failed"
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        decision = response["backtrack_decision"]
        assert decision["accepted"] is True
        assert decision["target_stage_id"] == "stage0"
        assert decision["stages_to_invalidate"] == ["stage1", "stage2"]
        assert decision["reason"] == "Material validation failed"

    def test_supervisor_backtrack_decision_empty_invalidate_list(self):
        """Test backtrack_decision with empty stages_to_invalidate."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "backtrack_decision": {
                "accepted": False,
                "target_stage_id": "stage0",
                "stages_to_invalidate": [],
                "reason": "No stages to invalidate"
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["backtrack_decision"]["stages_to_invalidate"] == []

    def test_supervisor_user_question_with_verdict(self):
        """Test user_question field when verdict is ask_user."""
        response = {
            "verdict": "ask_user",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "user_question": "What should we do?",
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["user_question"] == "What should we do?"

    def test_supervisor_all_overall_confidence_enums(self):
        """Test all valid overall_confidence enum values."""
        valid_confidences = ["high", "medium", "low"]
        
        for confidence in valid_confidences:
            response = {
                "verdict": "ok_continue",
                "validation_hierarchy_status": {
                    "material_validation": "not_done",
                    "single_structure": "not_done",
                    "arrays_systems": "not_done",
                    "parameter_sweeps": "not_done",
                },
                "main_physics_assessment": {
                    "physics_plausible": True,
                    "conservation_satisfied": True,
                    "value_ranges_reasonable": True,
                },
                "progress_summary": {
                    "stages_completed": 0,
                    "stages_remaining": 1,
                    "overall_confidence": confidence
                },
                "summary": "Test",
            }
            schema = load_schema("supervisor_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["progress_summary"]["overall_confidence"] == confidence

    def test_supervisor_progress_summary_boundary_values(self):
        """Test progress_summary with boundary integer values."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "progress_summary": {
                "stages_completed": 0,
                "stages_remaining": 0,
                "overall_confidence": "high",
                "key_achievements": [],
                "key_blockers": []
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        summary = response["progress_summary"]
        assert summary["stages_completed"] == 0
        assert summary["stages_remaining"] == 0
        assert summary["key_achievements"] == []
        assert summary["key_blockers"] == []

    def test_supervisor_should_stop_true_with_reason(self):
        """Test should_stop True with stop_reason."""
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
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["should_stop"] is True
        assert response["stop_reason"] == "All stages completed"

    def test_supervisor_should_stop_false(self):
        """Test should_stop False."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "should_stop": False,
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["should_stop"] is False

    def test_supervisor_minimal_summary(self):
        """Test that summary must be at least 1 character (minLength: 1)."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "X",  # Minimum length: 1 character
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        assert len(response["summary"]) == 1
        assert response["summary"] == "X"

    def test_supervisor_complex_full_structure(self):
        """Test complex supervisor response with all optional fields."""
        response = {
            "verdict": "backtrack_to_stage",
            "validation_hierarchy_status": {
                "material_validation": "failed",
                "single_structure": "partial",
                "arrays_systems": "passed",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": False,
                "conservation_satisfied": True,
                "value_ranges_reasonable": False,
                "systematic_issues": ["Issue 1", "Issue 2"],
                "notes": "Physics assessment notes"
            },
            "error_analysis": {
                "error_type": "systematic",
                "error_persistence": "worsening",
                "root_cause_hypothesis": "Material properties incorrect",
                "confidence": "high"
            },
            "recommendations": [
                {
                    "action": "Revalidate materials",
                    "priority": "critical",
                    "rationale": "Material validation failed"
                },
                {
                    "action": "Check physics model",
                    "priority": "high"
                }
            ],
            "backtrack_decision": {
                "accepted": True,
                "target_stage_id": "stage0",
                "stages_to_invalidate": ["stage1", "stage2"],
                "reason": "Material validation must be redone"
            },
            "user_question": "Should we proceed with backtrack?",
            "progress_summary": {
                "stages_completed": 2,
                "stages_remaining": 3,
                "overall_confidence": "medium",
                "key_achievements": ["Achievement 1"],
                "key_blockers": ["Blocker 1"]
            },
            "should_stop": False,
            "stop_reason": "",
            "summary": "Complex supervisor assessment"
        }
        schema = load_schema("supervisor_output_schema.json")
        validate(instance=response, schema=schema)
        
        # Verify complex structure
        assert len(response["recommendations"]) == 2
        assert len(response["backtrack_decision"]["stages_to_invalidate"]) == 2
        assert len(response["progress_summary"]["key_achievements"]) == 1
        assert len(response["progress_summary"]["key_blockers"]) == 1

    # ========== CODE GENERATOR SCHEMA EDGE CASES ==========

    def test_code_generator_all_artifact_type_enums(self):
        """Test all valid artifact_type enum values for code generator."""
        valid_artifacts = ["spectrum_csv", "field_data_npz", "field_plot_png", 
                          "spectrum_plot_png", "dispersion_csv", "raw_h5", "result_json"]
        
        for artifact_type in valid_artifacts:
            response = {
                "stage_id": "stage1",
                "code": "import meep as mp\n# code here",
                "expected_outputs": [
                    {
                        "artifact_type": artifact_type,
                        "filename": "output.txt",
                        "description": "Test output"
                    }
                ],
                "estimated_runtime_minutes": 1.0,
                "summary": "Test code generation summary"
            }
            schema = load_schema("code_generator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["expected_outputs"][0]["artifact_type"] == artifact_type

    def test_code_generator_expected_outputs_with_null_target_figure(self):
        """Test that target_figure can be null."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [
                {
                    "artifact_type": "spectrum_csv",
                    "filename": "output.csv",
                    "description": "Test",
                    "target_figure": None
                }
            ],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary"
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["expected_outputs"][0]["target_figure"] is None

    def test_code_generator_materials_used_with_null_path(self):
        """Test materials_used with null data_file_path."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "materials_used": [
                {
                    "material_name": "Gold",
                    "source": "palik",
                    "data_file_path": None
                }
            ]
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["materials_used"][0]["data_file_path"] is None

    def test_code_generator_revision_notes_null(self):
        """Test that revision_notes can be null."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "revision_notes": None
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["revision_notes"] is None

    def test_code_generator_revision_notes_string(self):
        """Test that revision_notes can be a string."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "revision_notes": "Fixed unit system"
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert isinstance(response["revision_notes"], str)

    def test_code_generator_safety_checks_all_false(self):
        """Test safety_checks with all boolean fields False."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "safety_checks": {
                "no_plt_show": False,
                "no_input": False,
                "uses_plt_savefig_close": False,
                "relative_paths_only": False,
                "includes_result_json": False
            }
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        checks = response["safety_checks"]
        assert checks["no_plt_show"] is False
        assert checks["no_input"] is False
        assert checks["uses_plt_savefig_close"] is False
        assert checks["relative_paths_only"] is False
        assert checks["includes_result_json"] is False

    def test_code_generator_numeric_boundaries(self):
        """Test numeric fields with boundary values."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 0.001,
            "estimated_memory_gb": 0.1,
            "summary": "Test code generation summary"
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["estimated_runtime_minutes"] == 0.001
        assert response["estimated_memory_gb"] == 0.1

    def test_code_generator_empty_arrays(self):
        """Test empty arrays for optional list fields."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "materials_used": [],
            "dependencies_used": [],
            "progress_markers": []
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["materials_used"] == []
        assert response["dependencies_used"] == []
        assert response["progress_markers"] == []

    def test_code_generator_design_compliance_all_true(self):
        """Test design_compliance with all boolean fields True."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "design_compliance": {
                "unit_system_matches_design": True,
                "geometry_matches_design": True,
                "materials_match_design": True,
                "output_filenames_match_spec": True
            }
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        compliance = response["design_compliance"]
        assert compliance["unit_system_matches_design"] is True
        assert compliance["geometry_matches_design"] is True
        assert compliance["materials_match_design"] is True
        assert compliance["output_filenames_match_spec"] is True

    def test_code_generator_unit_system_used(self):
        """Test unit_system_used with valid values."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary",
            "unit_system_used": {
                "characteristic_length_m": 1e-6,
                "verified_from_design": True
            }
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["unit_system_used"]["characteristic_length_m"] == 1e-6
        assert response["unit_system_used"]["verified_from_design"] is True

    def test_code_generator_expected_outputs_with_columns(self):
        """Test expected_outputs with columns field for CSV files."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [
                {
                    "artifact_type": "spectrum_csv",
                    "filename": "spectrum.csv",
                    "description": "Output spectrum",
                    "columns": ["wavelength_nm", "transmission", "reflection"],
                    "target_figure": "fig2a"
                }
            ],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test code generation summary"
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        output = response["expected_outputs"][0]
        assert output["columns"] == ["wavelength_nm", "transmission", "reflection"]
        assert output["target_figure"] == "fig2a"

    # ========== DESIGN REVIEWER SCHEMA EDGE CASES ==========

    def test_design_reviewer_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["approve", "needs_revision"]
        
        for verdict in valid_verdicts:
            response = {
                "stage_id": "stage1",
                "verdict": verdict,
                "checklist_results": {
                    "geometry": {"status": "pass"},
                    "physics": {"status": "pass"},
                    "materials": {"status": "pass"},
                    "unit_system": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "resolution": {"status": "pass"},
                    "outputs": {"status": "pass"},
                    "runtime": {"status": "pass"}
                },
                "issues": [],
                "summary": "Test"
            }
            schema = load_schema("design_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_design_reviewer_all_checklist_status_enums(self):
        """Test all valid checklist status enum values."""
        valid_statuses = ["pass", "fail", "warning"]
        
        for status in valid_statuses:
            response = {
                "stage_id": "stage1",
                "verdict": "approve",
                "checklist_results": {
                    "geometry": {"status": status},
                    "physics": {"status": status},
                    "materials": {"status": status},
                    "unit_system": {"status": status},
                    "source": {"status": status},
                    "domain": {"status": status},
                    "resolution": {"status": status},
                    "outputs": {"status": status},
                    "runtime": {"status": status}
                },
                "issues": [],
                "summary": "Test"
            }
            schema = load_schema("design_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["checklist_results"]["geometry"]["status"] == status

    def test_design_reviewer_all_issue_severity_enums(self):
        """Test all valid issue severity enum values."""
        valid_severities = ["blocking", "major", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "checklist_results": {
                    "geometry": {"status": "pass"},
                    "physics": {"status": "pass"},
                    "materials": {"status": "pass"},
                    "unit_system": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "resolution": {"status": "pass"},
                    "outputs": {"status": "pass"},
                    "runtime": {"status": "pass"}
                },
                "issues": [
                    {
                        "severity": severity,
                        "category": "geometry",
                        "description": "Test",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test"
            }
            schema = load_schema("design_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["severity"] == severity

    def test_design_reviewer_all_issue_category_enums(self):
        """Test all valid issue category enum values."""
        valid_categories = ["geometry", "physics", "materials", "unit_system", 
                           "source", "domain", "resolution", "outputs", "runtime"]
        
        for category in valid_categories:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "checklist_results": {
                    "geometry": {"status": "pass"},
                    "physics": {"status": "pass"},
                    "materials": {"status": "pass"},
                    "unit_system": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "resolution": {"status": "pass"},
                    "outputs": {"status": "pass"},
                    "runtime": {"status": "pass"}
                },
                "issues": [
                    {
                        "severity": "minor",
                        "category": category,
                        "description": "Test",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test"
            }
            schema = load_schema("design_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["category"] == category

    def test_design_reviewer_escalate_to_user_boolean(self):
        """Test escalate_to_user as boolean."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"}
            },
            "issues": [],
            "summary": "Test",
            "escalate_to_user": False
        }
        schema = load_schema("design_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["escalate_to_user"] is False

    def test_design_reviewer_escalate_to_user_string(self):
        """Test escalate_to_user as string."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"}
            },
            "issues": [],
            "summary": "Test",
            "escalate_to_user": "What material should we use?"
        }
        schema = load_schema("design_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert isinstance(response["escalate_to_user"], str)
        assert response["escalate_to_user"] == "What material should we use?"

    def test_design_reviewer_backtrack_suggestion_all_severity_enums(self):
        """Test all valid backtrack suggestion severity enum values."""
        valid_severities = ["critical", "significant", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "approve",
                "checklist_results": {
                    "geometry": {"status": "pass"},
                    "physics": {"status": "pass"},
                    "materials": {"status": "pass"},
                    "unit_system": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "resolution": {"status": "pass"},
                    "outputs": {"status": "pass"},
                    "runtime": {"status": "pass"}
                },
                "issues": [],
                "summary": "Test",
                "backtrack_suggestion": {
                    "suggest_backtrack": True,
                    "target_stage_id": "stage0",
                    "reason": "Test",
                    "severity": severity,
                    "evidence": "Test evidence"
                }
            }
            schema = load_schema("design_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["backtrack_suggestion"]["severity"] == severity

    def test_design_reviewer_empty_arrays(self):
        """Test empty arrays for optional list fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "geometry": {"status": "pass"},
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"}
            },
            "issues": [],
            "strengths": [],
            "summary": "Test"
        }
        schema = load_schema("design_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["issues"] == []
        assert response["strengths"] == []

    # ========== PLAN REVIEWER SCHEMA EDGE CASES ==========

    def test_plan_reviewer_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["approve", "needs_revision"]
        
        for verdict in valid_verdicts:
            response = {
                "verdict": verdict,
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"}
                },
                "summary": "Test"
            }
            schema = load_schema("plan_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_plan_reviewer_all_checklist_status_enums(self):
        """Test all valid checklist status enum values."""
        valid_statuses = ["pass", "fail", "warning"]
        
        for status in valid_statuses:
            response = {
                "verdict": "approve",
                "checklist_results": {
                    "coverage": {"status": status},
                    "digitized_data": {"status": status},
                    "staging": {"status": status},
                    "parameter_extraction": {"status": status},
                    "assumptions": {"status": status},
                    "performance": {"status": status}
                },
                "summary": "Test"
            }
            schema = load_schema("plan_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["checklist_results"]["coverage"]["status"] == status

    def test_plan_reviewer_all_issue_severity_enums(self):
        """Test all valid issue severity enum values."""
        valid_severities = ["blocking", "major", "minor"]
        
        for severity in valid_severities:
            response = {
                "verdict": "needs_revision",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"}
                },
                "issues": [
                    {
                        "severity": severity,
                        "category": "coverage",
                        "description": "Test",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test"
            }
            schema = load_schema("plan_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["severity"] == severity

    def test_plan_reviewer_all_issue_category_enums(self):
        """Test all valid issue category enum values."""
        valid_categories = ["coverage", "staging", "parameters", "assumptions", 
                           "performance", "digitized_data", "material_validation", 
                           "output_specifications"]
        
        for category in valid_categories:
            response = {
                "verdict": "needs_revision",
                "checklist_results": {
                    "coverage": {"status": "pass"},
                    "digitized_data": {"status": "pass"},
                    "staging": {"status": "pass"},
                    "parameter_extraction": {"status": "pass"},
                    "assumptions": {"status": "pass"},
                    "performance": {"status": "pass"}
                },
                "issues": [
                    {
                        "severity": "minor",
                        "category": category,
                        "description": "Test",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test"
            }
            schema = load_schema("plan_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["category"] == category

    def test_plan_reviewer_escalate_to_user_boolean(self):
        """Test escalate_to_user as boolean."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"}
            },
            "summary": "Test",
            "escalate_to_user": False
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["escalate_to_user"] is False

    def test_plan_reviewer_escalate_to_user_string(self):
        """Test escalate_to_user as string."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass"},
                "digitized_data": {"status": "pass"},
                "staging": {"status": "pass"},
                "parameter_extraction": {"status": "pass"},
                "assumptions": {"status": "pass"},
                "performance": {"status": "pass"}
            },
            "summary": "Test",
            "escalate_to_user": "Should we proceed?"
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert isinstance(response["escalate_to_user"], str)
        assert response["escalate_to_user"] == "Should we proceed?"

    def test_plan_reviewer_empty_arrays(self):
        """Test empty arrays for optional list fields."""
        response = {
            "verdict": "approve",
            "checklist_results": {
                "coverage": {"status": "pass", "figures_covered": [], "figures_missing": []},
                "digitized_data": {"status": "pass", "excellent_targets": [], 
                                  "have_digitized": [], "missing_digitized": []},
                "staging": {"status": "pass", "dependency_issues": []},
                "parameter_extraction": {"status": "pass", "missing_critical": []},
                "assumptions": {"status": "pass", "risky_assumptions": [], 
                               "undocumented_gaps": []},
                "performance": {"status": "pass", "risky_stages": []}
            },
            "issues": [],
            "strengths": [],
            "summary": "Test"
        }
        schema = load_schema("plan_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["issues"] == []
        assert response["strengths"] == []

    # ========== SIMULATION DESIGNER SCHEMA EDGE CASES ==========

    def test_simulation_designer_all_dimensionality_enums(self):
        """Test all valid dimensionality enum values."""
        valid_dimensionalities = ["2D", "3D"]
        
        for dim in valid_dimensionalities:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {
                    "dimensionality": dim,
                    "structures": []
                },
                "materials": [],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["geometry"]["dimensionality"] == dim

    def test_simulation_designer_all_structure_type_enums(self):
        """Test all valid structure type enum values."""
        valid_types = ["cylinder", "sphere", "block", "ellipsoid", "cone", "prism", "custom"]
        
        for struct_type in valid_types:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {
                    "structures": [
                        {
                            "name": "test_structure",
                            "type": struct_type,
                            "material_ref": "gold"
                        }
                    ]
                },
                "materials": [],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["geometry"]["structures"][0]["type"] == struct_type

    def test_simulation_designer_all_material_model_type_enums(self):
        """Test all valid material model_type enum values."""
        valid_types = ["constant", "tabulated", "drude", "lorentz", "drude_lorentz"]
        
        for model_type in valid_types:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [
                    {
                        "id": "mat1",
                        "name": "Test Material",
                        "model_type": model_type
                    }
                ],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["materials"][0]["model_type"] == model_type

    def test_simulation_designer_all_source_type_enums(self):
        """Test all valid source type enum values."""
        valid_types = ["gaussian", "continuous", "eigenmode"]
        
        for source_type in valid_types:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [],
                "sources": [
                    {
                        "type": source_type,
                        "center": {"x": 0, "y": 0, "z": 0},
                        "size": {"x": 1, "y": 1, "z": 0}
                    }
                ],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["sources"][0]["type"] == source_type

    def test_simulation_designer_all_source_component_enums(self):
        """Test all valid source component enum values."""
        valid_components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        
        for component in valid_components:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [],
                "sources": [
                    {
                        "type": "gaussian",
                        "component": component,
                        "center": {"x": 0, "y": 0, "z": 0},
                        "size": {"x": 1, "y": 1, "z": 0}
                    }
                ],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["sources"][0]["component"] == component

    def test_simulation_designer_all_boundary_condition_enums(self):
        """Test all valid boundary condition enum values."""
        valid_bcs = ["pml", "periodic", "mirror", "metallic"]
        
        for bc in valid_bcs:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [],
                "sources": [],
                "boundary_conditions": {
                    "x_min": bc,
                    "x_max": bc,
                    "y_min": bc,
                    "y_max": bc,
                    "z_min": bc,
                    "z_max": bc
                },
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["boundary_conditions"]["x_min"] == bc

    def test_simulation_designer_all_monitor_type_enums(self):
        """Test all valid monitor type enum values."""
        valid_types = ["flux", "field", "dft_fields", "near2far"]
        
        for monitor_type in valid_types:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [
                    {
                        "type": monitor_type,
                        "name": "test_monitor"
                    }
                ],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["monitors"][0]["type"] == monitor_type

    def test_simulation_designer_all_symmetry_direction_enums(self):
        """Test all valid symmetry direction enum values."""
        valid_directions = ["x", "y", "z"]
        
        for direction in valid_directions:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {
                    "structures": [],
                    "symmetries": [
                        {
                            "direction": direction,
                            "phase": 1.0
                        }
                    ]
                },
                "materials": [],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [],
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["geometry"]["symmetries"][0]["direction"] == direction

    def test_simulation_designer_all_run_until_type_enums(self):
        """Test all valid run_until type enum values."""
        valid_types = ["time", "decay"]
        
        for run_type in valid_types:
            response = {
                "stage_id": "stage1",
                "design_description": "Test simulation design",
                "unit_system": {
                    "characteristic_length_m": 1e-6,
                    "length_unit": "µm"
                },
                "geometry": {"structures": []},
                "materials": [],
                "sources": [],
                "boundary_conditions": {},
                "monitors": [],
                "simulation_parameters": {
                    "run_until": {
                        "type": run_type,
                        "value": 100.0
                    }
                },
                "performance_estimate": {
                    "runtime_estimate_minutes": 10.0,
                    "memory_estimate_gb": 2.0
                },
                "summary": "Test design summary"
            }
            schema = load_schema("simulation_designer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["simulation_parameters"]["run_until"]["type"] == run_type

    def test_simulation_designer_material_with_null_data_file(self):
        """Test that material data_file can be null."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test simulation design",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm"
            },
            "geometry": {"structures": []},
            "materials": [
                {
                    "id": "mat1",
                    "name": "Gold",
                    "model_type": "drude",
                    "data_file": None
                }
            ],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10.0,
                "memory_estimate_gb": 2.0
            },
            "summary": "Test design summary"
        }
        schema = load_schema("simulation_designer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["materials"][0]["data_file"] is None

    def test_simulation_designer_empty_arrays(self):
        """Test empty arrays for optional list fields."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test simulation design",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm"
            },
            "geometry": {"structures": []},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10.0,
                "memory_estimate_gb": 2.0
            },
            "summary": "Test design summary",
            "output_specifications": [],
            "new_assumptions": [],
            "potential_issues": []
        }
        schema = load_schema("simulation_designer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["output_specifications"] == []
        assert response["new_assumptions"] == []
        assert response["potential_issues"] == []

    def test_simulation_designer_complex_structure(self):
        """Test complex simulation designer with all optional fields."""
        response = {
            "stage_id": "stage1_single_disk",
            "design_description": "FDTD simulation of gold nanodisk",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm",
                "example_conversions": {
                    "100nm_in_meep": 0.1,
                    "500nm_in_meep": 0.5
                }
            },
            "geometry": {
                "dimensionality": "3D",
                "cell_size": {"x": 2.0, "y": 2.0, "z": 1.5},
                "resolution": 50,
                "structures": [
                    {
                        "name": "gold_disk",
                        "type": "cylinder",
                        "material_ref": "gold",
                        "center": {"x": 0, "y": 0, "z": 0},
                        "dimensions": {"radius": 0.05, "height": 0.02},
                        "real_dimensions": {"radius_nm": 50, "height_nm": 20}
                    }
                ],
                "symmetries": [
                    {"direction": "x", "phase": 1.0},
                    {"direction": "y", "phase": 1.0}
                ]
            },
            "materials": [
                {
                    "id": "gold",
                    "name": "Gold",
                    "model_type": "drude_lorentz",
                    "source": "johnson_christy",
                    "data_file": "/materials/gold_jc.csv",
                    "parameters": {
                        "epsilon_inf": 1.0,
                        "drude_terms": [{"omega_p": 9.0, "gamma": 0.07}],
                        "lorentz_terms": [{"sigma": 0.5, "omega_0": 2.5, "gamma": 0.3}]
                    },
                    "wavelength_range": {"min_nm": 400, "max_nm": 800}
                }
            ],
            "sources": [
                {
                    "type": "gaussian",
                    "component": "Ez",
                    "center": {"x": 0, "y": 0, "z": -0.5},
                    "size": {"x": 2.0, "y": 2.0, "z": 0},
                    "wavelength_center_nm": 600,
                    "wavelength_width_nm": 200,
                    "frequency_center_meep": 1.67,
                    "frequency_width_meep": 0.56
                }
            ],
            "boundary_conditions": {
                "x_min": "pml",
                "x_max": "pml",
                "y_min": "pml",
                "y_max": "pml",
                "z_min": "pml",
                "z_max": "pml",
                "pml_thickness": 0.5,
                "pml_layers": 32
            },
            "monitors": [
                {
                    "type": "flux",
                    "name": "transmission",
                    "purpose": "Measure transmitted light",
                    "center": {"x": 0, "y": 0, "z": 0.3},
                    "size": {"x": 2.0, "y": 2.0, "z": 0},
                    "frequency_points": 100
                }
            ],
            "simulation_parameters": {
                "run_until": {"type": "decay", "value": 1e-6, "decay_by": 0.001},
                "subpixel_averaging": True,
                "force_complex_fields": False
            },
            "performance_estimate": {
                "runtime_estimate_minutes": 45.0,
                "memory_estimate_gb": 8.0,
                "total_cells": 1000000,
                "timesteps_estimate": 50000,
                "notes": "Large memory due to 3D"
            },
            "output_specifications": [
                {
                    "artifact_type": "spectrum_csv",
                    "filename_pattern": "transmission_*.csv",
                    "description": "Transmission spectrum",
                    "columns": ["wavelength_nm", "transmission"]
                }
            ],
            "new_assumptions": [
                {
                    "id": "assum1",
                    "category": "material",
                    "description": "Using J&C gold data",
                    "reason": "Common choice for nanoplasmonics",
                    "critical": True
                }
            ],
            "design_rationale": "Standard FDTD approach for plasmonic resonator",
            "potential_issues": ["Memory intensive for 3D simulation"],
            "summary": "Complete 3D FDTD simulation design for gold nanodisk"
        }
        schema = load_schema("simulation_designer_output_schema.json")
        validate(instance=response, schema=schema)
        assert len(response["geometry"]["structures"]) == 1
        assert len(response["materials"]) == 1
        assert len(response["sources"]) == 1
        assert len(response["monitors"]) == 1

    # ========== CODE REVIEWER SCHEMA EDGE CASES ==========

    def test_code_reviewer_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["approve", "needs_revision"]
        
        for verdict in valid_verdicts:
            response = {
                "stage_id": "stage1",
                "verdict": verdict,
                "checklist_results": {
                    "unit_normalization": {"status": "pass"},
                    "numerics": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "monitors": {"status": "pass"},
                    "visualization": {"status": "pass"},
                    "code_quality": {"status": "pass"},
                    "runtime": {"status": "pass"},
                    "meep_api": {"status": "pass"},
                    "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
                },
                "issues": [],
                "summary": "Test code review summary"
            }
            schema = load_schema("code_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_code_reviewer_all_checklist_status_enums(self):
        """Test all valid checklist status enum values."""
        valid_statuses = ["pass", "fail", "warning"]
        
        for status in valid_statuses:
            response = {
                "stage_id": "stage1",
                "verdict": "approve",
                "checklist_results": {
                    "unit_normalization": {"status": status},
                    "numerics": {"status": status},
                    "source": {"status": status},
                    "domain": {"status": status},
                    "monitors": {"status": status},
                    "visualization": {"status": status},
                    "code_quality": {"status": status},
                    "runtime": {"status": status},
                    "meep_api": {"status": status},
                    "expected_outputs": {"status": status, "all_outputs_accounted": True}
                },
                "issues": [],
                "summary": "Test code review summary"
            }
            schema = load_schema("code_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["checklist_results"]["unit_normalization"]["status"] == status

    def test_code_reviewer_all_issue_severity_enums(self):
        """Test all valid issue severity enum values."""
        valid_severities = ["blocking", "major", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "checklist_results": {
                    "unit_normalization": {"status": "pass"},
                    "numerics": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "monitors": {"status": "pass"},
                    "visualization": {"status": "pass"},
                    "code_quality": {"status": "fail"},
                    "runtime": {"status": "pass"},
                    "meep_api": {"status": "pass"},
                    "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
                },
                "issues": [
                    {
                        "severity": severity,
                        "category": "code_quality",
                        "description": "Test issue",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test code review summary"
            }
            schema = load_schema("code_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["severity"] == severity

    def test_code_reviewer_all_issue_category_enums(self):
        """Test all valid issue category enum values."""
        valid_categories = ["unit_normalization", "numerics", "source", "domain", 
                           "monitors", "visualization", "code_quality", "runtime", 
                           "meep_api", "expected_outputs"]
        
        for category in valid_categories:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "checklist_results": {
                    "unit_normalization": {"status": "pass"},
                    "numerics": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "monitors": {"status": "pass"},
                    "visualization": {"status": "pass"},
                    "code_quality": {"status": "pass"},
                    "runtime": {"status": "pass"},
                    "meep_api": {"status": "pass"},
                    "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
                },
                "issues": [
                    {
                        "severity": "minor",
                        "category": category,
                        "description": "Test issue",
                        "suggested_fix": "Fix it"
                    }
                ],
                "summary": "Test code review summary"
            }
            schema = load_schema("code_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["category"] == category

    def test_code_reviewer_escalate_to_user_boolean(self):
        """Test escalate_to_user as boolean."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
            },
            "issues": [],
            "summary": "Test code review summary",
            "escalate_to_user": False
        }
        schema = load_schema("code_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["escalate_to_user"] is False

    def test_code_reviewer_escalate_to_user_string(self):
        """Test escalate_to_user as string."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
            },
            "issues": [],
            "summary": "Test code review summary",
            "escalate_to_user": "Should we increase resolution?"
        }
        schema = load_schema("code_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["escalate_to_user"] == "Should we increase resolution?"

    def test_code_reviewer_backtrack_suggestion_all_severity_enums(self):
        """Test all valid backtrack suggestion severity enum values."""
        valid_severities = ["critical", "significant", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "checklist_results": {
                    "unit_normalization": {"status": "pass"},
                    "numerics": {"status": "pass"},
                    "source": {"status": "pass"},
                    "domain": {"status": "pass"},
                    "monitors": {"status": "pass"},
                    "visualization": {"status": "pass"},
                    "code_quality": {"status": "pass"},
                    "runtime": {"status": "pass"},
                    "meep_api": {"status": "pass"},
                    "expected_outputs": {"status": "pass", "all_outputs_accounted": True}
                },
                "issues": [],
                "summary": "Test code review summary",
                "backtrack_suggestion": {
                    "suggest_backtrack": True,
                    "target_stage_id": "stage0",
                    "reason": "Design issue found",
                    "severity": severity,
                    "evidence": "Test evidence"
                }
            }
            schema = load_schema("code_reviewer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["backtrack_suggestion"]["severity"] == severity

    def test_code_reviewer_unit_normalization_details(self):
        """Test unit_normalization with detailed fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {
                    "status": "pass",
                    "a_unit_value": 1e-6,
                    "design_a_unit": 1e-6,
                    "match": True,
                    "notes": "Units match design specification"
                },
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {
                    "status": "pass",
                    "has_plt_show": False,
                    "has_input": False
                },
                "runtime": {
                    "status": "pass",
                    "estimated_minutes": 30.0,
                    "budget_minutes": 60.0
                },
                "meep_api": {"status": "pass"},
                "expected_outputs": {
                    "status": "pass",
                    "all_outputs_accounted": True,
                    "outputs_checked": [
                        {
                            "artifact_type": "spectrum_csv",
                            "expected_filename": "spectrum.csv",
                            "code_produces_file": True,
                            "columns_match": True
                        }
                    ]
                }
            },
            "issues": [],
            "summary": "Test code review summary"
        }
        schema = load_schema("code_reviewer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["checklist_results"]["unit_normalization"]["match"] is True

    # ========== EXECUTION VALIDATOR SCHEMA EDGE CASES ==========

    def test_execution_validator_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["pass", "warning", "fail"]
        
        for verdict in valid_verdicts:
            response = {
                "stage_id": "stage1",
                "verdict": verdict,
                "execution_status": {"completed": True},
                "files_check": {
                    "expected_files": ["output.csv"],
                    "found_files": ["output.csv"],
                    "missing_files": [],
                    "all_present": True
                },
                "summary": "Test execution validation"
            }
            schema = load_schema("execution_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_execution_validator_all_error_severity_enums(self):
        """Test all valid error severity enum values."""
        valid_severities = ["critical", "warning", "info"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "fail",
                "execution_status": {"completed": False},
                "files_check": {
                    "expected_files": [],
                    "found_files": [],
                    "missing_files": [],
                    "all_present": True
                },
                "errors_detected": [
                    {
                        "error_type": "RuntimeError",
                        "message": "Test error",
                        "severity": severity
                    }
                ],
                "summary": "Test execution validation"
            }
            schema = load_schema("execution_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["errors_detected"][0]["severity"] == severity

    def test_execution_validator_execution_status_complete(self):
        """Test execution_status with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {
                "completed": True,
                "exit_code": 0,
                "runtime_seconds": 1234.5,
                "memory_peak_mb": 4096.0,
                "timed_out": False
            },
            "files_check": {
                "expected_files": ["output.csv"],
                "found_files": ["output.csv"],
                "missing_files": [],
                "all_present": True
            },
            "summary": "Test execution validation"
        }
        schema = load_schema("execution_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["execution_status"]["completed"] is True
        assert response["execution_status"]["exit_code"] == 0
        assert response["execution_status"]["timed_out"] is False

    def test_execution_validator_data_quality_checks(self):
        """Test data_quality with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "warning",
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": ["output.csv"],
                "found_files": ["output.csv"],
                "missing_files": [],
                "all_present": True
            },
            "data_quality": {
                "nan_detected": True,
                "inf_detected": False,
                "negative_where_unexpected": False,
                "suspicious_values": [
                    {
                        "file": "output.csv",
                        "issue": "NaN values in column 3"
                    }
                ]
            },
            "summary": "Test execution validation"
        }
        schema = load_schema("execution_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["data_quality"]["nan_detected"] is True
        assert len(response["data_quality"]["suspicious_values"]) == 1

    def test_execution_validator_spec_compliance_with_null_filename(self):
        """Test that spec_compliance actual_filename can be null."""
        response = {
            "stage_id": "stage1",
            "verdict": "fail",
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": ["output.csv"],
                "found_files": [],
                "missing_files": ["output.csv"],
                "all_present": False,
                "spec_compliance": [
                    {
                        "artifact_type": "spectrum_csv",
                        "expected_filename": "output.csv",
                        "actual_filename": None,
                        "exists": False,
                        "non_empty": False,
                        "valid_format": False,
                        "issues": ["File not found"]
                    }
                ]
            },
            "summary": "Test execution validation"
        }
        schema = load_schema("execution_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["files_check"]["spec_compliance"][0]["actual_filename"] is None

    def test_execution_validator_empty_arrays(self):
        """Test empty arrays for optional fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True
            },
            "errors_detected": [],
            "warnings": [],
            "summary": "Test execution validation"
        }
        schema = load_schema("execution_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["errors_detected"] == []
        assert response["warnings"] == []

    # ========== PHYSICS SANITY SCHEMA EDGE CASES ==========

    def test_physics_sanity_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["pass", "warning", "fail", "design_flaw"]
        
        for verdict in valid_verdicts:
            response = {
                "stage_id": "stage1",
                "verdict": verdict,
                "conservation_checks": [],
                "value_range_checks": [],
                "summary": "Test physics sanity check"
            }
            schema = load_schema("physics_sanity_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_physics_sanity_all_check_status_enums(self):
        """Test all valid check status enum values."""
        valid_statuses = ["pass", "warning", "fail"]
        
        for status in valid_statuses:
            response = {
                "stage_id": "stage1",
                "verdict": "pass",
                "conservation_checks": [
                    {
                        "law": "Energy conservation: T+R+A=1",
                        "status": status
                    }
                ],
                "value_range_checks": [
                    {
                        "quantity": "Transmission",
                        "status": status
                    }
                ],
                "summary": "Test physics sanity check"
            }
            schema = load_schema("physics_sanity_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["conservation_checks"][0]["status"] == status
            assert response["value_range_checks"][0]["status"] == status

    def test_physics_sanity_all_concern_severity_enums(self):
        """Test all valid concern severity enum values."""
        valid_severities = ["critical", "moderate", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "warning",
                "conservation_checks": [],
                "value_range_checks": [],
                "concerns": [
                    {
                        "concern": "Test concern",
                        "severity": severity,
                        "possible_cause": "Unknown",
                        "suggested_action": "Investigate"
                    }
                ],
                "summary": "Test physics sanity check"
            }
            schema = load_schema("physics_sanity_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["concerns"][0]["severity"] == severity

    def test_physics_sanity_all_backtrack_severity_enums(self):
        """Test all valid backtrack severity enum values."""
        valid_severities = ["critical", "significant", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "design_flaw",
                "conservation_checks": [],
                "value_range_checks": [],
                "backtrack_suggestion": {
                    "suggest_backtrack": True,
                    "target_stage_id": "stage0",
                    "reason": "Design problem detected",
                    "severity": severity,
                    "evidence": "Test evidence"
                },
                "summary": "Test physics sanity check"
            }
            schema = load_schema("physics_sanity_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["backtrack_suggestion"]["severity"] == severity

    def test_physics_sanity_conservation_check_complete(self):
        """Test conservation check with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [
                {
                    "law": "Energy conservation: T+R+A=1",
                    "status": "pass",
                    "expected_value": 1.0,
                    "actual_value": 0.998,
                    "deviation_percent": 0.2,
                    "threshold_percent": 1.0,
                    "notes": "Within acceptable tolerance"
                }
            ],
            "value_range_checks": [],
            "summary": "Test physics sanity check"
        }
        schema = load_schema("physics_sanity_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["conservation_checks"][0]["deviation_percent"] == 0.2

    def test_physics_sanity_value_range_check_complete(self):
        """Test value range check with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "value_range_checks": [
                {
                    "quantity": "Transmission coefficient",
                    "status": "pass",
                    "value": 0.85,
                    "expected_range": {
                        "min": 0.0,
                        "max": 1.0
                    },
                    "notes": "Within physical bounds"
                }
            ],
            "summary": "Test physics sanity check"
        }
        schema = load_schema("physics_sanity_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["value_range_checks"][0]["value"] == 0.85

    def test_physics_sanity_numerical_quality(self):
        """Test numerical_quality with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "value_range_checks": [],
            "numerical_quality": {
                "field_decay_achieved": True,
                "convergence_observed": True,
                "artifacts_detected": [],
                "notes": "Good numerical quality"
            },
            "summary": "Test physics sanity check"
        }
        schema = load_schema("physics_sanity_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["numerical_quality"]["field_decay_achieved"] is True

    def test_physics_sanity_physical_plausibility(self):
        """Test physical_plausibility with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "conservation_checks": [],
            "value_range_checks": [],
            "physical_plausibility": {
                "resonance_positions_reasonable": True,
                "linewidths_reasonable": True,
                "magnitude_scale_reasonable": True,
                "spectral_features_expected": True,
                "concerns": []
            },
            "summary": "Test physics sanity check"
        }
        schema = load_schema("physics_sanity_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["physical_plausibility"]["resonance_positions_reasonable"] is True

    # ========== RESULTS ANALYZER SCHEMA EDGE CASES ==========

    def test_results_analyzer_all_overall_classification_enums(self):
        """Test all valid overall_classification enum values."""
        valid_classifications = ["EXCELLENT_MATCH", "ACCEPTABLE_MATCH", 
                                 "PARTIAL_MATCH", "POOR_MATCH", "FAILED"]
        
        for classification in valid_classifications:
            response = {
                "stage_id": "stage1",
                "per_result_reports": [],
                "figure_comparisons": [],
                "overall_classification": classification,
                "summary": "Test results analysis"
            }
            schema = load_schema("results_analyzer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["overall_classification"] == classification

    def test_results_analyzer_all_discrepancy_classification_enums(self):
        """Test all valid discrepancy classification enum values."""
        valid_classifications = ["excellent", "acceptable", "investigate", "unacceptable"]
        
        for classification in valid_classifications:
            response = {
                "stage_id": "stage1",
                "per_result_reports": [
                    {
                        "result_id": "result1",
                        "target_figure": "fig2a",
                        "quantity": "Transmission peak position",
                        "discrepancy": {
                            "classification": classification
                        }
                    }
                ],
                "figure_comparisons": [],
                "overall_classification": "ACCEPTABLE_MATCH",
                "summary": "Test results analysis"
            }
            schema = load_schema("results_analyzer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["per_result_reports"][0]["discrepancy"]["classification"] == classification

    def test_results_analyzer_all_comparison_type_enums(self):
        """Test all valid comparison_type enum values."""
        valid_types = ["overlay", "side_by_side", "difference_map", "quantitative_only"]
        
        for comp_type in valid_types:
            response = {
                "stage_id": "stage1",
                "per_result_reports": [],
                "figure_comparisons": [
                    {
                        "paper_figure_id": "fig2",
                        "simulated_figure_path": "/output/fig2_sim.png",
                        "comparison_type": comp_type
                    }
                ],
                "overall_classification": "ACCEPTABLE_MATCH",
                "summary": "Test results analysis"
            }
            schema = load_schema("results_analyzer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["figure_comparisons"][0]["comparison_type"] == comp_type

    def test_results_analyzer_all_visual_agreement_enums(self):
        """Test all valid visual_agreement enum values."""
        valid_agreements = ["excellent", "good", "fair", "poor"]
        
        for agreement in valid_agreements:
            response = {
                "stage_id": "stage1",
                "per_result_reports": [],
                "figure_comparisons": [
                    {
                        "paper_figure_id": "fig2",
                        "simulated_figure_path": "/output/fig2_sim.png",
                        "comparison_type": "overlay",
                        "visual_agreement": agreement
                    }
                ],
                "overall_classification": "ACCEPTABLE_MATCH",
                "summary": "Test results analysis"
            }
            schema = load_schema("results_analyzer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["figure_comparisons"][0]["visual_agreement"] == agreement

    def test_results_analyzer_confidence_boundaries(self):
        """Test confidence field with boundary values (0 to 1)."""
        boundary_values = [0.0, 0.5, 1.0]
        
        for confidence in boundary_values:
            response = {
                "stage_id": "stage1",
                "per_result_reports": [],
                "figure_comparisons": [],
                "overall_classification": "ACCEPTABLE_MATCH",
                "confidence": confidence,
                "summary": "Test results analysis"
            }
            schema = load_schema("results_analyzer_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["confidence"] == confidence

    def test_results_analyzer_per_result_report_complete(self):
        """Test per_result_reports with all fields."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [
                {
                    "result_id": "result1",
                    "target_figure": "fig2a",
                    "quantity": "Resonance wavelength",
                    "simulated_value": {
                        "value": 550.0,
                        "unit": "nm"
                    },
                    "paper_value": {
                        "value": 545.0,
                        "unit": "nm",
                        "source": "Figure 2a caption"
                    },
                    "discrepancy": {
                        "absolute": 5.0,
                        "relative_percent": 0.92,
                        "classification": "excellent"
                    },
                    "notes": "Good agreement within digitization error"
                }
            ],
            "figure_comparisons": [],
            "overall_classification": "EXCELLENT_MATCH",
            "summary": "Test results analysis"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        validate(instance=response, schema=schema)
        report = response["per_result_reports"][0]
        assert report["simulated_value"]["value"] == 550.0
        assert report["paper_value"]["source"] == "Figure 2a caption"
        assert report["discrepancy"]["relative_percent"] == 0.92

    def test_results_analyzer_figure_comparison_complete(self):
        """Test figure_comparisons with all fields."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [
                {
                    "paper_figure_id": "fig2",
                    "simulated_figure_path": "/output/fig2_sim.png",
                    "comparison_type": "overlay",
                    "visual_agreement": "good",
                    "key_features_matched": ["Peak position", "Peak width"],
                    "key_features_missed": ["Secondary peak"],
                    "notes": "Main features captured well"
                }
            ],
            "overall_classification": "ACCEPTABLE_MATCH",
            "summary": "Test results analysis"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        validate(instance=response, schema=schema)
        comparison = response["figure_comparisons"][0]
        assert len(comparison["key_features_matched"]) == 2
        assert len(comparison["key_features_missed"]) == 1

    def test_results_analyzer_systematic_discrepancies(self):
        """Test systematic_discrepancies field."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "PARTIAL_MATCH",
            "systematic_discrepancies": [
                {
                    "pattern": "Consistent 5nm red shift",
                    "affected_results": ["result1", "result2"],
                    "possible_cause": "Substrate index mismatch"
                }
            ],
            "summary": "Test results analysis"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        validate(instance=response, schema=schema)
        assert len(response["systematic_discrepancies"]) == 1

    def test_results_analyzer_empty_arrays(self):
        """Test empty arrays for optional fields."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "FAILED",
            "systematic_discrepancies": [],
            "recommendations": [],
            "confidence_factors": [],
            "summary": "Test results analysis"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["systematic_discrepancies"] == []
        assert response["recommendations"] == []

    # ========== COMPARISON VALIDATOR SCHEMA EDGE CASES ==========

    def test_comparison_validator_all_verdict_enums(self):
        """Test all valid verdict enum values."""
        valid_verdicts = ["approve", "needs_revision"]
        
        for verdict in valid_verdicts:
            response = {
                "stage_id": "stage1",
                "verdict": verdict,
                "accuracy_check": {"status": "pass"},
                "math_check": {"status": "pass"},
                "summary": "Test comparison validation"
            }
            schema = load_schema("comparison_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["verdict"] == verdict

    def test_comparison_validator_all_check_status_enums(self):
        """Test all valid check status enum values."""
        valid_statuses = ["pass", "fail", "warning"]
        
        for status in valid_statuses:
            response = {
                "stage_id": "stage1",
                "verdict": "approve",
                "accuracy_check": {"status": status},
                "math_check": {"status": status},
                "summary": "Test comparison validation"
            }
            schema = load_schema("comparison_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["accuracy_check"]["status"] == status
            assert response["math_check"]["status"] == status

    def test_comparison_validator_all_issue_severity_enums(self):
        """Test all valid issue severity enum values."""
        valid_severities = ["blocking", "major", "minor"]
        
        for severity in valid_severities:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "accuracy_check": {"status": "pass"},
                "math_check": {"status": "fail"},
                "issues": [
                    {
                        "severity": severity,
                        "category": "math",
                        "description": "Test issue"
                    }
                ],
                "summary": "Test comparison validation"
            }
            schema = load_schema("comparison_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["severity"] == severity

    def test_comparison_validator_all_issue_category_enums(self):
        """Test all valid issue category enum values."""
        valid_categories = ["accuracy", "math", "classification", "documentation"]
        
        for category in valid_categories:
            response = {
                "stage_id": "stage1",
                "verdict": "needs_revision",
                "accuracy_check": {"status": "pass"},
                "math_check": {"status": "pass"},
                "issues": [
                    {
                        "severity": "minor",
                        "category": category,
                        "description": "Test issue"
                    }
                ],
                "summary": "Test comparison validation"
            }
            schema = load_schema("comparison_validator_output_schema.json")
            validate(instance=response, schema=schema)
            assert response["issues"][0]["category"] == category

    def test_comparison_validator_accuracy_check_complete(self):
        """Test accuracy_check with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "accuracy_check": {
                "status": "pass",
                "paper_values_verified": True,
                "simulation_values_verified": True,
                "units_consistent": True,
                "axis_ranges_appropriate": True,
                "notes": "All values verified"
            },
            "math_check": {"status": "pass"},
            "summary": "Test comparison validation"
        }
        schema = load_schema("comparison_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["accuracy_check"]["paper_values_verified"] is True
        assert response["accuracy_check"]["units_consistent"] is True

    def test_comparison_validator_math_check_complete(self):
        """Test math_check with all fields including errors."""
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "accuracy_check": {"status": "pass"},
            "math_check": {
                "status": "fail",
                "discrepancy_calculations_correct": False,
                "percentage_calculations_correct": True,
                "classification_matches_thresholds": True,
                "errors_found": [
                    {
                        "calculation": "relative_error",
                        "reported_value": 5.0,
                        "correct_value": 4.5,
                        "impact": "Minor impact on classification"
                    }
                ],
                "notes": "Calculation error found"
            },
            "summary": "Test comparison validation"
        }
        schema = load_schema("comparison_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["math_check"]["discrepancy_calculations_correct"] is False
        assert len(response["math_check"]["errors_found"]) == 1

    def test_comparison_validator_classification_check(self):
        """Test classification_check with misclassifications."""
        response = {
            "stage_id": "stage1",
            "verdict": "needs_revision",
            "accuracy_check": {"status": "pass"},
            "math_check": {"status": "pass"},
            "classification_check": {
                "status": "fail",
                "misclassifications": [
                    {
                        "result_id": "result1",
                        "reported_classification": "excellent",
                        "correct_classification": "acceptable",
                        "discrepancy_value": 3.5,
                        "threshold_used": 2.0
                    }
                ],
                "notes": "Classification error found"
            },
            "summary": "Test comparison validation"
        }
        schema = load_schema("comparison_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert len(response["classification_check"]["misclassifications"]) == 1

    def test_comparison_validator_documentation_check(self):
        """Test documentation_check with all fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "accuracy_check": {"status": "pass"},
            "math_check": {"status": "pass"},
            "documentation_check": {
                "status": "pass",
                "all_discrepancies_logged": True,
                "sources_cited": True,
                "assumptions_documented": True,
                "missing_documentation": [],
                "notes": "Documentation complete"
            },
            "summary": "Test comparison validation"
        }
        schema = load_schema("comparison_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["documentation_check"]["all_discrepancies_logged"] is True

    def test_comparison_validator_empty_arrays(self):
        """Test empty arrays for optional fields."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "accuracy_check": {"status": "pass"},
            "math_check": {"status": "pass"},
            "issues": [],
            "revision_suggestions": [],
            "summary": "Test comparison validation"
        }
        schema = load_schema("comparison_validator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["issues"] == []
        assert response["revision_suggestions"] == []


class TestEdgeCaseRejections:
    """Test that invalid edge cases are properly REJECTED.
    
    These tests verify that schemas are strict enough to catch bugs.
    A test that fails to reject invalid data indicates a schema bug.
    """

    # ========== INVALID ENUM VALUES ==========

    def test_planner_invalid_paper_domain_rejected(self):
        """Invalid paper_domain value must be rejected."""
        response = {
            "paper_id": "test",
            "paper_domain": "invalid_domain",  # Not in enum
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "paper_domain" in str(exc_info.value) or "invalid_domain" in str(exc_info.value)

    def test_supervisor_invalid_verdict_rejected(self):
        """Invalid verdict value must be rejected."""
        response = {
            "verdict": "invalid_verdict",  # Not in enum
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
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
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value) or "invalid_verdict" in str(exc_info.value)

    def test_execution_validator_invalid_verdict_rejected(self):
        """Invalid verdict value must be rejected."""
        response = {
            "stage_id": "stage1",
            "verdict": "invalid",  # Should be pass/warning/fail
            "execution_status": {"completed": True},
            "files_check": {
                "expected_files": [],
                "found_files": [],
                "missing_files": [],
                "all_present": True
            },
            "summary": "Test"
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value) or "invalid" in str(exc_info.value)

    def test_physics_sanity_invalid_verdict_rejected(self):
        """Invalid verdict value must be rejected."""
        response = {
            "stage_id": "stage1",
            "verdict": "approved",  # Should be pass/warning/fail/design_flaw
            "conservation_checks": [],
            "value_range_checks": [],
            "summary": "Test"
        }
        schema = load_schema("physics_sanity_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "verdict" in str(exc_info.value) or "approved" in str(exc_info.value)

    def test_results_analyzer_invalid_classification_rejected(self):
        """Invalid overall_classification value must be rejected."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "GOOD_MATCH",  # Not a valid enum
            "summary": "Test"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "overall_classification" in str(exc_info.value) or "GOOD_MATCH" in str(exc_info.value)

    def test_design_reviewer_invalid_checklist_status_rejected(self):
        """Invalid checklist status value must be rejected."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "geometry": {"status": "ok"},  # Should be pass/fail/warning
                "physics": {"status": "pass"},
                "materials": {"status": "pass"},
                "unit_system": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "resolution": {"status": "pass"},
                "outputs": {"status": "pass"},
                "runtime": {"status": "pass"}
            },
            "issues": [],
            "summary": "Test"
        }
        schema = load_schema("design_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail because "ok" is not a valid status enum value
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["status", "ok", "enum"])

    # ========== INVALID TYPE TESTS ==========

    def test_supervisor_physics_assessment_boolean_type_enforced(self):
        """physics_plausible must be boolean, not string."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": "true",  # String instead of boolean
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        assert "physics_plausible" in str(exc_info.value) or "boolean" in str(exc_info.value)

    def test_code_generator_runtime_must_be_number(self):
        """estimated_runtime_minutes must be a number, not string."""
        response = {
            "stage_id": "stage1",
            "code": "import meep as mp\n# code here",
            "expected_outputs": [],
            "estimated_runtime_minutes": "10",  # String instead of number
            "summary": "Test"
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["estimated_runtime_minutes", "number", "type"])

    def test_planner_extracted_parameter_value_invalid_type_rejected(self):
        """extracted_parameters value must be number, string, or array of numbers."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [
                {
                    "name": "p1",
                    "value": {"nested": "object"},  # Object not allowed
                    "unit": "nm",
                    "source": "text"
                }
            ],
            "targets": [],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail because object is not a valid type for value

    # ========== BOUNDARY VALUE TESTS ==========

    def test_results_analyzer_confidence_exceeds_maximum_rejected(self):
        """Confidence > 1.0 must be rejected."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "ACCEPTABLE_MATCH",
            "confidence": 1.5,  # Exceeds maximum of 1
            "summary": "Test"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["confidence", "maximum", "1.5"])

    def test_results_analyzer_confidence_below_minimum_rejected(self):
        """Confidence < 0.0 must be rejected."""
        response = {
            "stage_id": "stage1",
            "per_result_reports": [],
            "figure_comparisons": [],
            "overall_classification": "ACCEPTABLE_MATCH",
            "confidence": -0.5,  # Below minimum of 0
            "summary": "Test"
        }
        schema = load_schema("results_analyzer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["confidence", "minimum", "-0.5"])

    def test_supervisor_summary_empty_string_rejected(self):
        """Empty summary string must be rejected (minLength: 1)."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": "",  # Empty string violates minLength: 1
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["summary", "minLength", "short"])

    # ========== REQUIRED FIELD TESTS ==========

    def test_simulation_designer_missing_unit_system_characteristic_length_rejected(self):
        """Missing required characteristic_length_m must be rejected."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                # Missing required: characteristic_length_m
                "length_unit": "µm"
            },
            "geometry": {"structures": []},
            "materials": [],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10.0,
                "memory_estimate_gb": 2.0
            },
            "summary": "Test"
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["characteristic_length_m", "required"])

    def test_code_reviewer_missing_all_outputs_accounted_rejected(self):
        """Missing required all_outputs_accounted must be rejected."""
        response = {
            "stage_id": "stage1",
            "verdict": "approve",
            "checklist_results": {
                "unit_normalization": {"status": "pass"},
                "numerics": {"status": "pass"},
                "source": {"status": "pass"},
                "domain": {"status": "pass"},
                "monitors": {"status": "pass"},
                "visualization": {"status": "pass"},
                "code_quality": {"status": "pass"},
                "runtime": {"status": "pass"},
                "meep_api": {"status": "pass"},
                "expected_outputs": {
                    "status": "pass"
                    # Missing required: all_outputs_accounted
                }
            },
            "issues": [],
            "summary": "Test"
        }
        schema = load_schema("code_reviewer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["all_outputs_accounted", "required"])

    def test_execution_validator_missing_files_check_fields_rejected(self):
        """Missing required files_check fields must be rejected."""
        response = {
            "stage_id": "stage1",
            "verdict": "pass",
            "execution_status": {"completed": True},
            "files_check": {
                # Missing required fields: expected_files, found_files, missing_files, all_present
            },
            "summary": "Test"
        }
        schema = load_schema("execution_validator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["expected_files", "found_files", "missing_files", "all_present", "required"])

    # ========== ARRAY ITEM TYPE TESTS ==========

    def test_planner_stages_array_invalid_item_type_rejected(self):
        """stages array must contain objects, not strings."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": ["stage1", "stage2"],  # Strings instead of objects
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        # Should fail because array items must be objects

    def test_supervisor_recommendations_array_invalid_item_rejected(self):
        """recommendations array items must be objects with required fields."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "recommendations": [
                {
                    # Missing required: action, priority
                    "rationale": "Test"
                }
            ],
            "summary": "Test",
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["action", "priority", "required"])

    # ========== NESTED OBJECT VALIDATION TESTS ==========

    def test_planner_stage_missing_required_fields_rejected(self):
        """Stage object missing required fields must be rejected."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [],
            "stages": [
                {
                    "stage_id": "stage1",
                    # Missing: stage_type, name, description, targets, dependencies
                }
            ],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["stage_type", "name", "description", "targets", "dependencies", "required"])

    def test_planner_target_missing_required_fields_rejected(self):
        """Target object missing required fields must be rejected."""
        response = {
            "paper_id": "test",
            "paper_domain": "other",
            "title": "Test",
            "summary": "Test",
            "extracted_parameters": [],
            "targets": [
                {
                    "figure_id": "fig1",
                    # Missing: description, type, simulation_class
                }
            ],
            "stages": [],
            "assumptions": {},
            "progress": {"stages": []},
        }
        schema = load_schema("planner_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["description", "type", "simulation_class", "required"])

    def test_simulation_designer_source_missing_required_fields_rejected(self):
        """Source object missing required fields must be rejected."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm"
            },
            "geometry": {"structures": []},
            "materials": [],
            "sources": [
                {
                    # Missing required: type, center, size
                    "component": "Ez"
                }
            ],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10.0,
                "memory_estimate_gb": 2.0
            },
            "summary": "Test"
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["type", "center", "size", "required"])

    def test_simulation_designer_material_missing_required_fields_rejected(self):
        """Material object missing required fields must be rejected."""
        response = {
            "stage_id": "stage1",
            "design_description": "Test",
            "unit_system": {
                "characteristic_length_m": 1e-6,
                "length_unit": "µm"
            },
            "geometry": {"structures": []},
            "materials": [
                {
                    # Missing required: id, name, model_type
                    "source": "palik"
                }
            ],
            "sources": [],
            "boundary_conditions": {},
            "monitors": [],
            "performance_estimate": {
                "runtime_estimate_minutes": 10.0,
                "memory_estimate_gb": 2.0
            },
            "summary": "Test"
        }
        schema = load_schema("simulation_designer_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["id", "name", "model_type", "required"])

    # ========== NULL VALUE CONSTRAINT TESTS ==========

    def test_supervisor_summary_null_rejected(self):
        """summary field cannot be null (required string)."""
        response = {
            "verdict": "ok_continue",
            "validation_hierarchy_status": {
                "material_validation": "not_done",
                "single_structure": "not_done",
                "arrays_systems": "not_done",
                "parameter_sweeps": "not_done",
            },
            "main_physics_assessment": {
                "physics_plausible": True,
                "conservation_satisfied": True,
                "value_ranges_reasonable": True,
            },
            "summary": None,  # Null not allowed for required string field
        }
        schema = load_schema("supervisor_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["summary", "null", "type", "string"])

    def test_code_generator_code_null_rejected(self):
        """code field cannot be null (required string)."""
        response = {
            "stage_id": "stage1",
            "code": None,  # Null not allowed for required string field
            "expected_outputs": [],
            "estimated_runtime_minutes": 1.0,
            "summary": "Test"
        }
        schema = load_schema("code_generator_output_schema.json")
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=response, schema=schema)
        error_str = str(exc_info.value)
        assert any(term in error_str for term in ["code", "null", "type", "string"])

