"""Edge case handling tests for LLM responses."""

from jsonschema import validate

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
                "estimated_runtime_minutes": 1.0
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
            "estimated_runtime_minutes": 1.0
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
            "estimated_memory_gb": 0.1
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
            "materials_used": [],
            "dependencies_used": [],
            "progress_markers": []
        }
        schema = load_schema("code_generator_output_schema.json")
        validate(instance=response, schema=schema)
        assert response["materials_used"] == []
        assert response["dependencies_used"] == []
        assert response["progress_markers"] == []

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

