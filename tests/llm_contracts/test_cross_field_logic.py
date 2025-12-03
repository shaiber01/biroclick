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

            # Verdict must be one of the valid enum values
            assert verdict in ["approve", "needs_revision"], (
                f"{agent}: Invalid verdict '{verdict}'. Must be 'approve' or 'needs_revision'."
            )

            # Issues must be a list
            assert isinstance(issues, list), f"{agent}: 'issues' must be a list, got {type(issues)}."

            if verdict == "needs_revision":
                # If rejecting, MUST have issues
                assert issues, f"{agent}: Verdict is 'needs_revision' but 'issues' list is empty."
                
                # Each issue must have all required fields
                required_fields = ["severity", "category", "description", "suggested_fix"]
                for idx, issue in enumerate(issues):
                    assert isinstance(issue, dict), (
                        f"{agent}: Issue {idx} must be a dict, got {type(issue)}."
                    )
                    for field in required_fields:
                        assert field in issue, (
                            f"{agent}: Issue {idx} missing required field '{field}'."
                        )
                        assert issue[field], (
                            f"{agent}: Issue {idx} has empty '{field}' field."
                        )
                    
                    # Severity must be valid enum
                    assert issue["severity"] in ["blocking", "major", "minor"], (
                        f"{agent}: Issue {idx} has invalid severity '{issue['severity']}'."
                    )
                    
                    # Category must be valid (varies by reviewer type)
                    valid_categories = {
                        "plan_reviewer": ["coverage", "staging", "parameters", "assumptions", 
                                         "performance", "digitized_data", "material_validation", 
                                         "output_specifications"],
                        "design_reviewer": ["geometry", "physics", "materials", "unit_system", 
                                           "source", "domain", "resolution", "outputs", "runtime"],
                        "code_reviewer": ["unit_normalization", "numerics", "source", "domain", 
                                         "monitors", "visualization", "code_quality", "runtime", 
                                         "meep_api", "expected_outputs"]
                    }
                    if agent in valid_categories:
                        assert issue["category"] in valid_categories[agent], (
                            f"{agent}: Issue {idx} has invalid category '{issue['category']}'."
                        )
                    
                    # Blocking issues should not exist with approve verdict (checked below)
                    # But if needs_revision, blocking issues are expected
                    if issue["severity"] == "blocking":
                        # Blocking issues should have detailed descriptions
                        assert len(issue["description"]) > 20, (
                            f"{agent}: Issue {idx} is blocking but has insufficient description."
                        )

            elif verdict == "approve":
                # If approving, cannot have blocking or critical issues
                blocking_issues = [i for i in issues if i.get("severity") == "blocking"]
                assert not blocking_issues, (
                    f"{agent}: Verdict is 'approve' but has {len(blocking_issues)} blocking issue(s)."
                )
                
                # If there are issues with approve, they should only be minor
                for idx, issue in enumerate(issues):
                    assert issue.get("severity") != "blocking", (
                        f"{agent}: Issue {idx} has blocking severity but verdict is approve."
                    )

    def test_supervisor_decision_consistency(self):
        """Supervisor decision fields must match the verdict."""
        try:
            response = load_mock_response("supervisor")
        except FileNotFoundError:
            pytest.skip("Supervisor mock not found")

        verdict = response.get("verdict")
        
        # Verdict must be valid enum value
        valid_verdicts = ["ok_continue", "replan_needed", "change_priority", "ask_user", 
                          "backtrack_to_stage", "all_complete"]
        assert verdict in valid_verdicts, (
            f"Invalid verdict '{verdict}'. Must be one of {valid_verdicts}."
        )

        if verdict == "backtrack_to_stage":
            assert "backtrack_decision" in response, (
                "Missing 'backtrack_decision' for backtrack verdict."
            )
            backtrack = response["backtrack_decision"]
            assert isinstance(backtrack, dict), (
                "'backtrack_decision' must be a dict."
            )
            
            # All required fields must be present
            assert backtrack.get("accepted") is not None, (
                "backtrack_decision missing required field 'accepted'."
            )
            assert backtrack.get("target_stage_id"), (
                "Backtrack target stage ID missing or empty."
            )
            assert isinstance(backtrack.get("target_stage_id"), str), (
                "target_stage_id must be a string."
            )
            assert backtrack.get("stages_to_invalidate") is not None, (
                "backtrack_decision missing required field 'stages_to_invalidate'."
            )
            assert isinstance(backtrack.get("stages_to_invalidate"), list), (
                "stages_to_invalidate must be a list."
            )
            assert backtrack.get("reason"), (
                "backtrack_decision missing required field 'reason'."
            )
            
            # If backtracking, should_stop should typically be False (unless final stage)
            # But we don't enforce this as it could be valid to stop after backtrack
            
        elif verdict == "ask_user":
            user_question = response.get("user_question")
            assert user_question, (
                "Verdict is 'ask_user' but 'user_question' is empty/missing."
            )
            assert isinstance(user_question, str), (
                f"user_question must be a string, got {type(user_question)}."
            )
            assert len(user_question.strip()) > 0, (
                "user_question cannot be empty or whitespace."
            )
            # User question should be a meaningful question
            assert len(user_question) > 10, (
                "user_question appears too short to be meaningful."
            )
            
        elif verdict == "all_complete":
            should_stop = response.get("should_stop")
            assert should_stop is True, (
                f"Verdict is 'all_complete' but 'should_stop' is {should_stop}, not True."
            )
            # If all complete, progress should reflect completion
            progress = response.get("progress_summary", {})
            if progress:
                assert progress.get("stages_remaining", 1) == 0, (
                    "all_complete verdict but stages_remaining is not 0."
                )
        
        elif verdict == "ok_continue":
            # Should not have backtrack_decision or user_question
            if "backtrack_decision" in response:
                backtrack = response["backtrack_decision"]
                if backtrack and backtrack.get("suggest_backtrack"):
                    # This is inconsistent - ok_continue shouldn't suggest backtrack
                    assert False, (
                        "Verdict is 'ok_continue' but backtrack_decision suggests backtrack."
                    )
        
        # Validation hierarchy status must be present and valid
        hierarchy = response.get("validation_hierarchy_status", {})
        assert isinstance(hierarchy, dict), (
            "validation_hierarchy_status must be a dict."
        )
        required_hierarchy_fields = ["material_validation", "single_structure", 
                                   "arrays_systems", "parameter_sweeps"]
        for field in required_hierarchy_fields:
            assert field in hierarchy, (
                f"validation_hierarchy_status missing required field '{field}'."
            )
            assert hierarchy[field] in ["passed", "partial", "failed", "not_done"], (
                f"validation_hierarchy_status.{field} has invalid value '{hierarchy[field]}'."
            )
        
        # Main physics assessment must be present
        physics = response.get("main_physics_assessment", {})
        assert isinstance(physics, dict), (
            "main_physics_assessment must be a dict."
        )
        required_physics_fields = ["physics_plausible", "conservation_satisfied", 
                                  "value_ranges_reasonable"]
        for field in required_physics_fields:
            assert field in physics, (
                f"main_physics_assessment missing required field '{field}'."
            )
            assert isinstance(physics[field], bool), (
                f"main_physics_assessment.{field} must be a boolean."
            )

    def test_planner_stages_integrity(self):
        """Planner stages must form a coherent plan."""
        try:
            response = load_mock_response("planner")
        except FileNotFoundError:
            pytest.skip("Planner mock not found")

        stages = response.get("stages", [])
        assert isinstance(stages, list), "stages must be a list."

        if not stages:
            pytest.skip("No stages in planner response")

        # Collect all stage IDs and validate uniqueness
        stage_ids = {}
        for idx, stage in enumerate(stages):
            assert isinstance(stage, dict), f"Stage {idx} must be a dict."
            
            stage_id = stage.get("stage_id")
            assert stage_id, f"Stage {idx} missing required field 'stage_id'."
            assert isinstance(stage_id, str), f"Stage {idx} stage_id must be a string."
            
            # Stage IDs must be unique
            assert stage_id not in stage_ids, (
                f"Duplicate stage_id '{stage_id}' found at indices {stage_ids[stage_id]} and {idx}."
            )
            stage_ids[stage_id] = idx
        
        # Validate all required stage fields
        valid_stage_types = ["MATERIAL_VALIDATION", "SINGLE_STRUCTURE", "ARRAY_SYSTEM", 
                           "PARAMETER_SWEEP", "COMPLEX_PHYSICS"]
        
        for stage in stages:
            stage_id = stage["stage_id"]
            
            # Required fields
            assert "stage_type" in stage, f"Stage {stage_id} missing required field 'stage_type'."
            assert stage["stage_type"] in valid_stage_types, (
                f"Stage {stage_id} has invalid stage_type '{stage['stage_type']}'."
            )
            assert "name" in stage, f"Stage {stage_id} missing required field 'name'."
            assert "description" in stage, f"Stage {stage_id} missing required field 'description'."
            assert "targets" in stage, f"Stage {stage_id} missing required field 'targets'."
            assert "dependencies" in stage, f"Stage {stage_id} missing required field 'dependencies'."
            
            # Dependencies must be a list
            deps = stage.get("dependencies", [])
            assert isinstance(deps, list), (
                f"Stage {stage_id} dependencies must be a list."
            )
            
            # Targets must be a list
            targets = stage.get("targets", [])
            assert isinstance(targets, list), (
                f"Stage {stage_id} targets must be a list."
            )
            
            # Dependencies must reference existing stages
            for dep in deps:
                assert dep in stage_ids, (
                    f"Stage {stage_id} depends on unknown stage '{dep}'."
                )
                # Cannot depend on itself
                assert dep != stage_id, (
                    f"Stage {stage_id} cannot depend on itself."
                )
            
            # Stage 0 (MATERIAL_VALIDATION) should have no dependencies
            if stage["stage_type"] == "MATERIAL_VALIDATION":
                assert not deps, (
                    f"Stage {stage_id} (MATERIAL_VALIDATION) should have no dependencies."
                )
        
        # Validate dependency ordering (topological sort)
        seen_ids = set()
        for stage in stages:
            stage_id = stage["stage_id"]
            missing_deps = [dep for dep in stage.get("dependencies", []) if dep not in seen_ids]
            assert not missing_deps, (
                f"Stage {stage_id} appears before its dependencies: {missing_deps}."
            )
            seen_ids.add(stage_id)
        
        # Validate targets reference valid figure IDs
        all_targets = response.get("targets", [])
        target_figure_ids = {t.get("figure_id") for t in all_targets if isinstance(t, dict)}
        
        for stage in stages:
            stage_id = stage["stage_id"]
            stage_targets = stage.get("targets", [])
            
            for target_ref in stage_targets:
                # Target references should match figure_ids from targets list
                # But some targets might be material IDs (for stage 0), so we check if it's a figure_id
                if target_ref in target_figure_ids:
                    # This is a valid figure reference
                    pass
                elif stage["stage_type"] == "MATERIAL_VALIDATION":
                    # Material validation targets might be material IDs, which is OK
                    pass
                else:
                    # For non-material stages, targets should reference figures
                    # But we can't enforce this strictly without knowing all valid IDs
                    # So we just check it's not empty
                    assert target_ref, (
                        f"Stage {stage_id} has empty target reference."
                    )
        
        # Validate expected_outputs consistency
        for stage in stages:
            stage_id = stage["stage_id"]
            expected_outputs = stage.get("expected_outputs", [])
            
            if expected_outputs:
                assert isinstance(expected_outputs, list), (
                    f"Stage {stage_id} expected_outputs must be a list."
                )
                
                for idx, output_spec in enumerate(expected_outputs):
                    assert isinstance(output_spec, dict), (
                        f"Stage {stage_id} expected_outputs[{idx}] must be a dict."
                    )
                    
                    # Required fields for output spec
                    assert "artifact_type" in output_spec, (
                        f"Stage {stage_id} expected_outputs[{idx}] missing 'artifact_type'."
                    )
                    assert "filename_pattern" in output_spec, (
                        f"Stage {stage_id} expected_outputs[{idx}] missing 'filename_pattern'."
                    )
                    assert "description" in output_spec, (
                        f"Stage {stage_id} expected_outputs[{idx}] missing 'description'."
                    )
                    
                    # Artifact type must be valid
                    valid_artifact_types = ["spectrum_csv", "field_data_npz", "field_plot_png", 
                                          "spectrum_plot_png", "dispersion_csv", "raw_h5"]
                    assert output_spec["artifact_type"] in valid_artifact_types, (
                        f"Stage {stage_id} expected_outputs[{idx}] has invalid artifact_type."
                    )

    def test_code_generator_safety_compliance(self):
        """Code generator must confirm safety checks are passed."""
        try:
            response = load_mock_response("code_generator")
        except FileNotFoundError:
            pytest.skip("Code generator mock not found")

        # Required fields
        assert "stage_id" in response, "Code generator missing required field 'stage_id'."
        assert "code" in response, "Code generator missing required field 'code'."
        assert "expected_outputs" in response, "Code generator missing required field 'expected_outputs'."
        assert "estimated_runtime_minutes" in response, (
            "Code generator missing required field 'estimated_runtime_minutes'."
        )
        
        # Code must be non-empty
        code = response.get("code", "")
        assert isinstance(code, str), "code must be a string."
        assert len(code.strip()) > 0, "code cannot be empty."
        assert len(code) > 50, "code appears too short to be valid simulation code."
        
        # Stage ID must be valid
        stage_id = response.get("stage_id")
        assert isinstance(stage_id, str), "stage_id must be a string."
        assert stage_id, "stage_id cannot be empty."
        
        # Safety checks must all pass
        safety = response.get("safety_checks", {})
        assert isinstance(safety, dict), "safety_checks must be a dict."
        
        required_safety_checks = ["no_plt_show", "no_input", "uses_plt_savefig_close", 
                                 "relative_paths_only", "includes_result_json"]
        for check in required_safety_checks:
            assert check in safety, f"Missing required safety check '{check}'."
            assert isinstance(safety[check], bool), (
                f"safety_checks.{check} must be a boolean."
            )
            assert safety[check] is True, (
                f"Code generator failed safety check: {check}"
            )
        
        # Runtime must be positive
        runtime = response.get("estimated_runtime_minutes", 0)
        assert isinstance(runtime, (int, float)), "estimated_runtime_minutes must be a number."
        assert runtime > 0, f"Estimated runtime must be positive, got {runtime}."
        assert runtime < 10000, "Estimated runtime appears unreasonably large (>10000 minutes)."
        
        # Expected outputs must be consistent
        expected_outputs = response.get("expected_outputs", [])
        assert isinstance(expected_outputs, list), "expected_outputs must be a list."
        assert len(expected_outputs) > 0, "expected_outputs cannot be empty."
        
        for idx, output_spec in enumerate(expected_outputs):
            assert isinstance(output_spec, dict), (
                f"expected_outputs[{idx}] must be a dict."
            )
            
            # Required fields
            assert "artifact_type" in output_spec, (
                f"expected_outputs[{idx}] missing 'artifact_type'."
            )
            assert "filename" in output_spec, (
                f"expected_outputs[{idx}] missing 'filename'."
            )
            assert "description" in output_spec, (
                f"expected_outputs[{idx}] missing 'description'."
            )
            
            # Filename must be non-empty
            filename = output_spec.get("filename", "")
            assert filename, f"expected_outputs[{idx}] has empty filename."
            assert isinstance(filename, str), (
                f"expected_outputs[{idx}] filename must be a string."
            )
        
        # Unit system consistency (if present)
        unit_system = response.get("unit_system_used", {})
        if unit_system:
            assert isinstance(unit_system, dict), "unit_system_used must be a dict."
            assert "characteristic_length_m" in unit_system, (
                "unit_system_used missing 'characteristic_length_m'."
            )
            char_length = unit_system.get("characteristic_length_m")
            assert isinstance(char_length, (int, float)), (
                "characteristic_length_m must be a number."
            )
            assert char_length > 0, "characteristic_length_m must be positive."
            assert char_length < 1, "characteristic_length_m should be < 1 (in meters)."
        
        # Materials used consistency (if present)
        materials_used = response.get("materials_used", [])
        if materials_used:
            assert isinstance(materials_used, list), "materials_used must be a list."
            for idx, material in enumerate(materials_used):
                assert isinstance(material, dict), (
                    f"materials_used[{idx}] must be a dict."
                )
                assert "material_name" in material, (
                    f"materials_used[{idx}] missing 'material_name'."
                )
                assert material.get("material_name"), (
                    f"materials_used[{idx}] has empty material_name."
                )

    def test_simulation_designer_content(self):
        """Simulation design must be non-empty."""
        try:
            response = load_mock_response("simulation_designer")
        except FileNotFoundError:
            pytest.skip("Simulation designer mock not found")

        # Required fields
        assert "stage_id" in response, "Simulation designer missing required field 'stage_id'."
        assert "design_description" in response, (
            "Simulation designer missing required field 'design_description'."
        )
        assert "unit_system" in response, (
            "Simulation designer missing required field 'unit_system'."
        )
        assert "geometry" in response, (
            "Simulation designer missing required field 'geometry'."
        )
        assert "materials" in response, (
            "Simulation designer missing required field 'materials'."
        )
        assert "sources" in response, (
            "Simulation designer missing required field 'sources'."
        )
        assert "boundary_conditions" in response, (
            "Simulation designer missing required field 'boundary_conditions'."
        )
        assert "monitors" in response, (
            "Simulation designer missing required field 'monitors'."
        )
        assert "performance_estimate" in response, (
            "Simulation designer missing required field 'performance_estimate'."
        )
        
        # Stage ID validation
        stage_id = response.get("stage_id")
        assert isinstance(stage_id, str), "stage_id must be a string."
        assert stage_id, "stage_id cannot be empty."
        
        # Design description must be meaningful
        design_desc = response.get("design_description", "")
        assert isinstance(design_desc, str), "design_description must be a string."
        assert len(design_desc.strip()) > 0, "design_description cannot be empty."
        assert len(design_desc) > 20, "design_description appears too short."
        
        # Geometry validation
        geometry = response.get("geometry", {})
        assert isinstance(geometry, dict), "geometry must be a dict."
        
        structures = geometry.get("structures", [])
        assert isinstance(structures, list), "geometry.structures must be a list."
        assert structures, "Simulation design has no structures."
        
        # Collect material references from structures
        material_refs_in_structures = set()
        for idx, structure in enumerate(structures):
            assert isinstance(structure, dict), (
                f"geometry.structures[{idx}] must be a dict."
            )
            assert "name" in structure, (
                f"geometry.structures[{idx}] missing 'name'."
            )
            assert "type" in structure, (
                f"geometry.structures[{idx}] missing 'type'."
            )
            assert structure["type"] in ["cylinder", "sphere", "block", "ellipsoid", 
                                       "cone", "prism", "custom"], (
                f"geometry.structures[{idx}] has invalid type '{structure['type']}'."
            )
            
            # Material reference must exist
            if "material_ref" in structure:
                material_ref = structure.get("material_ref")
                assert material_ref, (
                    f"geometry.structures[{idx}] has empty material_ref."
                )
                material_refs_in_structures.add(material_ref)
        
        # Unit system validation
        unit_system = response.get("unit_system", {})
        assert isinstance(unit_system, dict), "unit_system must be a dict."
        
        assert "characteristic_length_m" in unit_system, (
            "unit_system missing required field 'characteristic_length_m'."
        )
        char_length = unit_system.get("characteristic_length_m", 0)
        assert isinstance(char_length, (int, float)), (
            "characteristic_length_m must be a number."
        )
        assert char_length > 0, f"Characteristic length must be positive, got {char_length}."
        assert char_length < 1, "characteristic_length_m should be < 1 (in meters)."
        
        assert "length_unit" in unit_system, (
            "unit_system missing required field 'length_unit'."
        )
        length_unit = unit_system.get("length_unit", "")
        assert isinstance(length_unit, str), "length_unit must be a string."
        assert length_unit, "length_unit cannot be empty."
        
        # Materials validation
        materials = response.get("materials", [])
        assert isinstance(materials, list), "materials must be a list."
        assert len(materials) > 0, "materials list cannot be empty."
        
        material_ids = set()
        for idx, material in enumerate(materials):
            assert isinstance(material, dict), f"materials[{idx}] must be a dict."
            
            # Required fields
            assert "id" in material, f"materials[{idx}] missing required field 'id'."
            assert "name" in material, f"materials[{idx}] missing required field 'name'."
            assert "model_type" in material, (
                f"materials[{idx}] missing required field 'model_type'."
            )
            
            material_id = material.get("id")
            assert material_id, f"materials[{idx}] has empty id."
            assert isinstance(material_id, str), f"materials[{idx}] id must be a string."
            
            # Material IDs must be unique
            assert material_id not in material_ids, (
                f"Duplicate material id '{material_id}' found."
            )
            material_ids.add(material_id)
            
            # Model type must be valid
            assert material["model_type"] in ["constant", "tabulated", "drude", 
                                            "lorentz", "drude_lorentz"], (
                f"materials[{idx}] has invalid model_type '{material['model_type']}'."
            )
        
        # Structures must reference valid materials
        for material_ref in material_refs_in_structures:
            assert material_ref in material_ids, (
                f"Structure references unknown material '{material_ref}'."
            )
        
        # Sources validation
        sources = response.get("sources", [])
        assert isinstance(sources, list), "sources must be a list."
        assert len(sources) > 0, "sources list cannot be empty."
        
        for idx, source in enumerate(sources):
            assert isinstance(source, dict), f"sources[{idx}] must be a dict."
            assert "type" in source, f"sources[{idx}] missing 'type'."
            assert source["type"] in ["gaussian", "continuous", "eigenmode"], (
                f"sources[{idx}] has invalid type '{source['type']}'."
            )
        
        # Monitors validation
        monitors = response.get("monitors", [])
        assert isinstance(monitors, list), "monitors must be a list."
        assert len(monitors) > 0, "monitors list cannot be empty."
        
        for idx, monitor in enumerate(monitors):
            assert isinstance(monitor, dict), f"monitors[{idx}] must be a dict."
            assert "type" in monitor, f"monitors[{idx}] missing 'type'."
            assert monitor["type"] in ["flux", "field", "dft_fields", "near2far"], (
                f"monitors[{idx}] has invalid type '{monitor['type']}'."
            )
            assert "name" in monitor, f"monitors[{idx}] missing 'name'."
            assert monitor.get("name"), f"monitors[{idx}] has empty name."
        
        # Performance estimate validation
        perf = response.get("performance_estimate", {})
        assert isinstance(perf, dict), "performance_estimate must be a dict."
        
        assert "runtime_estimate_minutes" in perf, (
            "performance_estimate missing 'runtime_estimate_minutes'."
        )
        runtime = perf.get("runtime_estimate_minutes")
        assert isinstance(runtime, (int, float)), (
            "runtime_estimate_minutes must be a number."
        )
        assert runtime > 0, "runtime_estimate_minutes must be positive."
        
        assert "memory_estimate_gb" in perf, (
            "performance_estimate missing 'memory_estimate_gb'."
        )
        memory = perf.get("memory_estimate_gb")
        assert isinstance(memory, (int, float)), "memory_estimate_gb must be a number."
        assert memory > 0, "memory_estimate_gb must be positive."

    def test_execution_validator_logic(self):
        """Execution validator consistency check."""
        try:
            response = load_mock_response("execution_validator")
        except FileNotFoundError:
            pytest.skip("Execution validator mock not found")

        # Required fields
        assert "stage_id" in response, "Execution validator missing required field 'stage_id'."
        assert "verdict" in response, "Execution validator missing required field 'verdict'."
        assert "execution_status" in response, (
            "Execution validator missing required field 'execution_status'."
        )
        assert "files_check" in response, (
            "Execution validator missing required field 'files_check'."
        )
        assert "summary" in response, (
            "Execution validator missing required field 'summary'."
        )
        
        # Stage ID validation
        stage_id = response.get("stage_id")
        assert isinstance(stage_id, str), "stage_id must be a string."
        assert stage_id, "stage_id cannot be empty."
        
        # Verdict validation
        verdict = response.get("verdict")
        assert verdict in ["pass", "warning", "fail"], (
            f"Invalid verdict '{verdict}'. Must be 'pass', 'warning', or 'fail'."
        )
        
        # Execution status validation
        exec_status = response.get("execution_status", {})
        assert isinstance(exec_status, dict), "execution_status must be a dict."
        
        assert "completed" in exec_status, (
            "execution_status missing required field 'completed'."
        )
        completed = exec_status.get("completed")
        assert isinstance(completed, bool), "execution_status.completed must be a boolean."
        
        # Verdict must match execution status
        if completed is False:
            assert verdict == "fail", (
                f"Verdict should be 'fail' if execution not completed, got '{verdict}'."
            )
        
        # If execution completed successfully, verdict should not be fail
        if completed is True and exec_status.get("exit_code") == 0:
            assert verdict != "fail", (
                "Verdict should not be 'fail' if execution completed successfully."
            )
        
        # Files check validation
        files_check = response.get("files_check", {})
        assert isinstance(files_check, dict), "files_check must be a dict."
        
        assert "expected_files" in files_check, (
            "files_check missing required field 'expected_files'."
        )
        assert "found_files" in files_check, (
            "files_check missing required field 'found_files'."
        )
        assert "missing_files" in files_check, (
            "files_check missing required field 'missing_files'."
        )
        assert "all_present" in files_check, (
            "files_check missing required field 'all_present'."
        )
        
        expected = set(files_check.get("expected_files", []))
        found = set(files_check.get("found_files", []))
        missing = set(files_check.get("missing_files", []))
        all_present = files_check.get("all_present")
        
        assert isinstance(all_present, bool), "files_check.all_present must be a boolean."
        
        # Validate missing_files computation
        computed_missing = expected - found
        assert missing == computed_missing, (
            f"missing_files ({missing}) does not match expected - found ({computed_missing})."
        )
        
        # Validate all_present consistency
        if expected and expected.issubset(found):
            assert all_present is True, (
                f"all_present should be True if all expected files found, got {all_present}."
            )
        elif missing:
            assert all_present is False, (
                f"all_present should be False if files are missing, got {all_present}."
            )
        
        # If all_present is True, there should be no missing files
        if all_present is True:
            assert not missing, (
                f"all_present is True but missing_files is not empty: {missing}."
            )
            assert expected.issubset(found), (
                "all_present is True but not all expected files are in found_files."
            )
        
        # Spec compliance validation (if present)
        spec_compliance = files_check.get("spec_compliance", [])
        if spec_compliance:
            assert isinstance(spec_compliance, list), (
                "files_check.spec_compliance must be a list."
            )
            
            for idx, spec in enumerate(spec_compliance):
                assert isinstance(spec, dict), (
                    f"files_check.spec_compliance[{idx}] must be a dict."
                )
                
                # If exists=True, actual_filename should be present
                if spec.get("exists") is True:
                    assert spec.get("actual_filename"), (
                        f"spec_compliance[{idx}] exists=True but actual_filename is missing."
                    )
                
                # If exists=False, actual_filename should be null
                if spec.get("exists") is False:
                    assert spec.get("actual_filename") is None, (
                        f"spec_compliance[{idx}] exists=False but actual_filename is not None."
                    )
        
        # Data quality validation (if present)
        data_quality = response.get("data_quality", {})
        if data_quality:
            assert isinstance(data_quality, dict), "data_quality must be a dict."
            
            # If verdict is pass, data quality should be good
            if verdict == "pass":
                assert data_quality.get("nan_detected") is not True, (
                    "Verdict is 'pass' but nan_detected is True."
                )
                assert data_quality.get("inf_detected") is not True, (
                    "Verdict is 'pass' but inf_detected is True."
                )
        
        # Errors detected validation
        errors_detected = response.get("errors_detected", [])
        if errors_detected:
            assert isinstance(errors_detected, list), "errors_detected must be a list."
            
            # If there are critical errors, verdict should not be pass
            critical_errors = [e for e in errors_detected 
                              if isinstance(e, dict) and e.get("severity") == "critical"]
            if critical_errors:
                assert verdict != "pass", (
                    f"Verdict is 'pass' but {len(critical_errors)} critical error(s) detected."
                )

    def test_physics_sanity_logic(self):
        """Physics sanity logic check."""
        try:
            response = load_mock_response("physics_sanity")
        except FileNotFoundError:
            pytest.skip("Physics sanity mock not found")

        # Required fields
        assert "stage_id" in response, "Physics sanity missing required field 'stage_id'."
        assert "verdict" in response, "Physics sanity missing required field 'verdict'."
        assert "conservation_checks" in response, (
            "Physics sanity missing required field 'conservation_checks'."
        )
        assert "value_range_checks" in response, (
            "Physics sanity missing required field 'value_range_checks'."
        )
        assert "summary" in response, (
            "Physics sanity missing required field 'summary'."
        )
        
        # Stage ID validation
        stage_id = response.get("stage_id")
        assert isinstance(stage_id, str), "stage_id must be a string."
        assert stage_id, "stage_id cannot be empty."
        
        # Verdict validation
        verdict = response.get("verdict")
        assert verdict in ["pass", "warning", "fail", "design_flaw"], (
            f"Invalid verdict '{verdict}'. Must be 'pass', 'warning', 'fail', or 'design_flaw'."
        )
        
        # Conservation checks validation
        conservation_checks = response.get("conservation_checks", [])
        assert isinstance(conservation_checks, list), (
            "conservation_checks must be a list."
        )
        
        for idx, check in enumerate(conservation_checks):
            assert isinstance(check, dict), (
                f"conservation_checks[{idx}] must be a dict."
            )
            assert "law" in check, (
                f"conservation_checks[{idx}] missing required field 'law'."
            )
            assert "status" in check, (
                f"conservation_checks[{idx}] missing required field 'status'."
            )
            assert check["status"] in ["pass", "warning", "fail"], (
                f"conservation_checks[{idx}] has invalid status '{check['status']}'."
            )
        
        # Value range checks validation
        value_range_checks = response.get("value_range_checks", [])
        assert isinstance(value_range_checks, list), (
            "value_range_checks must be a list."
        )
        
        for idx, check in enumerate(value_range_checks):
            assert isinstance(check, dict), (
                f"value_range_checks[{idx}] must be a dict."
            )
            assert "quantity" in check, (
                f"value_range_checks[{idx}] missing required field 'quantity'."
            )
            assert "status" in check, (
                f"value_range_checks[{idx}] missing required field 'status'."
            )
            assert check["status"] in ["pass", "warning", "fail"], (
                f"value_range_checks[{idx}] has invalid status '{check['status']}'."
            )
        
        # Concerns validation
        concerns = response.get("concerns", [])
        assert isinstance(concerns, list), "concerns must be a list."
        
        for idx, concern in enumerate(concerns):
            assert isinstance(concern, dict), f"concerns[{idx}] must be a dict."
            assert "concern" in concern, (
                f"concerns[{idx}] missing required field 'concern'."
            )
            assert "severity" in concern, (
                f"concerns[{idx}] missing required field 'severity'."
            )
            assert concern["severity"] in ["critical", "moderate", "minor"], (
                f"concerns[{idx}] has invalid severity '{concern['severity']}'."
            )
        
        # Failed verdicts must have justification
        if verdict in ["fail", "design_flaw"]:
            failed_conservation = [
                c for c in conservation_checks 
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            failed_ranges = [
                c for c in value_range_checks 
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            critical_concerns = [
                c for c in concerns 
                if isinstance(c, dict) and c.get("severity") == "critical"
            ]
            
            assert (
                concerns or failed_conservation or failed_ranges or critical_concerns
            ), (
                f"Failed physics verdict '{verdict}' requires concerns or failed checks, "
                f"but none found."
            )
        
        # Pass verdict cannot have critical concerns or failed checks
        if verdict == "pass":
            critical_concerns = [
                c for c in concerns 
                if isinstance(c, dict) and c.get("severity") == "critical"
            ]
            assert not critical_concerns, (
                f"Pass verdict cannot have critical concerns, found {len(critical_concerns)}."
            )
            
            # All conservation checks should pass
            failed_conservation = [
                c for c in conservation_checks 
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            assert not failed_conservation, (
                f"Pass verdict cannot have failed conservation checks, "
                f"found {len(failed_conservation)}."
            )
            
            # All value range checks should pass
            failed_ranges = [
                c for c in value_range_checks 
                if isinstance(c, dict) and c.get("status") == "fail"
            ]
            assert not failed_ranges, (
                f"Pass verdict cannot have failed value range checks, "
                f"found {len(failed_ranges)}."
            )
        
        # Warning verdict should have some concerns or warnings
        if verdict == "warning":
            warning_concerns = [
                c for c in concerns 
                if isinstance(c, dict) and c.get("severity") in ["moderate", "minor"]
            ]
            warning_conservation = [
                c for c in conservation_checks 
                if isinstance(c, dict) and c.get("status") == "warning"
            ]
            warning_ranges = [
                c for c in value_range_checks 
                if isinstance(c, dict) and c.get("status") == "warning"
            ]
            
            # Warning should have at least some warnings or concerns
            assert (
                warning_concerns or warning_conservation or warning_ranges
            ), (
                "Warning verdict should have some warnings or concerns, but none found."
            )
        
        # Backtrack suggestion validation (if present)
        backtrack_suggestion = response.get("backtrack_suggestion", {})
        if backtrack_suggestion:
            assert isinstance(backtrack_suggestion, dict), (
                "backtrack_suggestion must be a dict."
            )
            assert "suggest_backtrack" in backtrack_suggestion, (
                "backtrack_suggestion missing required field 'suggest_backtrack'."
            )
            
            if backtrack_suggestion.get("suggest_backtrack") is True:
                assert "target_stage_id" in backtrack_suggestion, (
                    "backtrack_suggestion.suggest_backtrack=True but target_stage_id missing."
                )
                assert backtrack_suggestion.get("target_stage_id"), (
                    "backtrack_suggestion.target_stage_id cannot be empty."
                )
                assert "reason" in backtrack_suggestion, (
                    "backtrack_suggestion.suggest_backtrack=True but reason missing."
                )
                
                # If suggesting backtrack, verdict should typically be design_flaw
                # But we don't enforce this strictly as it could be valid for fail too

