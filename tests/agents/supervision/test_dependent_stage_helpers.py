"""Tests for _get_dependent_stages helper."""

import pytest

from src.agents.supervision import _get_dependent_stages

class TestGetDependentStages:
    """Tests for _get_dependent_stages helper function."""

    def test_finds_direct_dependents(self):
        """Should find stages that directly depend on target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "stage0")
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 2, "Should find exactly 2 dependents"
        assert set(result) == {"stage1", "stage2"}, "Should find correct direct dependents"
        # Verify no duplicates
        assert len(result) == len(set(result)), "Result should not contain duplicates"

    def test_finds_transitive_dependents(self):
        """Should find stages that transitively depend on target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage1"]},
                {"stage_id": "stage3", "dependencies": ["stage2"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "stage0")
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 3, "Should find exactly 3 transitive dependents"
        assert set(result) == {"stage1", "stage2", "stage3"}, "Should find all transitive dependents"
        # Verify all stages are included
        assert "stage1" in result, "Direct dependent should be included"
        assert "stage2" in result, "Transitive dependent should be included"
        assert "stage3" in result, "Deep transitive dependent should be included"

    def test_returns_empty_for_leaf_stage(self):
        """Should return empty list for stage with no dependents."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "stage1")
        
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list for leaf stage"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_empty_plan(self):
        """Should handle empty plan gracefully."""
        result = _get_dependent_stages({}, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list for empty plan"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_missing_stages(self):
        """Should handle plan with no stages key."""
        plan = {"title": "Test Plan"}
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list when stages key is missing"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_complex_dependency_graph(self):
        """Should handle diamond dependency pattern."""
        plan = {
            "stages": [
                {"stage_id": "A", "dependencies": []},
                {"stage_id": "B", "dependencies": ["A"]},
                {"stage_id": "C", "dependencies": ["A"]},
                {"stage_id": "D", "dependencies": ["B", "C"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "A")
        
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 3, "Should find exactly 3 dependents in diamond pattern"
        assert set(result) == {"B", "C", "D"}, "Should find all dependents in diamond pattern"
        # Verify D is included even though it depends on both B and C
        assert "D" in result, "Stage D should be included as it depends on A transitively"

    def test_handles_empty_stages_list(self):
        """Should handle plan with empty stages list."""
        plan = {"stages": []}
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list when stages list is empty"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_nonexistent_target_stage(self):
        """Should handle target stage that doesn't exist in plan."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "nonexistent_stage")
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list when target stage doesn't exist"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_dependencies_referencing_nonexistent_stages(self):
        """Should handle dependencies that reference stages not in plan."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0", "nonexistent"]},
                {"stage_id": "stage2", "dependencies": ["stage1"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        # stage1 depends on stage0, stage2 depends on stage1
        # nonexistent dependency should be ignored
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"stage1", "stage2"}, "Should find dependents ignoring nonexistent dependencies"
        assert len(result) == 2, "Should find exactly 2 dependents"

    def test_handles_circular_dependencies(self):
        """Should handle circular dependencies without infinite loop."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage1", "stage2"]},  # Self-reference
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert "stage1" in result, "Should include stage1"
        assert "stage2" in result, "Should include stage2"
        assert len(result) == 2, "Should find exactly 2 dependents (no infinite loop)"
        # Verify no duplicates despite circular reference
        assert len(result) == len(set(result)), "Result should not contain duplicates even with circular deps"

    def test_handles_multiple_dependencies_on_same_stage(self):
        """Should handle stage with multiple dependencies on same target."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0", "stage0"]},  # Duplicate dependency
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"stage1"}, "Should find dependent even with duplicate dependencies"
        assert len(result) == 1, "Should find exactly one dependent"

    def test_handles_stage_without_dependencies_key(self):
        """Should handle stage dict missing dependencies key."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1"},  # Missing dependencies key
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"stage2"}, "Should find dependents, ignoring stages without dependencies"
        assert len(result) == 1, "Should find exactly one dependent"

    def test_handles_stage_with_none_dependencies(self):
        """Should handle stage with dependencies=None."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": None},
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        # This might raise TypeError if implementation doesn't handle None
        # If it does raise, that's a bug in the component, not the test
        try:
            result = _get_dependent_stages(plan, "stage0")
            assert isinstance(result, list), "Result should be a list"
            assert set(result) == {"stage2"}, "Should find dependents, handling None dependencies"
        except TypeError:
            # This is a bug in the component - the test correctly identifies it
            pytest.fail("Component should handle None dependencies gracefully, not raise TypeError")

    def test_handles_stage_with_empty_string_dependencies(self):
        """Should handle stage with dependencies containing empty strings."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0", ""]},
                {"stage_id": "stage2", "dependencies": ["stage1"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"stage1", "stage2"}, "Should find dependents, ignoring empty string dependencies"
        assert len(result) == 2, "Should find exactly 2 dependents"

    def test_handles_stage_missing_stage_id_key(self):
        """Should handle stage dict missing stage_id key."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"dependencies": ["stage0"]},  # Missing stage_id
                {"stage_id": "stage2", "dependencies": ["stage0"]},
            ]
        }
        # This should raise KeyError if implementation doesn't handle missing stage_id
        # If it does raise, that's a bug in the component, not the test
        try:
            result = _get_dependent_stages(plan, "stage0")
            assert isinstance(result, list), "Result should be a list"
            # If it doesn't raise, verify it handles gracefully
            assert "stage2" in result, "Should find valid dependents"
        except KeyError:
            # This is a bug in the component - the test correctly identifies it
            pytest.fail("Component should handle missing stage_id gracefully, not raise KeyError")

    def test_handles_none_stages_value(self):
        """Should handle plan with stages=None."""
        plan = {"stages": None}
        # plan.get("stages", []) should return None, not []
        # This might cause issues when iterating
        try:
            result = _get_dependent_stages(plan, "stage0")
            assert isinstance(result, list), "Result should be a list"
            assert result == [], "Should return empty list when stages is None"
        except TypeError:
            # This is a bug in the component - the test correctly identifies it
            pytest.fail("Component should handle None stages gracefully, not raise TypeError")

    def test_handles_single_stage_plan(self):
        """Should handle plan with only one stage."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert result == [], "Should return empty list for single stage with no dependents"
        assert len(result) == 0, "Result length should be 0"

    def test_handles_target_with_multiple_paths(self):
        """Should handle target that has multiple dependency paths to same stage."""
        plan = {
            "stages": [
                {"stage_id": "A", "dependencies": []},
                {"stage_id": "B", "dependencies": ["A"]},
                {"stage_id": "C", "dependencies": ["A", "B"]},  # C depends on A both directly and via B
            ]
        }
        result = _get_dependent_stages(plan, "A")
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"B", "C"}, "Should find all dependents"
        assert len(result) == 2, "Should find exactly 2 dependents (no duplicates)"
        assert result.count("C") == 1, "Should not duplicate C even though it has multiple paths to A"

    def test_handles_large_dependency_chain(self):
        """Should handle very long dependency chain."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
            ]
        }
        # Create a chain of 100 stages
        for i in range(1, 101):
            plan["stages"].append({
                "stage_id": f"stage{i}",
                "dependencies": [f"stage{i-1}"]
            })
        
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 100, "Should find all 100 transitive dependents"
        assert "stage1" in result, "Should include first dependent"
        assert "stage100" in result, "Should include last dependent"
        assert len(result) == len(set(result)), "Should not contain duplicates"

    def test_verifies_result_is_list_not_set(self):
        """Should verify that result is a list, not a set or other type."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage0")
        assert isinstance(result, list), "Result must be a list type"
        assert not isinstance(result, set), "Result must not be a set"
        assert not isinstance(result, tuple), "Result must not be a tuple"
        assert not isinstance(result, dict), "Result must not be a dict"

    def test_handles_unicode_stage_ids(self):
        """Should handle stage IDs with unicode characters."""
        plan = {
            "stages": [
                {"stage_id": "stage_α", "dependencies": []},
                {"stage_id": "stage_β", "dependencies": ["stage_α"]},
                {"stage_id": "stage_γ", "dependencies": ["stage_β"]},
            ]
        }
        result = _get_dependent_stages(plan, "stage_α")
        assert isinstance(result, list), "Result should be a list"
        assert set(result) == {"stage_β", "stage_γ"}, "Should handle unicode stage IDs correctly"
        assert len(result) == 2, "Should find exactly 2 dependents"
