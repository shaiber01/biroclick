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
        
        assert set(result) == {"stage1", "stage2"}

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
        
        assert set(result) == {"stage1", "stage2", "stage3"}

    def test_returns_empty_for_leaf_stage(self):
        """Should return empty list for stage with no dependents."""
        plan = {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
            ]
        }
        
        result = _get_dependent_stages(plan, "stage1")
        
        assert result == []

    def test_handles_empty_plan(self):
        """Should handle empty plan gracefully."""
        result = _get_dependent_stages({}, "stage0")
        assert result == []

    def test_handles_missing_stages(self):
        """Should handle plan with no stages key."""
        plan = {"title": "Test Plan"}
        result = _get_dependent_stages(plan, "stage0")
        assert result == []

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
        
        assert set(result) == {"B", "C", "D"}
