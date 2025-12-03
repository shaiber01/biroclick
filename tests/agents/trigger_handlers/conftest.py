import pytest


@pytest.fixture(name="mock_state")
def trigger_state_fixture():
    """Basic state fixture with valid plan structure for trigger handler tests."""
    return {
        "paper_text": "Short paper text",
        "pending_validated_materials": ["mat1", "mat2"],
        # Plan with proper list structure (not dict)
        "plan": {
            "stages": [
                {"stage_id": "stage0", "dependencies": []},
                {"stage_id": "stage1", "dependencies": ["stage0"]},
                {"stage_id": "stage2", "dependencies": ["stage1"]},
            ]
        },
        "backtrack_decision": {"target_stage_id": "stage0"},
    }


@pytest.fixture(name="mock_result")
def trigger_result_fixture():
    """Empty result dict fixture for trigger handler tests."""
    return {}


@pytest.fixture(name="complex_dependency_state")
def complex_dependency_state_fixture():
    """State with diamond dependency structure for testing transitive dependencies."""
    return {
        "paper_text": "Test paper",
        "plan": {
            "stages": [
                {"stage_id": "base", "dependencies": []},
                {"stage_id": "left", "dependencies": ["base"]},
                {"stage_id": "right", "dependencies": ["base"]},
                {"stage_id": "merge", "dependencies": ["left", "right"]},
                {"stage_id": "final", "dependencies": ["merge"]},
            ]
        },
        "backtrack_decision": {"target_stage_id": "base"},
    }

