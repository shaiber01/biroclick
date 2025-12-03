import pytest


@pytest.fixture(name="mock_state")
def trigger_state_fixture():
    return {
        "paper_text": "Short paper text",
        "pending_validated_materials": ["mat1", "mat2"],
        "plan": {"stages": {}},
        "backtrack_decision": {"target_stage_id": "stage0"},
    }


@pytest.fixture(name="mock_result")
def trigger_result_fixture():
    return {}

