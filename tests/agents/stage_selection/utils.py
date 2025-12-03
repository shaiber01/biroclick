"""
Shared helpers for stage selection tests.
"""


def create_stage(stage_id, stage_type="MATERIAL_VALIDATION", status="not_started", deps=None):
    return {
        "stage_id": stage_id,
        "stage_type": stage_type,
        "status": status,
        "dependencies": deps or [],
    }

