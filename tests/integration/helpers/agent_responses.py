"""Reusable mock responses for integration tests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def planner_response(
    stages: Optional[List[Dict[str, Any]]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Default planner response that mirrors a valid plan."""
    response = {
        "paper_id": overrides.pop("paper_id", "test"),
        "paper_domain": overrides.pop("paper_domain", "plasmonics"),
        "title": overrides.pop("title", "Test Plan"),
        "stages": stages
        or [
            {
                "stage_id": "stage_0",
                "stage_type": "MATERIAL_VALIDATION",
                "targets": ["Fig1"],
                "dependencies": [],
            }
        ],
        "targets": overrides.pop("targets", [{"figure_id": "Fig1"}]),
        "extracted_parameters": overrides.pop(
            "extracted_parameters", [{"name": "length", "value": 100}]
        ),
    }
    response.update(overrides)
    return response


def reporting_summary_response(**overrides: Any) -> Dict[str, Any]:
    """Default generate_report_node LLM response."""
    response = {
        "executive_summary": overrides.pop(
            "executive_summary",
            {"overall_assessment": [{"aspect": "Test", "status": "OK"}]},
        ),
        "conclusions": overrides.pop(
            "conclusions", {"main_physics_reproduced": True, "key_findings": ["Test finding"]}
        ),
        "paper_citation": overrides.pop(
            "paper_citation", {"title": "Test Paper", "authors": "Test Author"}
        ),
    }
    response.update(overrides)
    return response


def execution_verdict_response(verdict: str = "pass", **overrides: Any) -> Dict[str, Any]:
    """Helper for execution/physics sanity verdict payloads."""
    payload = {
        "verdict": verdict,
        "summary": overrides.pop("summary", "OK"),
    }
    payload.update(overrides)
    return payload

