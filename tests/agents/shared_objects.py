"""
Reusable helpers and constants shared across the agent test suites.
"""

from __future__ import annotations

import json


class NonSerializable:
    """Helper object used to verify serialization fallbacks in prompts."""

    def __str__(self) -> str:
        return "NON_SERIALIZABLE_OBJECT"


LONG_FALLBACK_PAYLOAD = {
    "something": "else",
    "details": "x" * 80,
}

LONG_FALLBACK_JSON = json.dumps(LONG_FALLBACK_PAYLOAD, indent=2)


CLI_MATERIAL_CHECKPOINT_PROMPT = [
    "Material checkpoint: APPROVE or REJECT?",
    "If rejecting, specify CHANGE_MATERIAL or CHANGE_DATABASE.",
]

