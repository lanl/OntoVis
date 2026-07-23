"""Best-effort JSON extraction from LLM responses, shared by the planner, verifier, and
the isovalue specialist's internal loop (mirrors the pattern already used in
camera_reasoning/view_description_generator.py, centralized here as a small library
function instead of duplicated again)."""
import json
import re
from typing import Optional

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[dict]:
    """Return the first JSON object found in `text` (bare, fenced, or embedded in prose),
    or None if nothing parses."""
    if not text:
        return None

    fenced = _FENCED_JSON_RE.search(text)
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    brace_match = _BRACE_RE.search(text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None
    return None
