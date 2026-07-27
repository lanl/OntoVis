"""Final-goal verification prompt. Runs once after every planned task has succeeded, to
check the ORIGINAL user instruction end-to-end -- local per-task success does not imply
the whole request was satisfied (e.g. a compound "reveal internal structure AND show it
from behind" could have each half technically succeed while the combination doesn't read
as intended).

Deliberately a pure image-vs-goal comparison: no state summary or task-record text is
included (see verifier.py's module docstring for why -- numeric camera state invited the
wrong kind of reasoning). The model must judge success from what the attached image
actually shows.
"""
from typing import List

VERIFIER_INSTRUCTIONS = """You are the final verification stage of a visualization orchestrator. Every
individual planned task has already reported success, but local success does not guarantee
the user's complete original instruction was actually satisfied end-to-end -- judge ONLY
from the attached rendered image against the ORIGINAL instruction and the final success
criteria below. Do not assume anything about camera position, orientation, or state that
isn't visibly apparent in the image itself.

Respond with STRICT JSON ONLY -- no prose outside the JSON:
{
  "success": true | false,
  "confidence": 0.0-1.0,
  "satisfied_criteria": ["..."],
  "unsatisfied_criteria": ["..."],
  "diagnosis": "<concise explanation, especially if success is false>",
  "suggested_capabilities": ["<capability names that might resolve any unsatisfied criteria>"]
}"""


def build_verifier_prompt(user_instruction: str, final_success_criteria: List[str]) -> str:
    return f"""{VERIFIER_INSTRUCTIONS}

ORIGINAL USER INSTRUCTION:
{user_instruction}

FINAL SUCCESS CRITERIA TO CHECK:
{chr(10).join(f"- {c}" for c in final_success_criteria) or "(none specified)"}

The attached image is the final rendered result -- judge success from it directly.
"""
