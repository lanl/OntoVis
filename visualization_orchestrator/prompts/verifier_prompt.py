"""Final-goal verification prompt. Runs once after every planned task has succeeded, to
check the ORIGINAL user instruction end-to-end -- local per-task success does not imply
the whole request was satisfied (e.g. a compound "reveal internal structure AND show it
from behind" could have each half technically succeed while the combination doesn't read
as intended).
"""
import json
from typing import List


def _format_task_records(task_records) -> str:
    if not task_records:
        return "(no tasks were executed)"
    lines = []
    for record in task_records:
        lines.append(
            f"- {record.task_id} (agent={record.agent_id}): {record.status} -- {record.reason}"
        )
    return "\n".join(lines)


VERIFIER_INSTRUCTIONS = """You are the final verification stage of a visualization orchestrator. Every
individual planned task has already reported success, but local success does not guarantee
the user's complete original instruction was actually satisfied end-to-end -- check the
final rendered image and state against the ORIGINAL instruction and the final success
criteria, not against the individual task reports alone.

Respond with STRICT JSON ONLY -- no prose outside the JSON:
{
  "success": true | false,
  "confidence": 0.0-1.0,
  "satisfied_criteria": ["..."],
  "unsatisfied_criteria": ["..."],
  "diagnosis": "<concise explanation, especially if success is false>",
  "suggested_capabilities": ["<capability names that might resolve any unsatisfied criteria>"]
}"""


def build_verifier_prompt(
    user_instruction: str,
    state_summary: dict,
    task_records,
    final_success_criteria: List[str],
) -> str:
    return f"""{VERIFIER_INSTRUCTIONS}

ORIGINAL USER INSTRUCTION:
{user_instruction}

FINAL SUCCESS CRITERIA TO CHECK:
{json.dumps(final_success_criteria, indent=2)}

COMPLETED TASK RESULTS:
{_format_task_records(task_records)}

FINAL VISUALIZATION STATE:
{json.dumps(state_summary, indent=2)}

The attached image (if any) is the final rendered result.
"""
