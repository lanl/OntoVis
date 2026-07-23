"""Planner prompt template. Capability descriptions are always injected dynamically from
the live AgentRegistry (registry.capability_catalog()) -- this module never hard-codes
camera/isovalue capability names, so new specialists appear in the prompt automatically
just by registering them.
"""
import json
from typing import Dict, Optional

PLANNER_SYSTEM_INSTRUCTIONS = """You are the planning layer of a hierarchical, capability-based visualization orchestrator.
You never render, move a camera, or adjust a scalar value yourself -- you only decompose a
user's request into a small task graph that a deterministic executor will run.

Rules you must follow:
1. Interpret the user's desired FINAL visualization, not just their literal words.
2. Decompose it into the MINIMUM number of subtasks actually required -- do not add a task
   "just in case". Do not assume every request needs both a camera task and an isovalue/
   visibility task; many requests only need one.
3. Every task's "required_capability" MUST be exactly one of the capability names listed
   below. Never invent a capability name and never output an agent/implementation name
   (e.g. "camera_controller" or "isovalue_controller") anywhere in the plan.
4. Infer task dependencies from semantics and from what each capability needs to already be
   true (its precondition), not from a fixed template.
5. If the user's instruction contains explicit ordering language ("first", "then", "after",
   "before"), that ordering MUST be reflected in "dependencies" exactly -- it overrides any
   default assumption you would otherwise make.
6. If the current visualization state already satisfies part of the request (see CURRENT
   VISUALIZATION STATE below), omit the task that would redundantly redo it.
7. Every task needs clear, independently verifiable "success_criteria" (plain statements
   about the rendered result, not implementation steps) and the overall plan needs
   "final_success_criteria" covering the complete user instruction.
8. The task graph must be a DAG -- no cycles, no task depending on itself, no duplicate
   task_id values, no dependency on a task_id that doesn't exist in this plan.
9. State any assumptions you had to make in "assumptions" (e.g. "assuming the isovalue
   agent can expose the requested internal content").
10. Return ONLY the JSON object described below -- no prose, no markdown fences, no
    chain-of-thought.
"""

RESPONSE_SCHEMA_BLOCK = """Return only schema-valid JSON with exactly this structure:
{
  "interpreted_goal": "...",
  "tasks": [
    {
      "task_id": "...",
      "goal": "...",
      "required_capability": "...",
      "dependencies": ["..."],
      "success_criteria": ["..."],
      "constraints": [{"key": "...", "value": "..."}],
      "preferred_agent_id": null,
      "max_attempts": 1
    }
  ],
  "final_success_criteria": ["..."],
  "assumptions": ["..."]
}"""


def _format_capability_catalog(capability_catalog: Dict[str, dict]) -> str:
    if not capability_catalog:
        return "(no capabilities registered)"
    lines = []
    for name, info in sorted(capability_catalog.items()):
        examples = "; ".join(info.get("examples", [])[:4])
        lines.append(f"- {name}: {info['description']} (e.g. {examples})" if examples else f"- {name}: {info['description']}")
    return "\n".join(lines)


def _format_failure_block(failure_context: Optional[dict]) -> str:
    if not failure_context:
        return ""
    return (
        "\nA PREVIOUS ATTEMPT AT THIS REQUEST FAILED. Replan only what's necessary to address it "
        "-- do not restart tasks that already succeeded, and do not re-propose the exact same "
        "capability for the same unresolved goal unless no listed alternative exists:\n"
        f"  Failed task_id: {failure_context.get('task_id')}\n"
        f"  Capability that failed: {failure_context.get('required_capability')}\n"
        f"  Reason: {failure_context.get('reason')}\n"
        f"  Unsatisfied criteria: {failure_context.get('unsatisfied_criteria')}\n"
        f"  Suggested alternative capabilities: {failure_context.get('suggested_capabilities')}\n"
    )


def build_planner_prompt(
    user_instruction: str,
    state_summary: dict,
    capability_catalog: Dict[str, dict],
    history_summary: Optional[str] = None,
    failure_context: Optional[dict] = None,
) -> str:
    return f"""{PLANNER_SYSTEM_INSTRUCTIONS}
AVAILABLE CAPABILITIES (use these names verbatim as "required_capability" -- never an agent name):
{_format_capability_catalog(capability_catalog)}

CURRENT VISUALIZATION STATE:
{json.dumps(state_summary, indent=2)}

RECENT EXECUTION HISTORY:
{history_summary or "(none yet)"}
{_format_failure_block(failure_context)}
USER INSTRUCTION:
{user_instruction}

{RESPONSE_SCHEMA_BLOCK}
"""
