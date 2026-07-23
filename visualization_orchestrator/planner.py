"""Planner LLM: turns a user instruction + current state + registered capabilities into a
validated VisualizationPlan. Never executes anything itself -- see executor.py for that.
"""
from typing import List, Optional

from camera_reasoning.chatgpt_client import ask_chatgpt
from pydantic import ValidationError

from .json_utils import extract_json_object
from .models import VisualizationPlan
from .plan_validator import PlanValidationError, validate_plan
from .prompts.planner_prompt import build_planner_prompt
from .registry import AgentRegistry
from .state import VisualizationState, state_summary


class PlannerError(RuntimeError):
    """Raised when the planner LLM cannot produce a schema-valid, semantically valid plan
    within its retry budget."""


class Planner:
    def __init__(self, model: Optional[str] = None, max_attempts: int = 2):
        self.model = model
        self.max_attempts = max_attempts

    def plan(
        self,
        user_instruction: str,
        state: VisualizationState,
        registry: AgentRegistry,
        history_summary: Optional[str] = None,
        failure_context: Optional[dict] = None,
    ) -> VisualizationPlan:
        catalog = registry.capability_catalog()
        known_capabilities = set(catalog)

        rejection_reasons: List[str] = []
        for attempt in range(1, self.max_attempts + 1):
            prompt = build_planner_prompt(
                user_instruction=user_instruction,
                state_summary=state_summary(state),
                capability_catalog=catalog,
                history_summary=history_summary,
                failure_context=failure_context,
            )
            if rejection_reasons:
                prompt += (
                    "\nYour previous response was REJECTED for these reasons -- fix them and "
                    "return a corrected plan:\n" + "\n".join(f"- {r}" for r in rejection_reasons)
                )

            raw_response = ask_chatgpt(
                prompt=prompt, screenshot_path=state.rendered_image_path, model=self.model
            )

            parsed = extract_json_object(raw_response)
            if parsed is None:
                rejection_reasons = ["Response was not valid JSON."]
                continue

            try:
                plan = VisualizationPlan(**parsed)
            except ValidationError as exc:
                rejection_reasons = [str(exc)]
                continue

            try:
                validate_plan(plan, known_capabilities)
            except PlanValidationError as exc:
                rejection_reasons = exc.errors
                continue

            print(f"[Planner] Interpreted goal: {plan.interpreted_goal}")
            print(f"[Planner] Created {len(plan.tasks)} task(s): {[t.task_id for t in plan.tasks]}")
            return plan

        raise PlannerError(
            f"Planner failed to produce a valid plan after {self.max_attempts} attempt(s): "
            + "; ".join(rejection_reasons)
        )
