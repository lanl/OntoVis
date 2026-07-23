"""Thin wrapper around Planner used specifically for replanning after a task failure or a
failed final verification -- kept as its own module so the executor's failure-handling
path reads as "hand off to the replanner" rather than reaching back into planning
internals directly.
"""
from typing import Optional

from .models import VisualizationPlan
from .planner import Planner
from .registry import AgentRegistry
from .state import VisualizationState


class Replanner:
    def __init__(self, planner: Planner):
        self.planner = planner

    def replan(
        self,
        user_instruction: str,
        state: VisualizationState,
        registry: AgentRegistry,
        history_summary: Optional[str],
        failure_context: dict,
    ) -> VisualizationPlan:
        print(f"[Replanner] Requesting replan. Reason: {failure_context.get('reason')}")
        return self.planner.plan(
            user_instruction=user_instruction,
            state=state,
            registry=registry,
            history_summary=history_summary,
            failure_context=failure_context,
        )
