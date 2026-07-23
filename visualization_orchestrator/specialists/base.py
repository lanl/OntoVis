"""Common interface every specialist (existing or future) must implement.

The executor only ever calls `run_until_complete` -- it never controls a specialist's
individual internal steps (e.g. individual camera rotations or isovalue increments).
That internal render -> evaluate -> act -> repeat loop belongs entirely to the specialist.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from ..models import AgentExecutionResult
from ..state import VisualizationState


class VisualizationSpecialist(ABC):
    agent_id: str

    @abstractmethod
    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        raise NotImplementedError
