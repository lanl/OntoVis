"""Shared test doubles for the visualization_orchestrator test suite. Not a test module
itself (no test_* functions), so it is not pytest-collected -- imported by the
test_orchestrator_*.py files instead.
"""
from typing import Dict, List, Optional

from visualization_orchestrator.capabilities import AgentCapability, AgentSpec
from visualization_orchestrator.models import AgentExecutionResult, FinalVerificationResult, VisualizationPlan
from visualization_orchestrator.specialists.base import VisualizationSpecialist
from visualization_orchestrator.state import VisualizationState


def make_state(**overrides) -> VisualizationState:
    defaults = dict(
        dataset_path="data/fake.raw",
        camera_position=(0.0, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        isovalue=40.0,
        rendered_image_path="output/fake_render.png",
    )
    defaults.update(overrides)
    return VisualizationState(**defaults)


def make_camera_spec(agent_id: str = "camera_controller", priority: int = 0, cost: float = 1.0) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id,
        description="Fake camera agent.",
        capabilities=[
            AgentCapability(name="show_object_from_direction", description="...", examples=["from behind"]),
            AgentCapability(name="adjust_viewpoint", description="...", examples=["rotate left"]),
        ],
        owned_state_fields={"camera_position", "focal_point", "view_up", "zoom", "rendered_image_path", "current_view_description"},
        required_state_fields={"dataset_path", "rendered_image_path"},
        priority=priority,
        cost=cost,
    )


def make_isovalue_spec(agent_id: str = "isovalue_controller", priority: int = 0, cost: float = 1.0) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id,
        description="Fake isovalue agent.",
        capabilities=[
            AgentCapability(name="reveal_internal_structure", description="...", examples=["show the inside"]),
            AgentCapability(name="reduce_surface_noise", description="...", examples=["reduce noise"]),
        ],
        owned_state_fields={"isovalue", "rendered_image_path"},
        required_state_fields={"dataset_path", "rendered_image_path"},
        priority=priority,
        cost=cost,
    )


class FakeSpecialist(VisualizationSpecialist):
    """Specialist double that returns a fixed sequence of AgentExecutionResults, one per
    call to run_until_complete (the last one repeats once the sequence is exhausted).
    Records every call's (goal, constraints, success_criteria) for assertions."""

    def __init__(self, agent_id: str, results: List[AgentExecutionResult]):
        self.agent_id = agent_id
        self._results = results
        self.calls: List[dict] = []

    def run_until_complete(self, goal, state, constraints, success_criteria) -> AgentExecutionResult:
        index = min(len(self.calls), len(self._results) - 1)
        self.calls.append({"goal": goal, "constraints": constraints, "success_criteria": success_criteria})
        return self._results[index]


def success_result(agent_id: str, patch: dict, criteria: Optional[List[str]] = None) -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id=agent_id, status="success", goal_satisfied=True, state_patch=patch,
        confidence=0.9, reason="fake success", satisfied_criteria=criteria or [], unsatisfied_criteria=[],
        iterations_used=1,
    )


def failure_result(agent_id: str, criteria: Optional[List[str]] = None, suggested: Optional[List[str]] = None) -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id=agent_id, status="failed", goal_satisfied=False, state_patch={},
        confidence=0.1, reason="fake failure", satisfied_criteria=[], unsatisfied_criteria=criteria or [],
        suggested_capabilities=suggested or [], failure_type="capability_insufficient", iterations_used=1,
    )


class FakePlanner:
    """Planner double returning a fixed sequence of plans, one per call to .plan()."""

    def __init__(self, plans: List[VisualizationPlan]):
        self._plans = plans
        self.calls: List[dict] = []

    def plan(self, user_instruction, state, registry, history_summary=None, failure_context=None) -> VisualizationPlan:
        index = min(len(self.calls), len(self._plans) - 1)
        self.calls.append({"user_instruction": user_instruction, "failure_context": failure_context})
        return self._plans[index]


class FakeVerifier:
    """Verifier double returning a fixed sequence of FinalVerificationResults."""

    def __init__(self, results: List[FinalVerificationResult]):
        self._results = results
        self.calls: List[dict] = []

    def verify(self, user_instruction, state, task_records, final_success_criteria) -> FinalVerificationResult:
        index = min(len(self.calls), len(self._results) - 1)
        self.calls.append({"user_instruction": user_instruction, "final_success_criteria": final_success_criteria})
        return self._results[index]
