"""Adapter wrapping the existing camera agent (camera_reasoning.visual_rollout_agent) behind
the VisualizationSpecialist interface.

This does not reimplement camera reasoning: it syncs the shared VisualizationState's camera
fields into an existing, already-`.initialize()`d CameraReasoningSession, delegates the
entire render -> evaluate -> act -> repeat loop to
run_visual_rollout_alignment_loop (unmodified), and then reads the resulting camera state
back out into a state_patch restricted to the fields this specialist owns.
"""
from typing import Dict, List, Optional

from camera_reasoning.camera_state import get_camera_state, set_camera_state
from camera_reasoning.session import CameraReasoningSession
from camera_reasoning.visual_rollout_agent import run_visual_rollout_alignment_loop

from ..capabilities import AgentCapability, AgentSpec
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

DEFAULT_MAX_ITERATIONS = 8

CAMERA_AGENT_SPEC = AgentSpec(
    agent_id="camera_controller",
    description=(
        "Iteratively changes the camera until a requested viewpoint or framing goal is reached."
    ),
    capabilities=[
        AgentCapability(
            name="show_object_from_direction",
            description="Move the camera to an anatomical or object-relative viewing direction.",
            examples=["show from behind", "show from the left", "posterior view", "superior view"],
        ),
        AgentCapability(
            name="adjust_viewpoint",
            description="Change azimuth, elevation, roll, or general viewpoint.",
            examples=["rotate left", "rotate to the right", "look slightly downward"],
        ),
        AgentCapability(
            name="center_target",
            description="Center a visible target in the image.",
            examples=["center the skull"],
        ),
        AgentCapability(
            name="zoom_target",
            description="Adjust camera zoom around a target.",
            examples=["zoom in", "zoom into the left orbit"],
        ),
    ],
    owned_state_fields={
        "camera_position", "focal_point", "view_up", "zoom",
        "current_view_description", "rendered_image_path",
    },
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["changes viewpoint"],
)


class CameraSpecialist(VisualizationSpecialist):
    agent_id = "camera_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        default_max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ):
        self.session = session
        self.model = model
        self.default_max_iterations = default_max_iterations

    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        self._sync_session_camera(state)

        max_iterations = int(constraints.get("max_iterations", self.default_max_iterations))
        target_description = self._build_target_description(goal, success_criteria)

        original_target_description = self.session.target_description
        self.session.target_description = target_description
        try:
            loop_result = run_visual_rollout_alignment_loop(
                self.session,
                max_iterations=max_iterations,
                model=self.model,
                action_subset=None,
            )
        finally:
            self.session.target_description = original_target_description

        applied_actions = loop_result["applied_actions"]
        iterations_used = len(applied_actions)
        goal_satisfied = bool(applied_actions) and applied_actions[-1] == "STOP"

        camera_state = get_camera_state(self.session._renderer.GetActiveCamera())
        image_path = self.session.render_and_save()

        state_patch = {
            "camera_position": tuple(camera_state["position"]),
            "focal_point": tuple(camera_state["focal_point"]),
            "view_up": tuple(camera_state["view_up"]),
            "current_view_description": target_description,
            "rendered_image_path": image_path,
        }

        if goal_satisfied:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.8,
                reason=f"Camera loop converged (STOP) after {iterations_used} iteration(s).",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=iterations_used,
                artifacts=[image_path],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if applied_actions else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.4,
            reason=(
                f"Camera loop used all {max_iterations} iteration(s) without the model "
                "selecting STOP."
            ),
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=["adjust_viewpoint"],
            failure_type="iteration_limit_reached",
            iterations_used=iterations_used,
            artifacts=[image_path],
        )

    def _sync_session_camera(self, state: VisualizationState) -> None:
        camera = self.session._renderer.GetActiveCamera()
        current = get_camera_state(camera)
        current["position"] = list(state.camera_position)
        current["focal_point"] = list(state.focal_point)
        current["view_up"] = list(state.view_up)
        set_camera_state(camera, current)
        self.session._renderer.ResetCameraClippingRange()

    @staticmethod
    def _build_target_description(goal: str, success_criteria: List[str]) -> str:
        criteria_block = "; ".join(success_criteria)
        return f"{goal} Success criteria: {criteria_block}" if criteria_block else goal
