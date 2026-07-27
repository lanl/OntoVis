"""Orientation specialist: checks and corrects the camera's ROLL (view_up), independent of
camera POSITION.

`camera_reasoning`'s blind visual-rollout pipeline (see camera_adapter.py) only searches
positional movements (azimuth/elevation -- which SIDE of the object is shown); ROLL is not
even in its candidate action subset (BLIND_CANDIDATE_ACTION_SUBSET in
blind_visual_rollout_agent.py), so a correctly-positioned view can still come out tilted or
upside-down with nothing in that pipeline ever attempting to fix it. Whether an image looks
"upright" is a much simpler visual question than "which side is this", so this specialist
answers it directly: one plain LLM call per iteration, no blind candidate rendering, no
reference-bank matching, no three-pass pipeline.

Runs strictly AFTER whatever positioned the camera (a separate task in the plan, not baked
into camera_controller's own loop) -- it only ever adjusts roll, never re-litigates which
side of the object is being shown.
"""
from typing import Callable, Dict, List, Optional

from camera_reasoning.camera_state import get_camera_state
from camera_reasoning.chatgpt_client import ask_chatgpt
from camera_reasoning.session import CameraReasoningSession

from ..capabilities import AgentCapability, AgentSpec
from ..json_utils import extract_json_object
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

DEFAULT_MAX_ITERATIONS = 3

ORIENTATION_AGENT_SPEC = AgentSpec(
    agent_id="orientation_controller",
    description=(
        "Checks and corrects the camera's roll so the rendered image appears upright, "
        "independent of which side of the object is being shown. A single direct LLM "
        "judgment per iteration (not the blind candidate-comparison pipeline) -- 'is this "
        "image tilted?' is a much simpler visual question than 'which side is this?'."
    ),
    capabilities=[
        AgentCapability(
            name="correct_view_orientation",
            description="Adjust camera roll so the rendered image is upright, without changing which side is shown.",
            examples=["fix the tilted view", "make sure the image is right-side up", "the view looks rotated, straighten it"],
        ),
    ],
    owned_state_fields={"view_up", "rendered_image_path"},
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["adjusts camera roll"],
)

ORIENTATION_CHECK_PROMPT_TEMPLATE = """You are checking whether a rendered 3D view is correctly ORIENTED -- i.e. whether the
image appears upright, not tilted, rolled, or upside-down around the viewing axis. This is
entirely independent of WHICH SIDE of the object is being shown (that has already been
handled separately) -- only judge the roll/tilt of THIS image around its own center.

Goal: {goal}

Success criteria this correction must satisfy:
{success_criteria_block}

Look at the attached image and decide:
- Does it already look correctly oriented (upright, not visibly tilted or rolled)?
- If not, how many degrees would the camera need to roll, and in which direction
  (positive = clockwise, negative = counterclockwise, as seen on screen), to make it
  look upright?

Respond with STRICT JSON ONLY, no prose outside the JSON:
{{
  "observation": "<what you see -- is anything about the image's tilt/rotation off?>",
  "is_correctly_oriented": true | false,
  "roll_degrees": <number, positive = clockwise, negative = counterclockwise -- 0 or null if already correct>,
  "reasoning": "<why this is or isn't already correctly oriented>"
}}
"""


class OrientationSpecialist(VisualizationSpecialist):
    agent_id = "orientation_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        default_max_iterations: int = DEFAULT_MAX_ITERATIONS,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ):
        self.session = session
        self.model = model
        self.default_max_iterations = default_max_iterations
        self.on_iteration = on_iteration

    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        max_iterations = int(constraints.get("max_iterations", self.default_max_iterations))
        camera = self.session._renderer.GetActiveCamera()

        history: List[dict] = []
        for i in range(max_iterations):
            image_path = self.session.render_and_save()
            prompt = ORIENTATION_CHECK_PROMPT_TEMPLATE.format(
                goal=goal,
                success_criteria_block="\n".join(f"- {c}" for c in success_criteria) or "- (none specified)",
            )
            response_text = ask_chatgpt(prompt=prompt, screenshot_path=image_path, model=self.model)
            parsed = extract_json_object(response_text)

            if parsed is None or "is_correctly_oriented" not in parsed:
                entry = {"iteration": i, "image_path": image_path, "observation": None,
                          "is_correctly_oriented": False, "roll_degrees": None,
                          "reasoning": "Model response was not valid JSON; stopping."}
                history.append(entry)
                if self.on_iteration:
                    self._handle_iteration(entry)
                break

            entry = {
                "iteration": i, "image_path": image_path,
                "observation": parsed.get("observation"),
                "is_correctly_oriented": bool(parsed.get("is_correctly_oriented")),
                "roll_degrees": parsed.get("roll_degrees"),
                "reasoning": parsed.get("reasoning"),
            }
            history.append(entry)
            if self.on_iteration:
                self._handle_iteration(entry)

            if entry["is_correctly_oriented"]:
                break

            roll_degrees = entry["roll_degrees"]
            if not isinstance(roll_degrees, (int, float)) or isinstance(roll_degrees, bool) or roll_degrees == 0:
                break  # nothing actionable proposed -- avoid spinning without progress

            camera.Roll(float(roll_degrees))
            camera.OrthogonalizeViewUp()
            self.session._renderer.ResetCameraClippingRange()

        final_image_path = self.session.render_and_save()
        camera_state = get_camera_state(camera)
        state_patch = {
            "view_up": tuple(camera_state["view_up"]),
            "rendered_image_path": final_image_path,
        }
        iterations_used = len(history)
        converged = bool(history) and history[-1]["is_correctly_oriented"]

        if converged:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.8,
                reason=f"Orientation confirmed correct after {iterations_used} iteration(s): {history[-1]['reasoning']}",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=iterations_used,
                artifacts=[final_image_path],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if iterations_used else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.3,
            reason=f"Could not confirm correct orientation within {max_iterations} iteration(s).",
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=[],
            iterations_used=iterations_used,
            artifacts=[final_image_path],
        )

    def _handle_iteration(self, entry: dict) -> None:
        """Normalize one iteration into the shared on_iteration shape (see
        CameraSpecialist._handle_camera_iteration / IsovalueSpecialist._handle_band_result
        for the other agents' versions of this same normalized dict)."""
        self.on_iteration({
            "agent_id": self.agent_id,
            "iteration": entry["iteration"],
            "current_image_path": entry["image_path"],
            "candidates": [],
            "selected_label": None,
            "reasoning": entry.get("reasoning") or "",
            "extra": {
                "is_correctly_oriented": entry.get("is_correctly_oriented"),
                "roll_degrees": entry.get("roll_degrees"),
                "observation": entry.get("observation"),
            },
        })
