"""Isovalue specialist: owns its own render -> evaluate -> adjust -> repeat loop, mirroring
the camera agent's internal-loop shape (camera_reasoning.visual_rollout_agent) but for the
scalar isovalue threshold instead of the camera. No autonomous isovalue agent existed in
camera_reasoning/ before this -- the loop lives here rather than as a new top-level package,
since it is small and specific to this one adapter.

Operates on an existing, already-`.initialize()`d CameraReasoningSession in isosurface mode
(use_volume_rendering=False), via CameraReasoningSession.set_isovalue(), which rebuilds only
the isosurface actor in place and leaves the session's camera/renderer untouched -- so a
camera task that ran before or after this one composes onto the same render correctly.
"""
from typing import Dict, List, Optional

from camera_reasoning.chatgpt_client import ask_chatgpt
from camera_reasoning.session import CameraReasoningSession

from ..capabilities import AgentCapability, AgentSpec
from ..json_utils import extract_json_object
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 255.0

ISOVALUE_AGENT_SPEC = AgentSpec(
    agent_id="isovalue_controller",
    description=(
        "Iteratively adjusts a scalar isovalue until the desired structure or surface "
        "appearance is reached."
    ),
    capabilities=[
        AgentCapability(
            name="extract_surface",
            description="Find an isovalue that extracts a requested surface.",
            examples=["show the skull", "extract bone"],
        ),
        AgentCapability(
            name="reveal_internal_structure",
            description="Adjust the isovalue to expose a requested internal structure when possible.",
            examples=["show the inside of the head"],
        ),
        AgentCapability(
            name="reduce_surface_noise",
            description="Adjust the isovalue to suppress noisy or fragmented surfaces.",
            examples=["reduce skull noise", "make the surface denser"],
        ),
        AgentCapability(
            name="emphasize_material",
            description="Adjust the isovalue toward a requested material or density range.",
            examples=["emphasize bone"],
        ),
    ],
    owned_state_fields={"isovalue", "rendered_image_path"},
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["changes extracted geometry"],
)

ISOVALUE_PROMPT_TEMPLATE = """You are iteratively adjusting a scalar isovalue threshold used to extract an
isosurface from a 3D volume, by visually evaluating each rendered result.

Goal: {goal}

Success criteria this adjustment must satisfy:
{success_criteria_block}

Current isovalue: {current_isovalue} (valid range: {min_value}-{max_value})

{history_block}

Look at the attached rendered image (the CURRENT isosurface at the isovalue above) and decide:
- Does it already satisfy the goal and every success criterion listed above?
- If not, should the isovalue increase (extracts denser/higher-intensity material, typically
  removing surrounding softer/noisier structure) or decrease (includes more/lower-intensity
  material, which can reveal internal or fainter structure but may add noise)?

Respond with STRICT JSON ONLY, no prose outside the JSON:
{{
  "observation": "<what is visible in the current render>",
  "goal_satisfied": true | false,
  "new_isovalue": <number, or null -- null only if goal_satisfied is true>,
  "reasoning": "<why this isovalue does or does not satisfy the goal>"
}}
"""


def _format_history_block(history: List[dict]) -> str:
    if not history:
        return "History: (first iteration, no prior attempts yet)"
    lines = [
        f"  {i}. isovalue={h['isovalue']} goal_satisfied={h.get('goal_satisfied')} "
        f"reasoning={h.get('reasoning')}"
        for i, h in enumerate(history, 1)
    ]
    return "History of prior attempts this task:\n" + "\n".join(lines)


def run_isovalue_alignment_loop(
    session: CameraReasoningSession,
    goal: str,
    success_criteria: List[str],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    model: Optional[str] = None,
    min_value: float = DEFAULT_MIN_VALUE,
    max_value: float = DEFAULT_MAX_VALUE,
) -> dict:
    """Run the isovalue render/evaluate/adjust loop against an already-initialized session
    in isosurface mode. Returns {"converged": bool, "history": [...], "final_isovalue": float,
    "final_image_path": str}."""
    history: List[dict] = []

    for _ in range(max_iterations):
        image_path = session.render_and_save()
        prompt = ISOVALUE_PROMPT_TEMPLATE.format(
            goal=goal,
            success_criteria_block="\n".join(f"- {c}" for c in success_criteria) or "- (none specified)",
            current_isovalue=session.isovalue,
            min_value=min_value,
            max_value=max_value,
            history_block=_format_history_block(history),
        )
        response_text = ask_chatgpt(prompt=prompt, screenshot_path=image_path, model=model)
        parsed = extract_json_object(response_text)

        if parsed is None or "goal_satisfied" not in parsed:
            history.append({
                "isovalue": session.isovalue, "goal_satisfied": False,
                "reasoning": "Model response was not valid JSON; stopping.",
            })
            break

        history.append({
            "isovalue": session.isovalue,
            "observation": parsed.get("observation"),
            "goal_satisfied": bool(parsed.get("goal_satisfied")),
            "reasoning": parsed.get("reasoning"),
        })

        if parsed.get("goal_satisfied"):
            return {
                "converged": True, "history": history,
                "final_isovalue": session.isovalue, "final_image_path": image_path,
            }

        new_value = parsed.get("new_isovalue")
        if not isinstance(new_value, (int, float)) or isinstance(new_value, bool):
            break
        new_value = max(min_value, min(max_value, float(new_value)))
        if new_value == session.isovalue:
            break  # model proposed no change -- avoid spinning without progress
        session.set_isovalue(new_value)

    final_image_path = session.render_and_save()
    return {
        "converged": False, "history": history,
        "final_isovalue": session.isovalue, "final_image_path": final_image_path,
    }


class IsovalueSpecialist(VisualizationSpecialist):
    agent_id = "isovalue_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        default_max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_value: float = DEFAULT_MIN_VALUE,
        max_value: float = DEFAULT_MAX_VALUE,
    ):
        self.session = session
        self.model = model
        self.default_max_iterations = default_max_iterations
        self.min_value = min_value
        self.max_value = max_value

    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        if state.isovalue is not None:
            self.session.set_isovalue(state.isovalue)

        max_iterations = int(constraints.get("max_iterations", self.default_max_iterations))
        loop_result = run_isovalue_alignment_loop(
            self.session,
            goal=goal,
            success_criteria=success_criteria,
            max_iterations=max_iterations,
            model=self.model,
            min_value=self.min_value,
            max_value=self.max_value,
        )

        state_patch = {
            "isovalue": loop_result["final_isovalue"],
            "rendered_image_path": loop_result["final_image_path"],
        }
        iterations_used = len(loop_result["history"])

        if loop_result["converged"]:
            last_reasoning = loop_result["history"][-1]["reasoning"] if loop_result["history"] else ""
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.75,
                reason=f"Isovalue loop converged after {iterations_used} iteration(s): {last_reasoning}",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=iterations_used,
                artifacts=[loop_result["final_image_path"]],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if iterations_used else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.3,
            reason=(
                "A single isovalue could not be found that satisfies the goal within "
                f"{max_iterations} iteration(s)."
            ),
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=[
                "adjust_opacity_transfer_function",
                "apply_clipping",
                "segment_target_structure",
            ],
            failure_type="capability_insufficient",
            iterations_used=iterations_used,
            artifacts=[loop_result["final_image_path"]],
        )
