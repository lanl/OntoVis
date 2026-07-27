"""One direct, non-LLM specialist for instructions that already give exact, unambiguous
parameter values -- e.g. "set isovalue to 60", "reset the camera", "zoom in", "roll 30
degrees clockwise", "rotate left 60 degrees". Applies whichever recognized parameters are
given with a single deterministic CameraReasoningSession/VTK call apiece and no LLM call
at all, instead of going through the vision-LLM specialists (camera_controller/
isovalue_controller/orientation_controller), which exist for genuinely ambiguous or
goal-based requests ("show me the skull", "make it look upright") that need a
render-and-judge loop to actually resolve.

Deliberately ONE capability/specialist rather than one per parameter (isovalue, roll,
azimuth, elevation, ...): the set of possible exact values is unbounded (any float
degree, any isovalue number), so a bespoke module per parameter never actually stops
growing -- adding "rotate left 60 degrees" support after already having a fixed-magnitude
apply_exact_camera_action just proved that. Supporting a new exact parameter here means
adding one entry to _PARAMETER_HANDLERS below, not a new AgentSpec/class/registration.

Capability descriptions are injected into the planner prompt automatically (see
prompts/planner_prompt.py's module docstring), so the planner picks this over the
vision-LLM capabilities purely from reading the capability description/examples below --
no separate routing logic lives here or in the executor.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

from camera_reasoning.camera_actions import VALID_ACTIONS, apply_action
from camera_reasoning.camera_state import camera_distance, get_camera_state, set_camera_state
from camera_reasoning.session import MIN_CAMERA_DISTANCE, CameraReasoningSession

from ..capabilities import AgentCapability, AgentSpec
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

# Every fixed camera movement EXCEPT STOP/UNDO_LAST, which aren't movements to apply here.
DIRECT_CAMERA_ACTIONS = sorted(VALID_ACTIONS - {"STOP", "UNDO_LAST"})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ------------------------------------------------------------------
# Parameter handlers -- each applies ONE recognized constraint key's value directly via
# the session/VTK call it corresponds to. Returns (state_patch_fragment, error_message);
# exactly one of the two is None. Camera-affecting handlers don't call
# OrthogonalizeViewUp/ResetCameraClippingRange themselves -- that happens once, centrally,
# after every requested handler has run (see ApplyExactParametersSpecialist).
# ------------------------------------------------------------------

def _handle_isovalue(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not _is_number(value):
        return None, f"'isovalue' must be a number; got {value!r}."
    session.set_isovalue(float(value))
    return {"isovalue": float(value), "transfer_function": None}, None


def _handle_reset_camera(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if value is not True:
        return None, f"'reset_camera' must be true; got {value!r}."
    session.reset_camera()
    return {}, None


def _handle_action(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not isinstance(value, str) or value not in DIRECT_CAMERA_ACTIONS:
        return None, f"'action' must be one of {DIRECT_CAMERA_ACTIONS}; got {value!r}."
    camera = session._renderer.GetActiveCamera()
    apply_action(value, camera, session._renderer)
    return {}, None


def _handle_roll_degrees(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not _is_number(value):
        return None, f"'roll_degrees' must be a number; got {value!r}."
    session._renderer.GetActiveCamera().Roll(float(value))
    return {}, None


def _handle_azimuth_degrees(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not _is_number(value):
        return None, f"'azimuth_degrees' must be a number; got {value!r}."
    session._renderer.GetActiveCamera().Azimuth(float(value))
    return {}, None


def _handle_elevation_degrees(session: CameraReasoningSession, value: Any) -> Tuple[Optional[dict], Optional[str]]:
    if not _is_number(value):
        return None, f"'elevation_degrees' must be a number; got {value!r}."
    session._renderer.GetActiveCamera().Elevation(float(value))
    return {}, None


# Applied in this fixed order regardless of the order constraints were given in, so
# combining several in one task (e.g. isovalue + a rotation) behaves predictably.
_PARAMETER_HANDLERS: Dict[str, Callable[[CameraReasoningSession, Any], Tuple[Optional[dict], Optional[str]]]] = {
    "isovalue": _handle_isovalue,
    "reset_camera": _handle_reset_camera,
    "action": _handle_action,
    "roll_degrees": _handle_roll_degrees,
    "azimuth_degrees": _handle_azimuth_degrees,
    "elevation_degrees": _handle_elevation_degrees,
}


def _camera_state_patch(session: CameraReasoningSession, image_path: str) -> dict:
    camera_state = get_camera_state(session._renderer.GetActiveCamera())
    return {
        "camera_position": tuple(camera_state["position"]),
        "focal_point": tuple(camera_state["focal_point"]),
        "view_up": tuple(camera_state["view_up"]),
        "rendered_image_path": image_path,
    }


def _rejected(agent_id: str, reason: str, success_criteria: List[str], suggested: List[str]) -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id=agent_id,
        status="failed",
        goal_satisfied=False,
        state_patch={},
        confidence=0.0,
        reason=reason,
        satisfied_criteria=[],
        unsatisfied_criteria=list(success_criteria),
        suggested_capabilities=suggested,
        failure_type="capability_insufficient",
    )


_FALLBACK_CAPABILITIES = [
    "extract_surface", "reveal_internal_structure", "emphasize_material",
    "show_object_from_direction", "adjust_viewpoint", "correct_view_orientation",
]

APPLY_EXACT_PARAMETERS_SPEC = AgentSpec(
    agent_id="direct_specialist",
    description=(
        "Directly applies one or more exact, fully-specified visualization parameters the "
        "user explicitly stated, with no rendered-candidate comparison or LLM call."
    ),
    capabilities=[
        AgentCapability(
            name="apply_exact_parameters",
            description=(
                "Apply one or more EXACT parameter values the user explicitly stated -- no "
                "visual search, no goal-based judgment. Provide one task constraint per "
                "parameter (multiple may be combined in one task; all are applied together, "
                "in this fixed order regardless of how they're listed: isovalue -> "
                "reset_camera -> action -> roll_degrees -> azimuth_degrees -> "
                "elevation_degrees):\n"
                "  - {\"key\": \"isovalue\", \"value\": <number>} -- set the isosurface "
                "isovalue exactly (e.g. 'set isovalue to 60').\n"
                "  - {\"key\": \"reset_camera\", \"value\": true} -- reset the camera to fit "
                "the whole scene (e.g. 'reset the camera').\n"
                "  - {\"key\": \"action\", \"value\": \"<ACTION_NAME>\"} -- apply one fixed "
                "named camera movement. <ACTION_NAME> MUST be copied VERBATIM, exactly as "
                "spelled, from this list -- NEVER construct/invent a name by combining a "
                "direction with a degree number (there is no 'AZIMUTH_RIGHT_90'; COARSE "
                "means 90 degrees but the name itself never contains the digits): "
                + ", ".join(DIRECT_CAMERA_ACTIONS) + ". Use this key ONLY for wording with "
                "no explicit degree number, e.g. 'zoom in', 'pan up a little', 'rotate "
                "left' with no number (magnitude words 'a little'/'slightly' -> FINE, 'a "
                "lot'/no qualifier -> COARSE, 'halfway'/'somewhat' -> MEDIUM). If the "
                "instruction states an explicit degree number for a rotation (e.g. 'rotate "
                "right 90 degrees', 'tilt up 45 degrees'), do NOT use this key at all, even "
                "if that number happens to equal 90/45/15 -- use roll_degrees/"
                "azimuth_degrees/elevation_degrees below instead, since those take the "
                "number directly with no name-lookup step to get wrong.\n"
                "  - {\"key\": \"roll_degrees\", \"value\": <number, positive=clockwise>} -- "
                "roll the camera by an exact degree amount (e.g. 'roll 30 degrees "
                "clockwise', 'roll 90 degrees clockwise').\n"
                "  - {\"key\": \"azimuth_degrees\", \"value\": <number, positive=right, "
                "negative=left>} -- orbit the camera horizontally by an exact stated degree "
                "amount (e.g. 'rotate left 60 degrees', 'rotate right 90 degrees').\n"
                "  - {\"key\": \"elevation_degrees\", \"value\": <number, positive=up, "
                "negative=down>} -- orbit the camera vertically by an exact stated degree "
                "amount (e.g. 'look down 20 degrees', 'tilt up 45 degrees').\n"
                "Do NOT use this for a goal-based request with no explicit value, like 'show "
                "the bone', 'show it from behind', or 'make it upright' -- those need visual "
                "judgment to confirm the result; use extract_surface/"
                "reveal_internal_structure/emphasize_material/show_object_from_direction/"
                "adjust_viewpoint/correct_view_orientation instead."
            ),
            examples=[
                "set isovalue to 60", "reset the camera", "zoom in",
                "roll 30 degrees clockwise", "rotate left 60 degrees",
                "rotate right 90 degrees", "set isovalue to 40 and rotate left 30 degrees",
            ],
        ),
    ],
    owned_state_fields={
        "isovalue", "transfer_function", "camera_position", "focal_point", "view_up", "rendered_image_path",
    },
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["changes extracted geometry", "changes viewpoint"],
    deterministic=True,
)


class ApplyExactParametersSpecialist(VisualizationSpecialist):
    agent_id = "direct_specialist"

    def __init__(self, session: CameraReasoningSession):
        self.session = session

    def run_until_complete(
        self, goal: str, state: VisualizationState, constraints: Dict, success_criteria: List[str],
    ) -> AgentExecutionResult:
        recognized = [key for key in _PARAMETER_HANDLERS if key in constraints]
        if not recognized:
            return _rejected(
                self.agent_id,
                f"apply_exact_parameters requires at least one recognized constraint key "
                f"from {sorted(_PARAMETER_HANDLERS)}; got {sorted(constraints)}.",
                success_criteria,
                _FALLBACK_CAPABILITIES,
            )

        camera = self.session._renderer.GetActiveCamera()
        prev_camera_state = get_camera_state(camera)
        applied_patch: Dict[str, Any] = {}
        applied_summary: List[str] = []

        for key in recognized:
            value = constraints[key]
            patch, error = _PARAMETER_HANDLERS[key](self.session, value)
            if error is not None:
                set_camera_state(camera, prev_camera_state)
                self.session._renderer.ResetCameraClippingRange()
                return _rejected(
                    self.agent_id, f"apply_exact_parameters: {error}", success_criteria, _FALLBACK_CAPABILITIES,
                )
            applied_patch.update(patch)
            applied_summary.append(f"{key}={value}")

        camera.OrthogonalizeViewUp()
        self.session._renderer.ResetCameraClippingRange()

        if camera_distance(get_camera_state(camera)) < MIN_CAMERA_DISTANCE:
            set_camera_state(camera, prev_camera_state)
            self.session._renderer.ResetCameraClippingRange()
            return _rejected(
                self.agent_id,
                f"Applying {', '.join(applied_summary)} would move the camera too close to "
                "the focal point; reverted.",
                success_criteria,
                [],
            )

        image_path = self.session.render_and_save()
        state_patch = {**applied_patch, **_camera_state_patch(self.session, image_path)}
        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="success",
            goal_satisfied=True,
            state_patch=state_patch,
            confidence=1.0,
            reason=f"Applied {', '.join(applied_summary)} directly -- no visual search needed.",
            satisfied_criteria=list(success_criteria),
            unsatisfied_criteria=[],
            iterations_used=0,
            artifacts=[image_path],
        )
