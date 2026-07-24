"""Adapter wrapping the existing label-blind camera agent
(camera_reasoning.blind_visual_rollout_agent) behind the VisualizationSpecialist interface.

This does not reimplement camera reasoning: it syncs the shared VisualizationState's camera
fields into an existing, already-`.initialize()`d CameraReasoningSession, delegates the
entire three-pass render -> blind-diagnose -> deterministic-select -> magnitude -> act ->
repeat loop to run_blind_visual_rollout_alignment_loop (unmodified), and then reads the
resulting camera state back out into a state_patch restricted to the fields this specialist
owns.

Scope note: the blind loop's candidate set (BLIND_CANDIDATE_ACTION_SUBSET) is azimuth/
elevation framing only -- AZIMUTH_LEFT/RIGHT, ELEVATION_UP/DOWN (COARSE, with MEDIUM/FINE
reasoned about, never rendered), the two 180 degree actions, and STOP. It never proposes
ZOOM/PAN/ROLL/UNDO_LAST, unlike the plain camera_reasoning.visual_rollout_agent this adapter
used previously. CAMERA_AGENT_SPEC below only advertises the capabilities this loop can
actually satisfy -- no zoom_target/center_target capability is registered here.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from camera_reasoning.blind_visual_rollout_agent import run_blind_visual_rollout_alignment_loop
from camera_reasoning.camera_state import get_camera_state, set_camera_state
from camera_reasoning.session import CameraReasoningSession

from ..capabilities import AgentCapability, AgentSpec
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_STOP_SIMILARITY_THRESHOLD = 92
DEFAULT_STOP_IMPROVEMENT_MARGIN = 3
DEFAULT_MINIMUM_CONFIDENCE = 0.5

CAMERA_AGENT_SPEC = AgentSpec(
    agent_id="camera_controller",
    description=(
        "Iteratively changes the camera azimuth/elevation until a requested viewpoint or "
        "framing goal is reached, using label-blind candidate comparison (three-pass: "
        "visual diagnosis, deterministic selection, magnitude estimation)."
    ),
    capabilities=[
        AgentCapability(
            name="show_object_from_direction",
            description="Move the camera to an anatomical or object-relative viewing direction.",
            examples=["show from behind", "show from the left", "posterior view", "superior view"],
        ),
        AgentCapability(
            name="adjust_viewpoint",
            description="Change azimuth or elevation to a general viewpoint.",
            examples=["rotate left", "rotate to the right", "look slightly downward"],
        ),
    ],
    owned_state_fields={"camera_position", "focal_point", "view_up", "current_view_description", "rendered_image_path"},
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["changes viewpoint"],
)


def load_reference_bank(
    nodes_path: str,
    descriptions_path: str,
    base_dir: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load a node_id -> image_path / node_id -> description reference-view bank (the shape
    camera_reasoning.blind_visual_rollout_agent expects for `reference_image_paths`/
    `node_descriptions`) from a camera_nodes.json + view_descriptions.json pair, e.g. as
    produced by examples/generate_camera_relative_views.py.

    `base_dir`, if given, is joined onto each node's image_path -- the JSON stores paths
    relative to the repo root by convention (e.g. "reference_views_relative/view_000.png"),
    so a caller running from a different working directory (a notebook in notebooks/, say)
    can pass base_dir="..") to resolve them correctly. Not called automatically by
    CameraSpecialist or VisualizationOrchestrator -- which reference bank (if any) to use is
    the caller's choice, made explicitly at construction time.
    """
    nodes = json.loads(Path(nodes_path).read_text())
    descriptions = json.loads(Path(descriptions_path).read_text())

    image_paths = {}
    for node in nodes:
        image_path = node["image_path"]
        if base_dir:
            image_path = str(Path(base_dir) / image_path)
        image_paths[node["node_id"]] = image_path

    node_descriptions = {entry["node_id"]: entry["description"] for entry in descriptions}
    return image_paths, node_descriptions


def load_simple_reference_bank(
    descriptions_path: str,
    image_dir: Optional[str] = None,
    image_extension: str = ".png",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load a reference-view bank from the "simple" format used under
    reference_views_medical/<object>/ (e.g. reference_views_medical/skull/
    reference_views_simple.json, produced by camera_reasoning.simple_reference_labels) --
    a flat {node_id: description} JSON dict, with each node's image conventionally sitting
    right next to it as "<node_id><image_extension>" (no explicit image_path field, unlike
    load_reference_bank's camera_nodes.json format above).

    `image_dir` defaults to `descriptions_path`'s own directory, matching that convention
    (config.dataset_directory in medical_reference_views.py). Raises FileNotFoundError if a
    node's image doesn't exist there.
    """
    descriptions_path = Path(descriptions_path)
    node_descriptions = json.loads(descriptions_path.read_text())

    resolved_image_dir = Path(image_dir) if image_dir else descriptions_path.parent
    image_paths = {}
    for node_id in node_descriptions:
        image_path = resolved_image_dir / f"{node_id}{image_extension}"
        if not image_path.exists():
            raise FileNotFoundError(f"reference image for node {node_id!r} not found at {image_path}")
        image_paths[node_id] = str(image_path)

    return image_paths, node_descriptions


class CameraSpecialist(VisualizationSpecialist):
    agent_id = "camera_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        default_max_iterations: int = DEFAULT_MAX_ITERATIONS,
        reference_image_paths: Optional[Dict[str, str]] = None,
        node_descriptions: Optional[Dict[str, str]] = None,
        stop_similarity_threshold: int = DEFAULT_STOP_SIMILARITY_THRESHOLD,
        stop_improvement_margin: int = DEFAULT_STOP_IMPROVEMENT_MARGIN,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
        on_iteration: Optional[Callable[[dict], None]] = None,
        sequential_diagnosis: bool = True,
    ):
        self.session = session
        self.model = model
        self.default_max_iterations = default_max_iterations
        # Diagnose candidates one LLM call at a time (default) instead of all together in
        # one call -- more LLM calls per iteration, but avoids the model mislabeling which
        # candidate a score/description belongs to when several similar-looking renders
        # are shown at once (see diagnose_blind_candidates_sequentially's docstring).
        self.sequential_diagnosis = sequential_diagnosis
        # Optional reference-view bank for Pass 1's grounding judgment (see module
        # docstring, "Reference grounding" in blind_visual_rollout_agent.py). Passing
        # neither is fine -- Pass 1 just reports "unclear" and Pass 2 falls back to a
        # directional sweep -- but the bank meaningfully improves selection quality when
        # one matching this dataset/isovalue is available (see load_reference_bank()).
        self.reference_image_paths = reference_image_paths
        self.node_descriptions = node_descriptions
        self.stop_similarity_threshold = stop_similarity_threshold
        self.stop_improvement_margin = stop_improvement_margin
        self.minimum_confidence = minimum_confidence
        # Called once per internal loop iteration with a normalized dict (see
        # _handle_camera_iteration) -- e.g. a notebook display callback. Optional; the loop
        # runs identically without one.
        self.on_iteration = on_iteration

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
            loop_result = run_blind_visual_rollout_alignment_loop(
                self.session,
                max_iterations=max_iterations,
                model=self.model,
                reference_image_paths=self.reference_image_paths,
                node_descriptions=self.node_descriptions,
                stop_similarity_threshold=self.stop_similarity_threshold,
                stop_improvement_margin=self.stop_improvement_margin,
                minimum_confidence=self.minimum_confidence,
                on_iteration_end=self._handle_camera_iteration if self.on_iteration else None,
                sequential_diagnosis=self.sequential_diagnosis,
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
                reason=f"Blind camera loop converged (STOP) after {iterations_used} iteration(s).",
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
                f"Blind camera loop used all {max_iterations} iteration(s) without "
                "reaching STOP."
            ),
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=["adjust_viewpoint"],
            failure_type="iteration_limit_reached",
            iterations_used=iterations_used,
            artifacts=[image_path],
        )

    def _handle_camera_iteration(self, trace_path: str) -> None:
        """Read one iteration's saved blind-rollout trace (see
        blind_visual_rollout_agent.save_blind_rollout_trace) and normalize it into the
        shared on_iteration shape (see IsovalueSpecialist._handle_loop_iteration for the
        isovalue-side version of this same normalized dict -- a caller can register one
        callback that handles both agents identically).

        `real_action` on each candidate is included for the callback's benefit only (e.g.
        notebook debug display) -- it's read from the trace file's private mapping, which
        was never sent to or seen by the LLM during selection; surfacing it here after the
        fact doesn't compromise the blind selection that already happened.

        `similarity_score`/`confidence`/`reference_match_quality` are pulled from Pass 1's
        own diagnosis for that candidate_id and attached directly to each candidate here --
        without this, matching "which candidate had the highest score" (from the raw Pass 1
        JSON) to "which thumbnail is that" (from the image list) has to be done by eye
        across two separate places, which is exactly what made an earlier apparent
        selection mismatch hard to verify.
        """
        trace = json.loads(Path(trace_path).read_text())
        selection = trace["pass_2_deterministic_selection"]
        selected_candidate = selection.get("selected_candidate")
        candidate_mapping = trace["private_candidate_mapping"]
        diagnosis_by_id = {
            c["candidate_id"]: c for c in trace["pass_1_visual_diagnosis"]["parsed"]["candidates"]
        }

        candidates = []
        for c in trace["blind_candidates"]:
            diagnosis = diagnosis_by_id.get(c["candidate_id"], {})
            candidates.append({
                "label": c["candidate_id"],
                "image_path": c["image_path"],
                "selected": c["candidate_id"] == selected_candidate,
                "real_action": candidate_mapping.get(c["candidate_id"]),
                "similarity_score": diagnosis.get("similarity_score"),
                "confidence": diagnosis.get("confidence"),
                "reference_match_quality": diagnosis.get("reference_match_quality"),
                "comparison_to_target": diagnosis.get("comparison_to_target"),
            })

        self.on_iteration({
            "agent_id": self.agent_id,
            "iteration": trace["iteration"],
            "current_image_path": trace["current_image_path"],
            "candidates": candidates,
            "selected_label": selected_candidate,
            "reasoning": selection.get("selection_reason", ""),
            "extra": {
                "final_action": trace.get("final_action"),
                "applied_action": trace.get("applied_action"),
            },
        })

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
