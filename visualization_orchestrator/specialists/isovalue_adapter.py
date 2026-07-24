"""Isovalue specialist: a histogram-informed candidate sweep instead of blind iterative
up/down guessing. No autonomous isovalue agent existed in camera_reasoning/ before this --
the logic lives here rather than as a new top-level package, since it is small and
specific to this one adapter.

Candidate isovalues are the LOCAL MINIMA ("valleys") of the volume's intensity histogram
(mirrors examples/render_head_iso_candidates.ipynb's approach, restricted to minima only
-- see compute_histogram_local_minima_isovalues's docstring for why). Each candidate is
rendered from several camera angles (not just one) so noise/quality can be judged from more
than a single, possibly-flattering view -- a surface that looks clean head-on can still be
fragmented from the side. Those angles are combined into ONE labeled tiled image per
candidate (see _combine_views_into_grid) rather than sent as several separate image
attachments, so comparing N candidates still only means N image attachments in the single
comparison call -- not N times the number of angles.

Operates on an existing, already-`.initialize()`d CameraReasoningSession in isosurface mode
(use_volume_rendering=False), via CameraReasoningSession.set_isovalue(), which rebuilds only
the isosurface actor in place and leaves the session's camera/renderer untouched -- so a
camera task that ran before or after this one composes onto the same render correctly.
"""
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage, ImageDraw

from camera_reasoning.camera_actions import apply_action
from camera_reasoning.camera_state import get_camera_state, set_camera_state
from camera_reasoning.chatgpt_client import ask_chatgpt
from camera_reasoning.session import CameraReasoningSession
from camera_reasoning.volume_scene import save_screenshot

from ..capabilities import AgentCapability, AgentSpec
from ..json_utils import extract_json_object
from ..models import AgentExecutionResult
from ..state import VisualizationState
from .base import VisualizationSpecialist

DEFAULT_NUM_BINS = 256
DEFAULT_VALUE_RANGE = (0.0, 256.0)
DEFAULT_SMOOTHING_WINDOW = 9

# Six views per candidate, computed RELATIVE to the session's camera state at the moment
# the sweep runs (via camera_actions.apply_action) rather than fixed absolute world axes.
# examples/render_head_iso_candidates.ipynb's six_view_cameras() used absolute axes tuned
# specifically for vis_male_128x256x256_uint8.raw ("determined empirically for THIS
# dataset... would need to be re-verified for a different volume") -- that doesn't transfer
# to other datasets (e.g. the skull dataset this orchestrator uses), so these labels mean
# "relative to wherever the camera already was", not true anatomical front/back/etc.
MULTI_ANGLE_VIEWS: List[Tuple[str, Optional[str]]] = [
    ("front", None),
    ("right", "AZIMUTH_RIGHT_COARSE"),
    ("back", "AZIMUTH_RIGHT_180"),
    ("left", "AZIMUTH_LEFT_COARSE"),
    ("top", "ELEVATION_UP_COARSE"),
    ("bottom", "ELEVATION_DOWN_COARSE"),
]

GRID_COLUMNS = 3
GRID_LABEL_HEIGHT = 28

ISOVALUE_AGENT_SPEC = AgentSpec(
    agent_id="isovalue_controller",
    description=(
        "Selects a scalar isovalue by rendering every histogram-local-minimum candidate "
        "as a multi-angle tiled image and comparing them all in one batched LLM call."
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

ISOVALUE_CANDIDATE_SELECTION_PROMPT_TEMPLATE = """You are selecting the best scalar isovalue threshold for extracting an isosurface
from a 3D volume, by comparing ACTUAL RENDERED OUTCOMES at several candidate thresholds --
not by reasoning about the numbers alone.

Goal: {goal}

Success criteria this selection must satisfy:
{success_criteria_block}

Each candidate isovalue sits at a local minimum ("valley") of the volume's intensity
histogram -- a threshold between two materials, where relatively few voxels share that
exact intensity. Thresholding at a valley tends to produce a cleaner, less noisy surface
than thresholding inside a homogeneous material (a histogram peak), which is why every
candidate below was chosen this way rather than swept uniformly.

You are given one image per candidate, each preceded by "[ISOVALUE_<value>]".{multi_angle_note}

Candidates available this call: {candidate_list}

Evaluate every candidate listed above exactly once -- do not shortlist, group, or skip any.

Respond with STRICT JSON ONLY, no prose outside the JSON:
{{
  "candidate_evaluation": [
    {{"isovalue": <number>, "observation": "<what is visible at this threshold>", "noise_level": "clean" | "minor_noise" | "noisy" | "fragmented", "satisfies_goal": true | false}}
  ],
  "selected_isovalue": <number, MUST be exactly one of the candidates listed above>,
  "goal_satisfied": true | false,
  "reasoning": "<why the selected isovalue is the best match for the goal, referencing the other candidates>"
}}
"""

MULTI_ANGLE_NOTE = (
    " Each candidate's image is a single tiled grid combining 6 different camera angles "
    "(front/right/back/left/top/bottom), each sub-panel labeled with its angle name in the "
    "top-left corner. Judge overall surface quality and noise level using ALL sub-panels "
    "together -- a surface that looks clean in one sub-panel can still be fragmented or "
    "speckly in another, and vice versa -- never judge from a single sub-panel alone."
)


def compute_histogram_local_minima_isovalues(
    raw_path: str,
    scalar_type: str = "uint8",
    num_bins: int = DEFAULT_NUM_BINS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> List[int]:
    """Find candidate isovalues at the local minima ("valleys") of the volume's smoothed
    intensity histogram. A valley sits between two materials -- few voxels share that exact
    intensity there -- so thresholding at one tends to produce a clean, low-noise surface
    boundary. A local maximum sits in the middle of a homogeneous material -- thresholding
    there is what produces fragmented, speckly surfaces -- so maxima are deliberately
    excluded as candidates (mirrors examples/render_head_iso_candidates.ipynb's approach,
    which used both minima and maxima; this only uses minima).

    Reads the raw scalar values directly from `raw_path` (shape doesn't matter for a
    histogram, so no `dimensions` argument is needed here).
    """
    intensities = np.fromfile(raw_path, dtype=np.dtype(scalar_type))
    hist, _ = np.histogram(intensities, bins=num_bins, range=value_range)

    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed_hist = np.convolve(hist, kernel, mode="same")

    diff = np.diff(smoothed_hist)
    sign = np.sign(diff)
    sign[sign == 0] = 1
    sign_changes = np.diff(sign)
    local_minima = np.flatnonzero(sign_changes > 0) + 1

    return sorted(int(v) for v in local_minima)


def _combine_views_into_grid(
    views: List[Tuple[str, str]],
    output_path: str,
    columns: int = GRID_COLUMNS,
) -> str:
    """Tile several labeled view images into ONE combined image -- each tile's angle label
    is drawn directly onto the tile (not just a separate caption), so it survives being
    sent as a single image attachment rather than several labeled ones. Assumes all views
    are the same size (true here -- every render uses the session's fixed render window).
    """
    tiles = [(view_label, PILImage.open(path).convert("RGB")) for view_label, path in views]
    tile_w, tile_h = tiles[0][1].size
    rows = math.ceil(len(tiles) / columns)

    canvas = PILImage.new("RGB", (tile_w * columns, (tile_h + GRID_LABEL_HEIGHT) * rows), color=(15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, (view_label, image) in enumerate(tiles):
        col, row = index % columns, index // columns
        x, y = col * tile_w, row * (tile_h + GRID_LABEL_HEIGHT)
        draw.rectangle([x, y, x + tile_w, y + GRID_LABEL_HEIGHT], fill=(15, 15, 15))
        draw.text((x + 8, y + 6), view_label, fill=(255, 255, 255))
        canvas.paste(image, (x, y + GRID_LABEL_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def render_isovalue_candidates(
    session: CameraReasoningSession,
    isovalues: List[int],
    output_dir: str,
    multi_angle: bool = True,
) -> List[Tuple[int, str]]:
    """Render each candidate isovalue and return ONE representative image path per
    candidate. When `multi_angle` (default True), that image is a single tiled grid of 6
    angles (see MULTI_ANGLE_VIEWS, _combine_views_into_grid); when False, it's just the
    current/"front" angle, matching the original one-image-per-candidate behavior.

    Restores the session's original isovalue AND camera state afterward, even if a render
    fails partway through.

    Returns [(isovalue, image_path), ...] in the same order as `isovalues`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_isovalue = session.isovalue
    camera = session._renderer.GetActiveCamera()
    original_camera_state = get_camera_state(camera)

    views_to_render = MULTI_ANGLE_VIEWS if multi_angle else MULTI_ANGLE_VIEWS[:1]

    rendered: List[Tuple[int, str]] = []
    try:
        for isovalue in isovalues:
            session.set_isovalue(isovalue)
            candidate_views: List[Tuple[str, str]] = []
            for view_label, action in views_to_render:
                set_camera_state(camera, original_camera_state)
                if action is not None:
                    apply_action(action, camera, session._renderer)
                else:
                    session._renderer.ResetCameraClippingRange()
                image_path = str(out_dir / f"isovalue_{isovalue}_{view_label}.png")
                save_screenshot(session._render_window, image_path)
                candidate_views.append((view_label, image_path))

            if multi_angle:
                grid_path = str(out_dir / f"isovalue_{isovalue}_grid.png")
                _combine_views_into_grid(candidate_views, grid_path)
                rendered.append((isovalue, grid_path))
            else:
                rendered.append((isovalue, candidate_views[0][1]))
    finally:
        session.set_isovalue(original_isovalue)
        set_camera_state(camera, original_camera_state)
        session._renderer.ResetCameraClippingRange()

    return rendered


def run_isovalue_candidate_sweep(
    session: CameraReasoningSession,
    goal: str,
    success_criteria: List[str],
    model: Optional[str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    output_dir: Optional[str] = None,
    multi_angle: bool = True,
) -> dict:
    """One-shot isovalue selection: compute candidate isovalues from the volume's histogram
    local minima, render each as one (optionally multi-angle tiled) image, and ask the LLM
    to pick the best match for `goal` in a single batched comparison call.

    Returns {"converged": bool, "candidates": [(isovalue, image_path), ...],
    "candidate_evaluation": [...], "final_isovalue": int, "final_image_path": str,
    "reasoning": str, "raw_response": str}.
    """
    candidate_isovalues = compute_histogram_local_minima_isovalues(
        session.raw_path,
        scalar_type=session.scalar_type,
        num_bins=num_bins,
        value_range=value_range,
        smoothing_window=smoothing_window,
    )
    if not candidate_isovalues:
        # Degenerate histogram (e.g. perfectly flat) -- nothing to compare against; fall
        # back to the session's current isovalue rather than guessing.
        candidate_isovalues = [int(round(session.isovalue))]

    render_dir = output_dir or str(Path(session.output_dir) / "screenshots" / "isovalue_candidates")
    rendered = render_isovalue_candidates(session, candidate_isovalues, render_dir, multi_angle=multi_angle)

    reference_items = [(f"ISOVALUE_{isovalue}", image_path, "") for isovalue, image_path in rendered]
    prompt = ISOVALUE_CANDIDATE_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        success_criteria_block="\n".join(f"- {c}" for c in success_criteria) or "- (none specified)",
        multi_angle_note=MULTI_ANGLE_NOTE if multi_angle else "",
        candidate_list=", ".join(str(isovalue) for isovalue, _ in rendered),
    )
    response_text = ask_chatgpt(prompt=prompt, reference_items=reference_items, model=model)
    parsed = extract_json_object(response_text)

    valid_isovalues = {isovalue for isovalue, _ in rendered}
    selected_raw = parsed.get("selected_isovalue") if parsed else None
    selection_is_valid = (
        isinstance(selected_raw, (int, float))
        and not isinstance(selected_raw, bool)
        and int(round(selected_raw)) in valid_isovalues
    )

    if not selection_is_valid:
        # Invalid/unresolvable response -- render_isovalue_candidates already restored the
        # session's original isovalue, so leave it there and report failure rather than
        # guessing an arbitrary candidate.
        return {
            "converged": False,
            "candidates": rendered,
            "candidate_evaluation": (parsed or {}).get("candidate_evaluation", []),
            "final_isovalue": session.isovalue,
            "final_image_path": session.render_and_save(),
            "reasoning": "Model response was not valid JSON or selected an unknown isovalue.",
            "raw_response": response_text,
        }

    selected_isovalue = int(round(selected_raw))
    session.set_isovalue(selected_isovalue)
    final_image_path = session.render_and_save()

    return {
        "converged": bool(parsed.get("goal_satisfied", True)),
        "candidates": rendered,
        "candidate_evaluation": parsed.get("candidate_evaluation", []),
        "final_isovalue": selected_isovalue,
        "final_image_path": final_image_path,
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": response_text,
    }


class IsovalueSpecialist(VisualizationSpecialist):
    agent_id = "isovalue_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        num_bins: int = DEFAULT_NUM_BINS,
        value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        multi_angle: bool = True,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ):
        self.session = session
        self.model = model
        self.num_bins = num_bins
        self.value_range = value_range
        self.smoothing_window = smoothing_window
        # Render each candidate as a 6-angle tiled grid instead of 1 flat view (default) --
        # still just one image attachment per candidate in the comparison call, so this
        # doesn't change the call count, only what each image shows. Set False to render
        # only the current angle, matching the original single-view behavior.
        self.multi_angle = multi_angle
        # Called once with a normalized dict after the candidate sweep completes (see
        # _handle_sweep_result) -- e.g. a notebook display callback. Optional; the sweep
        # runs identically without one.
        self.on_iteration = on_iteration

    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        if state.isovalue is not None:
            self.session.set_isovalue(state.isovalue)

        sweep_result = run_isovalue_candidate_sweep(
            self.session,
            goal=goal,
            success_criteria=success_criteria,
            model=self.model,
            num_bins=self.num_bins,
            value_range=self.value_range,
            smoothing_window=self.smoothing_window,
            multi_angle=self.multi_angle,
        )

        if self.on_iteration:
            self._handle_sweep_result(sweep_result)

        state_patch = {
            "isovalue": sweep_result["final_isovalue"],
            "rendered_image_path": sweep_result["final_image_path"],
        }
        iterations_used = len(sweep_result["candidates"])

        if sweep_result["converged"]:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.75,
                reason=f"Selected isovalue={sweep_result['final_isovalue']} from {iterations_used} "
                       f"histogram-valley candidate(s): {sweep_result['reasoning']}",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=iterations_used,
                artifacts=[sweep_result["final_image_path"]],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if iterations_used else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.3,
            reason=(
                f"None of the {iterations_used} histogram-valley candidate isovalue(s) "
                f"satisfied the goal: {sweep_result['reasoning']}"
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
            artifacts=[sweep_result["final_image_path"]],
        )

    def _handle_sweep_result(self, sweep_result: dict) -> None:
        """Normalize one candidate sweep into the shared on_iteration shape (see
        CameraSpecialist._handle_camera_iteration for the camera-side version of this same
        normalized dict -- a caller can register one callback that handles both agents
        identically). Reported as a single "iteration" (index 0) since the sweep is a
        one-shot batched comparison, not a multi-step loop."""
        evaluation_by_isovalue = {
            int(round(e["isovalue"])): e
            for e in sweep_result.get("candidate_evaluation", [])
            if isinstance(e, dict) and isinstance(e.get("isovalue"), (int, float)) and not isinstance(e.get("isovalue"), bool)
        }

        candidates = []
        for isovalue, image_path in sweep_result["candidates"]:
            evaluation = evaluation_by_isovalue.get(isovalue, {})
            candidates.append({
                "label": f"ISOVALUE_{isovalue}",
                "image_path": image_path,
                "selected": isovalue == sweep_result["final_isovalue"],
                "real_action": None,
                "observation": evaluation.get("observation"),
                "satisfies_goal": evaluation.get("satisfies_goal"),
                "noise_level": evaluation.get("noise_level"),
            })

        self.on_iteration({
            "agent_id": self.agent_id,
            "iteration": 0,
            "current_image_path": sweep_result["final_image_path"],
            "candidates": candidates,
            "selected_label": f"ISOVALUE_{sweep_result['final_isovalue']}",
            "reasoning": sweep_result.get("reasoning") or "",
            "extra": {
                "final_isovalue": sweep_result["final_isovalue"],
                "converged": sweep_result["converged"],
            },
        })
