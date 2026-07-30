"""Isovalue specialist: splits the volume's full intensity range into a FIXED set of
equal-width WINDOWS (no dependence on the volume's own histogram shape), then selects one
via a TWO-STAGE, GOAL-BLIND pipeline, then DETERMINISTICALLY derives an opacity/color
transfer function from that window's own [low, high] range and applies it via direct volume
rendering.

Why fixed windows instead of histogram-derived bands: an earlier version segmented the
histogram into bands via valley-finding (local minima between peaks), which worked when a
target material formed a genuinely separate histogram peak but produced only 1-2 useless
bands for datasets where materials blend as a smooth, unimodal intensity gradient with no
interior valley at all (bone never separates from soft tissue in some CT volumes). A fixed,
dataset-agnostic grid sidesteps that: it never depends on the histogram having any
particular shape.

Why the selection pipeline is two-stage and GOAL-BLIND in its first stage: earlier versions
let the vision LLM see the rendered image AND the goal in the same call (both in one batched
comparison, and later one call per window). Both were found, by direct visual inspection, to
confidently hallucinate goal-implied structure that wasn't actually present -- e.g. asked to
"show the skull", the model described a smooth, featureless, soft-tissue head (visible ear,
scalp dome, neck skin folds -- no sutures, no facial bones, nothing bone-like) as "a complete
skull... clear anatomical detail including the cranium, facial bones, and suture lines",
with high self-reported confidence, EVEN when that image was the only one shown in an
isolated single-image call. That ruled out simple cross-candidate mislabeling as the (sole)
cause -- the model was inventing goal-consistent detail from a plausible silhouette and
color, not confusing one candidate's description for another's. The fix: never let a stage
that's looking at an image also know what it's "supposed" to find there.

    Window renders (6 SEPARATE full-resolution views per window, never tiled)
        |
        v
    Stage 1 (per window, BLIND): ONE call showing all of that window's views TOGETHER as
        separate images -- describes only visible geometry -- no goal, no window
        name/intensity, no other candidates, opaque "candidate_N" ids only, opaque
        "view_N" ids only (never front/back/left/right/top/bottom)
        |
        v
    Stage 2 (ONE call, TEXT-ONLY): given the goal and every candidate's stored blind
        observation (no images), first judge each candidate independently against every
        success criterion (a candidate that's merely relatively better than the others
        does not pass), then select among the ones that passed, or abstain ("no_match")
        if none did -- re-checked in code, not just trusted from the model's top-level
        "decision" field (see run_isovalue_band_selection's "absolute floor")
        |
        v
    Selected window (mapped back from its opaque id) or no_match

Each window's 6 views are sent to Stage 1 as 6 SEPARATE full-resolution images in one call
(via `extra_images`), not combined into a single tiled composite -- tiling shrinks each view
and can visually compress detail that matters for the observation. View ids ("view_0",
"view_1", ...) are opaque for the same reason window/goal identity is hidden: the initial
camera orientation the sweep starts from is arbitrary (whatever the session's camera
happened to be pointed at before this call), so labeling them "front"/"back"/"left"/"right"/
"top"/"bottom" would assert a semantic/anatomical meaning that doesn't actually exist for an
arbitrary starting orientation. The real camera transformation behind each view id (see
VIEW_ACTIONS) is kept internal, for our own debugging/logging only -- never sent to the LLM.

Stage 2 never sees the images again -- it can only reason from what Stage 1 already
observed, so it cannot introduce visual "evidence" beyond what was actually reported.
Abstention ("no_match") is a legitimate outcome, not a failure to paper over -- if no
candidate's blind observation supports the goal, nothing gets silently guessed. This
includes the case where every candidate is flawed but one is merely LESS flawed than the
rest: Stage 2 must judge each candidate independently against the success criteria before
comparing candidates against each other, so a "best of a uniformly poor set" pick is
rejected (both by the prompt's own instructions and, redundantly, by a code-side check)
rather than accepted just because nothing else was better.

Evaluation previews are rendered in a NEUTRAL GRAYSCALE palette (build_evaluation_ramp_for_
window), not the warm beige/tan used for the final applied result -- every window
previously rendered in the same warm "bone-like" tone regardless of what material it
actually contained, which is a plausible contributor to the hallucination above (pattern-
matching on a shared "looks bone-colored" cue instead of on structure). All windows share
the SAME grayscale mapping (not a unique color per window), since a per-window color would
just create a new candidate-identity shortcut for the model to exploit. The final applied
result (after a window is actually selected) still uses build_opacity_ramp_for_band's normal
warm palette -- neutrality is specifically an evaluation-time concern.

No numerical confidence score (no "match_quality") drives selection anywhere in this
pipeline -- Stage 1 reports only structured, ungraded visual observations, and Stage 2's
selection is a categorical decision ("selected" or "no_match") grounded in those
observations, not a number.

This module deliberately avoids any goal-, dataset-, or anatomy-specific vocabulary in its
own code and prompts -- window/candidate descriptions are generic geometric properties
(shape, continuity, cavities, fragments, symmetry, etc.) that apply to arbitrary scientific
volumes, not just medical scans.

Operates on an existing, already-`.initialize()`d CameraReasoningSession, via
CameraReasoningSession.set_transfer_function() -- used both for per-window previews
(restored afterward) and the final applied result -- which preserves camera state and can
switch the session between isosurface and volume-rendering mode in place, so a camera task
that ran before or after this one composes onto the same render correctly.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

DEFAULT_VALUE_RANGE = (0.0, 256.0)
DEFAULT_NUM_WINDOWS = 8  # how many equal-width windows to split the full intensity range into
DEFAULT_PEAK_OPACITY = 0.55
DEFAULT_SUSTAIN_OPACITY = 0.70

# Neutral grayscale shades (dim, bright) used for EVERY window's evaluation preview -- same
# hue and relative brightness curve regardless of a window's actual [low, high] (see
# build_evaluation_ramp_for_window for how these get anchored to each window's own range),
# so color can't act as a false "this looks like the target material" cue, and no window
# gets a visually distinctive color that could itself become a new identity shortcut. Only
# used for evaluation renders; the final applied result uses build_opacity_ramp_for_band's
# normal warm palette instead.
EVALUATION_GRAY_SHADES: Tuple[float, float] = (0.35, 0.8)

# The 6 real camera transformations behind the opaque "view_0".."view_5" ids exposed to
# Stage 1 -- computed RELATIVE to the session's camera state at the moment the sweep runs
# (via camera_actions.apply_action) rather than fixed absolute world axes -- see
# examples/render_head_iso_candidates.ipynb's six_view_cameras, which used absolute axes
# tuned specifically for one dataset and wouldn't transfer to others -- so this mapping
# means "relative to wherever the camera already was", not any semantic direction. Kept OUT
# of every prompt; this exists purely for our own debugging/logging (see
# run_isovalue_band_selection's returned "view_action_map"). Position 0 (None) is the
# current/starting orientation.
VIEW_ACTIONS: List[Optional[str]] = [
    None,
    "AZIMUTH_RIGHT_COARSE",
    "AZIMUTH_RIGHT_180",
    "AZIMUTH_LEFT_COARSE",
    "ELEVATION_UP_COARSE",
    "ELEVATION_DOWN_COARSE",
]

ISOVALUE_AGENT_SPEC = AgentSpec(
    agent_id="isovalue_controller",
    description=(
        "Splits the volume's full intensity range into fixed equal-width windows, "
        "describes each window's preview BLIND to the goal (Stage 1), then makes one "
        "text-only, goal-aware selection from those stored observations (Stage 2, may "
        "abstain), then deterministically derives an opacity ramp from the selected "
        "window's own range and renders it via direct volume rendering."
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
    owned_state_fields={"transfer_function", "rendered_image_path"},
    required_state_fields={"dataset_path", "rendered_image_path"},
    side_effects=["changes extracted geometry"],
)

# --- Stage 1: blind visual observation ----------------------------------------------------
#
# Deliberately STATIC -- no goal, window name, intensity range, or other candidate's
# information is ever interpolated into this prompt. See module docstring for why.
BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE = """
Observe the provided images of one scientific volume-rendering candidate.

The provided images show the same rendered candidate from different camera viewpoints, each
preceded by its own opaque id (e.g. "view_0", "view_1", ...).

The identifiers view_0, view_1, and so on are arbitrary. They do not indicate the semantic
front, back, left, right, top, or bottom of the object -- the camera's starting orientation
for this sweep was arbitrary, so no such labels are meaningful here.

You do not know the user's intended visualization goal. Do not guess it.

Inspect each view independently first. Then summarize visual properties that are
consistently supported across one or more views.

Describe only visual evidence directly supported by the images.

Do not infer object identity, semantic class, anatomy, or material from:
- color,
- overall familiarity,
- expected dataset contents,
- assumptions about what the user may want.

Analyze general visible properties:

1. Overall geometry
   - smooth or irregular
   - continuous or fragmented
   - compact, elongated, layered, branching, sheet-like, tubular, or other
   - enclosed outer surface or exposed internal structures

2. Visible structures
   - cavities or openings
   - protrusions
   - thin structures
   - repeated components
   - nested or layered regions
   - disconnected fragments
   - surface folds or ridges

3. Image quality
   - amount of isolated noise
   - occlusion
   - missing regions
   - whether important structures appear obscured

4. Cross-view consistency
   - for every observation, record exactly which opaque view ids support it
   - distinguish clearly visible observations from uncertain interpretations

Rules:
- Do not mention the intended goal.
- Do not decide whether the candidate satisfies a goal.
- Do not use color as evidence of identity or material.
- Do not claim a structure unless you can state which view id(s) it appears in.
- Do not treat view ids or their ordering as meaningful beyond distinguishing one image from
  another.
- Prefer neutral geometric descriptions over semantic labels.
- When uncertain, explicitly mark the observation as uncertain.
- Do not invent absent features.

Respond with STRICT JSON ONLY, no prose outside the JSON, matching this shape:
{
  "per_view_observations": {
    "view_0": ["<short observation specific to this view>", "..."],
    "view_1": ["..."]
  },
  "cross_view_summary": {
    "clear_observations": [
      {"description": "<specific visible structure/feature>", "supporting_views": ["<view ids>"]}
    ],
    "uncertain_observations": [
      {"description": "<possible but not clearly confirmed feature>", "supporting_views": ["<view ids>"]}
    ]
  },
  "noise_and_artifacts": {
    "isolated_fragments": "<e.g. none | low | moderate | high>",
    "surface_noise": "<e.g. none | low | moderate | high>",
    "occlusion": "<e.g. none | low | moderate | high>"
  }
}

Include an entry in "per_view_observations" for every view id you were shown, even if its
value is an empty list.
"""

# --- Stage 2: goal-aware selection from stored observations (text-only) -------------------
#
# Two-step by design (see run_isovalue_band_selection's absolute-floor check): Step 1 forces
# an ABSOLUTE, per-candidate verdict -- does THIS candidate's own observation satisfy every
# success criterion, judged in isolation -- before Step 2 is allowed to choose among the
# passers. This exists because a single "pick the best candidate" instruction invites purely
# RELATIVE reasoning ("candidate_0 is better than 2-7"), which can select a candidate that
# doesn't actually satisfy the goal at all, merely one that's least-bad among uniformly poor
# options. The per-candidate verdict is re-checked in code, not just trusted from "decision".
GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE = """Decide which candidate, if any, satisfies the user's visualization goal.

User goal:
{goal}

Success criteria this selection must satisfy:
{success_criteria_block}

Candidate observations:
{candidate_observations}

The candidate observations were created without knowing the user goal.

Evaluate candidates in two separate steps. Do not skip to comparison before finishing step 1.

Step 1 -- independent, absolute assessment (per candidate):
For EVERY candidate listed above, decide separately whether ITS OWN observation -- judged
on its own merits, not by comparison to any other candidate -- actually satisfies EVERY
listed success criterion. A candidate's verdict is "passes" only if all criteria are
clearly met by evidence in ITS OWN observation. If even one criterion is not clearly met
by that candidate's own evidence, its verdict is "fails" -- regardless of how it compares
to the other candidates.

Step 2 -- selection (only among candidates that passed step 1):
- If exactly one candidate passed step 1, select it.
- If more than one passed, select whichever has the clearest, least uncertain supporting
  evidence.
- If NO candidate passed step 1, the decision is "no_match" -- do not select the
  least-bad candidate among ones that failed step 1, even if it is clearly better than
  the rest.

Rules:
- Base every judgment only on the supplied observations.
- Do not introduce structures or properties absent from the observations.
- Favor clear observations over uncertain observations.
- Do not select a candidate merely because its general silhouette could be associated with
  the requested target.
- Do not use candidate IDs or their ordering as evidence.
- A candidate that is merely "better than the others" but does not itself satisfy every
  success criterion must fail step 1 and cannot be selected in step 2.
- Do not force a selection.

Respond with STRICT JSON ONLY, no prose outside the JSON:
{{
  "candidate_verdicts": {{
    "<candidate_N>": {{
      "verdict": "passes" | "fails",
      "criteria_met": ["<criterion text(s) satisfied by this candidate's own observation>"],
      "criteria_not_met": ["<criterion text(s) NOT satisfied by this candidate's own observation>"]
    }}
  }},
  "decision": "selected" | "no_match",
  "selected_candidate": "<candidate_N whose verdict is 'passes', or null if no_match>",
  "supporting_observations": ["<observation text(s) that justify the decision>"],
  "contradictory_observations": ["<observation text(s) that argue against it, if any>"],
  "explanation": "<why this candidate was selected, or why none qualified>"
}}

Include an entry in "candidate_verdicts" for every candidate listed above.
"""

WINDOW_LABEL_PREFIX = "WINDOW_"
CANDIDATE_ID_PREFIX = "candidate_"


def compute_fixed_windows(
    num_windows: int = DEFAULT_NUM_WINDOWS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
) -> List[dict]:
    """Split the full intensity range into `num_windows` equal-width [low, high) windows,
    covering the whole range regardless of the volume's actual data distribution -- no file
    is read, this is pure arithmetic. See the module docstring for why this replaced an
    earlier histogram-band approach.

    Returns windows sorted by intensity, each {"low": int, "high": int, "peak": int} --
    "peak" here is just each window's own midpoint (there's no histogram to find a real
    peak in), used only as build_opacity_ramp_for_band's "where peak_opacity is reached"
    reference point.
    """
    low0, high0 = value_range
    width = (high0 - low0) / num_windows

    windows = []
    for i in range(num_windows):
        low = low0 + i * width
        high = high0 if i == num_windows - 1 else low0 + (i + 1) * width
        windows.append({
            "low": int(round(low)),
            "high": int(round(high)),
            "peak": int(round((low + high) / 2)),
        })
    return windows


def _strictly_increasing(points: List[tuple]) -> List[tuple]:
    """Drop any point whose first element doesn't strictly exceed the previous kept point's
    -- vtkPiecewiseFunction/vtkColorTransferFunction expect strictly increasing scalar
    positions; a degenerate window (e.g. peak coinciding with low) could otherwise produce
    duplicate x-values."""
    kept: List[tuple] = []
    for point in points:
        if kept and point[0] <= kept[-1][0]:
            continue
        kept.append(point)
    return kept


def build_opacity_ramp_for_band(
    band: dict,
    min_value: float = DEFAULT_VALUE_RANGE[0],
    max_value: float = DEFAULT_VALUE_RANGE[1],
    peak_opacity: float = DEFAULT_PEAK_OPACITY,
    sustain_opacity: float = DEFAULT_SUSTAIN_OPACITY,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """Deterministically derive an opacity/color transfer function from ONE window's own
    [low, high] range and peak -- instead of hand-tuned constants. (Named `band`/
    `build_opacity_ramp_for_band` rather than `window` for historical reasons -- the
    fixed-window dict shape is identical to the earlier histogram-band dict shape, {"low",
    "high", "peak"}, so this function works unchanged for either.) Used for the FINAL
    applied result -- see build_evaluation_ramp_for_window for the neutral-color variant
    used during evaluation.

    Shape: opacity stays 0 below the window, ramps up GRADUALLY starting at `low` (never a
    hard step) to `peak_opacity` by the window's own peak (its midpoint), then a slightly
    higher `sustain_opacity` from `high` onward. Gradual ramps let THICK, continuous
    material (the real target, many voxels deep along the viewing ray) accumulate to full
    opacity via depth, while THIN/isolated structures at the same intensity (noise) stay
    comparatively faint since they have little depth to accumulate over -- see the isovalue
    specialist's module docstring.

    Color: near-black below the window (invisible anyway), transitioning to a neutral warm
    tone by the peak and held through to max_value (mostly occluded by then regardless).
    """
    low, high, peak = band["low"], band["high"], band["peak"]
    transition = max(1, (peak - low) // 2)
    ramp_start = max(min_value, low - transition)

    opacity_points = _strictly_increasing([
        (min_value, 0.0),
        (ramp_start, 0.0),
        (low, 0.05),
        (peak, peak_opacity),
        (high, sustain_opacity),
        (max_value, sustain_opacity),
    ])
    color_points = _strictly_increasing([
        (min_value, 0.0, 0.0, 0.0),
        (low, 0.35, 0.25, 0.2),
        (peak, 0.85, 0.75, 0.65),
        (max_value, 0.85, 0.75, 0.65),
    ])
    return opacity_points, color_points


def build_evaluation_ramp_for_window(
    band: dict,
    min_value: float = DEFAULT_VALUE_RANGE[0],
    max_value: float = DEFAULT_VALUE_RANGE[1],
    peak_opacity: float = DEFAULT_PEAK_OPACITY,
    sustain_opacity: float = DEFAULT_SUSTAIN_OPACITY,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """Same opacity ramp SHAPE as build_opacity_ramp_for_band, and the same idea for color
    -- anchored to the window's OWN [low, peak] range, not the absolute [min_value,
    max_value] scale -- but with a NEUTRAL GRAYSCALE hue (R=G=B, EVALUATION_GRAY_SHADES)
    instead of the warm beige/tan used for the final applied result. Used only for Stage 1's
    blind-observation previews.

    Anchoring to the window's own range (rather than a fixed absolute-position ramp like
    "black at 0, light gray at 256") matters for visibility: a LOW window's actual material
    sits at a low absolute intensity, so a globally-anchored ramp would render it very
    close to black regardless of hue -- confirmed by direct observation, low windows were
    rendering nearly invisible. Anchoring locally keeps every window's material at a
    consistent, visible brightness, exactly like build_opacity_ramp_for_band already does
    for its warm palette -- only the HUE changes here (gray, not warm), not the shape.

    Every window still gets the SAME neutral hue and the SAME relative brightness curve
    (not a per-window-unique color), so color still can't act as a candidate-identity
    shortcut -- see the module docstring.
    """
    opacity_points, _ = build_opacity_ramp_for_band(
        band, min_value=min_value, max_value=max_value,
        peak_opacity=peak_opacity, sustain_opacity=sustain_opacity,
    )
    low, high, peak = band["low"], band["high"], band["peak"]
    transition = max(1, (peak - low) // 2)
    ramp_start = max(min_value, low - transition)
    dim_shade, bright_shade = EVALUATION_GRAY_SHADES

    color_points = _strictly_increasing([
        (min_value, 0.0, 0.0, 0.0),
        (ramp_start, 0.0, 0.0, 0.0),
        (low, dim_shade, dim_shade, dim_shade),
        (peak, bright_shade, bright_shade, bright_shade),
        (high, bright_shade, bright_shade, bright_shade),
        (max_value, bright_shade, bright_shade, bright_shade),
    ])
    return opacity_points, color_points


def view_action_map(multi_angle: bool = True) -> Dict[str, Optional[str]]:
    """The opaque "view_N" id -> real camera-action mapping currently in effect -- for our
    own debugging/logging only (see run_isovalue_band_selection's returned
    "view_action_map"); never sent to the LLM."""
    actions = VIEW_ACTIONS if multi_angle else VIEW_ACTIONS[:1]
    return {f"view_{i}": action for i, action in enumerate(actions)}


def render_window_previews(
    session: CameraReasoningSession,
    windows: List[dict],
    output_dir: str,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    multi_angle: bool = True,
) -> List[Tuple[dict, Dict[str, str]]]:
    """Render EVALUATION previews per window -- one full-resolution image PER VIEW, saved
    SEPARATELY (never combined into a tile), using the neutral-grayscale ramp
    (build_evaluation_ramp_for_window) via direct volume rendering -- NOT the final warm
    palette (that's only applied after a window is actually selected, in
    run_isovalue_band_selection). Each view is labeled only with an OPAQUE id ("view_0",
    "view_1", ...) -- see module docstring for why semantic labels aren't used.

    Restores the session's transfer function/isovalue (whichever mode it was in) AND camera
    state to whatever they were before this call afterward, even if a render fails partway
    through.

    Returns [(window, {view_id: image_path}), ...] in the same order as `windows`. When
    `multi_angle=False`, each window's dict has just `{"view_0": path}`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_use_volume_rendering = session.use_volume_rendering
    original_isovalue = session.isovalue
    original_opacity_points = session.opacity_points
    original_color_points = session.color_points
    camera = session._renderer.GetActiveCamera()
    original_camera_state = get_camera_state(camera)

    actions_to_render = VIEW_ACTIONS if multi_angle else VIEW_ACTIONS[:1]

    rendered: List[Tuple[dict, Dict[str, str]]] = []
    try:
        for window in windows:
            opacity_points, color_points = build_evaluation_ramp_for_window(
                window, min_value=value_range[0], max_value=value_range[1]
            )
            session.set_transfer_function(opacity_points, color_points)
            view_images: Dict[str, str] = {}
            for view_index, action in enumerate(actions_to_render):
                view_id = f"view_{view_index}"
                set_camera_state(camera, original_camera_state)
                if action is not None:
                    apply_action(action, camera, session._renderer)
                else:
                    session._renderer.ResetCameraClippingRange()
                image_path = str(out_dir / f"window_{window['low']}_{window['high']}_{view_id}.png")
                save_screenshot(session._render_window, image_path)
                view_images[view_id] = image_path

            rendered.append((window, view_images))
    finally:
        if original_use_volume_rendering:
            session.set_transfer_function(original_opacity_points, original_color_points)
        else:
            session.set_isovalue(original_isovalue)
        set_camera_state(camera, original_camera_state)
        session._renderer.ResetCameraClippingRange()

    return rendered


def _observe_window_blind(
    view_images: Dict[str, str], model: Optional[str]
) -> Tuple[Optional[dict], str]:
    """Stage 1: ONE LLM call per window/candidate, showing ALL of that window's rendered
    views TOGETHER as SEPARATE full-resolution images (via `extra_images`, never combined
    into a tile) -- the prompt is fully static (BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE),
    so no goal, window name, intensity range, or other candidate's information is ever sent
    in this call. Each image is preceded only by its OPAQUE view id ("view_0", ...) -- see
    module docstring for why semantic labels (front/back/etc.) aren't used.

    `view_images`: {view_id: image_path}, in the order views should be shown (dict
    insertion order, as produced by render_window_previews).

    Returns (parsed_response_or_None, raw_response_text).
    """
    extra_images = list(view_images.items())
    response_text = ask_chatgpt(
        prompt=BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE, extra_images=extra_images, model=model
    )
    parsed = extract_json_object(response_text)
    return parsed, response_text


def _select_candidate_from_observations(
    goal: str,
    success_criteria: List[str],
    candidate_observations: List[dict],
    model: Optional[str],
) -> Tuple[Optional[dict], str]:
    """Stage 2: ONE text-only LLM call (no images) given the goal and every candidate's
    Stage-1 blind observation, keyed by OPAQUE candidate ids ("candidate_0", ...) only --
    real window labels/intensities are never included, so the selection can't be biased by
    them (see module docstring). Grounded only in what Stage 1 already reported; may return
    "no_match" (abstain) rather than forcing a selection.

    `candidate_observations`: [{"candidate_id": str, "observation": dict | None}, ...].
    Entries with observation=None (a failed Stage-1 call) are still listed so the model
    knows that candidate exists but has no usable observation.

    Returns (parsed_response_or_None, raw_response_text).
    """
    blocks = []
    for entry in candidate_observations:
        observation = entry["observation"] if entry["observation"] is not None else {"error": "no observation available"}
        blocks.append(f"[{entry['candidate_id']}]\n{json.dumps(observation, indent=2)}")

    prompt = GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        success_criteria_block="\n".join(f"- {c}" for c in success_criteria) or "- (none specified)",
        candidate_observations="\n\n".join(blocks),
    )
    response_text = ask_chatgpt(prompt=prompt, model=model)
    parsed = extract_json_object(response_text)
    return parsed, response_text


def _summarize_observation(observation: Optional[dict]) -> str:
    """Turn a Stage-1 structured observation into one readable line -- for on_iteration
    display and human-facing reasoning text, not used in any prompt."""
    if not observation:
        return "(no observation available)"
    summary = observation.get("cross_view_summary") or {}
    parts = [
        item["description"]
        for item in summary.get("clear_observations", []) or []
        if isinstance(item, dict) and item.get("description")
    ]
    return "; ".join(parts) if parts else "(no clear observations)"


def run_isovalue_band_selection(
    session: CameraReasoningSession,
    goal: str,
    success_criteria: List[str],
    model: Optional[str] = None,
    num_windows: int = DEFAULT_NUM_WINDOWS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    output_dir: Optional[str] = None,
    multi_angle: bool = True,
) -> dict:
    """Split the full intensity range into `num_windows` fixed windows, render one
    (neutral-grayscale) evaluation preview per window, run the two-stage blind-observation
    pipeline (see module docstring) to pick one, then deterministically derive and apply an
    opacity ramp from the selected window's own range using the normal warm palette.
    (Function name kept as `run_isovalue_band_selection` for backward compatibility with
    existing callers -- "band" here means "fixed window".)

    Stage 1 (`_observe_window_blind`): `num_windows` calls, one per window -- each call
    shows ALL of that window's rendered views TOGETHER as separate full-resolution images
    (never tiled), with a fully static, goal-blind prompt. Stage 2
    (`_select_candidate_from_observations`): ONE text-only call given the goal and every
    candidate's stored observation (opaque "candidate_N" ids, no images, no real window
    labels), which judges each candidate independently against every success criterion
    (step 1) before choosing among the ones that passed (step 2) -- see the prompt template.
    Real window labels are only mapped back from the selected opaque id AFTER Stage 2
    completes.

    Absolute floor (only enforced when `success_criteria` is non-empty): even if Stage 2's
    top-level "decision" says "selected", the selected candidate's OWN step-1 verdict must be
    "passes" -- a candidate chosen merely for being relatively less bad than the other
    windows, without independently satisfying every criterion, is rejected and treated the
    same as "no_match". This is re-checked in code rather than trusted from the model's
    "decision" field, since a "pick the best candidate" framing otherwise tends to produce
    purely relative reasoning ("candidate_0 has less noise than 2-7") that can select a
    window which doesn't actually satisfy the goal at all.

    Returns {"converged": bool, "bands": [...] (the windows), "band_images": {label: path}
    (one representative view per window, for simple display), "view_images": {label:
    {view_id: path}} (every view actually shown to Stage 1, for logging/debugging),
    "view_action_map": {view_id: real camera action or None}, "band_evaluation": [{"window":
    label, "observation": dict | None}, ...], "candidate_verdicts": {candidate_id: {"verdict":
    "passes"|"fails", "criteria_met": [...], "criteria_not_met": [...]}} (Stage 2's step-1
    output, opaque ids, for debugging), "selected_band_label": str | None,
    "final_opacity_points"/"final_color_points": the applied ramp or None,
    "final_image_path": str, "reasoning": str, "raw_response": str}.
    """
    windows = compute_fixed_windows(num_windows=num_windows, value_range=value_range)

    render_dir = output_dir or str(Path(session.output_dir) / "screenshots" / "isovalue_windows")
    rendered = render_window_previews(
        session, windows, render_dir, value_range=value_range, multi_angle=multi_angle
    )

    window_labels = [f"{WINDOW_LABEL_PREFIX}{w['low']}_{w['high']}" for w in windows]
    window_view_images = {label: views for label, (_, views) in zip(window_labels, rendered)}
    window_images = {label: next(iter(views.values())) for label, views in window_view_images.items()}
    candidate_ids = [f"{CANDIDATE_ID_PREFIX}{i}" for i in range(len(windows))]
    candidate_to_window = dict(zip(candidate_ids, windows))
    candidate_to_label = dict(zip(candidate_ids, window_labels))

    # --- Stage 1: blind, per-candidate, all views for that candidate in ONE call --------
    raw_responses = []
    candidate_observations = []
    for candidate_id, label in zip(candidate_ids, window_labels):
        parsed, response_text = _observe_window_blind(window_view_images[label], model)
        candidate_observations.append({"candidate_id": candidate_id, "observation": parsed})
        raw_responses.append(
            f"=== {candidate_id} (blind observation, views={list(window_view_images[label])}) ===\n"
            f"{response_text}"
        )

    band_evaluation = [
        {"window": candidate_to_label[c["candidate_id"]], "observation": c["observation"]}
        for c in candidate_observations
    ]

    # --- Stage 2: goal-aware, text-only, from stored observations only -------------------
    selection_parsed, selection_raw = _select_candidate_from_observations(
        goal, success_criteria, candidate_observations, model
    )
    raw_responses.append(f"=== stage 2 selection ===\n{selection_raw}")
    raw_response = "\n\n".join(raw_responses)

    decision = (selection_parsed or {}).get("decision")
    selected_candidate_id = (selection_parsed or {}).get("selected_candidate")
    verdicts = (selection_parsed or {}).get("candidate_verdicts") or {}

    floor_rejected = False
    if success_criteria and decision == "selected" and selected_candidate_id in candidate_to_window:
        # Absolute floor: don't trust "decision" alone -- require that THIS candidate's own
        # step-1 verdict was "passes". Rejects a purely relative "best of a uniformly poor
        # set" pick (see module docstring) even if the model's top-level decision claims a
        # selection; a missing/malformed verdict entry is treated as a failed floor, not a
        # free pass.
        if (verdicts.get(selected_candidate_id) or {}).get("verdict") != "passes":
            floor_rejected = True
            decision = "no_match"
            selected_candidate_id = None

    if not selection_parsed or decision != "selected" or selected_candidate_id not in candidate_to_window:
        # Either an invalid/unresolvable response, a legitimate "no_match" abstention, or a
        # selection that failed the absolute floor -- either way, report failure rather than
        # guessing a window.
        if floor_rejected:
            reasoning = (
                "Rejected: the model's chosen candidate did not independently satisfy every "
                "success criterion on its own evidence (only relative to the other "
                f"candidates). Model's original explanation: "
                f"{(selection_parsed or {}).get('explanation') or '(none given)'}"
            )
        else:
            reasoning = (
                (selection_parsed or {}).get("explanation")
                or ("Model abstained: no candidate had sufficient visible evidence for the goal."
                    if decision == "no_match" else
                    "Model response was not valid JSON or selected an unknown candidate.")
            )
        return {
            "converged": False,
            "bands": windows,
            "band_images": window_images,
            "view_images": window_view_images,
            "view_action_map": view_action_map(multi_angle),
            "band_evaluation": band_evaluation,
            "candidate_verdicts": verdicts,
            "selected_band_label": None,
            "final_opacity_points": session.opacity_points,
            "final_color_points": session.color_points,
            "final_image_path": session.render_and_save(),
            "reasoning": reasoning,
            "raw_response": raw_response,
        }

    selected_window = candidate_to_window[selected_candidate_id]
    selected_label = candidate_to_label[selected_candidate_id]

    opacity_points, color_points = build_opacity_ramp_for_band(
        selected_window, min_value=value_range[0], max_value=value_range[1]
    )
    session.set_transfer_function(opacity_points, color_points)
    final_image_path = session.render_and_save()

    return {
        "converged": True,
        "bands": windows,
        "band_images": window_images,
        "view_images": window_view_images,
        "view_action_map": view_action_map(multi_angle),
        "band_evaluation": band_evaluation,
        "candidate_verdicts": verdicts,
        "selected_band_label": selected_label,
        "final_opacity_points": opacity_points,
        "final_color_points": color_points,
        "final_image_path": final_image_path,
        "reasoning": selection_parsed.get("explanation", ""),
        "raw_response": raw_response,
    }


class IsovalueSpecialist(VisualizationSpecialist):
    agent_id = "isovalue_controller"

    def __init__(
        self,
        session: CameraReasoningSession,
        model: Optional[str] = None,
        num_windows: int = DEFAULT_NUM_WINDOWS,
        value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
        multi_angle: bool = True,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ):
        self.session = session
        self.model = model
        self.num_windows = num_windows
        self.value_range = value_range
        # Render 6 separate full-resolution views per window instead of 1 (default) -- all
        # 6 are sent together in ONE Stage-1 call per window (via extra_images), never
        # combined into a tile and never split into 6 separate calls.
        self.multi_angle = multi_angle
        # Called once with a normalized dict after selection completes (see
        # _handle_band_result) -- e.g. a notebook display callback. Optional.
        self.on_iteration = on_iteration

    def run_until_complete(
        self,
        goal: str,
        state: VisualizationState,
        constraints: Dict,
        success_criteria: List[str],
    ) -> AgentExecutionResult:
        if state.transfer_function is not None:
            opacity_points, color_points = state.transfer_function
            self.session.set_transfer_function(opacity_points, color_points)

        band_result = run_isovalue_band_selection(
            self.session,
            goal=goal,
            success_criteria=success_criteria,
            model=self.model,
            num_windows=self.num_windows,
            value_range=self.value_range,
            multi_angle=self.multi_angle,
        )

        if self.on_iteration:
            self._handle_band_result(band_result)

        state_patch = {
            "transfer_function": (self.session.opacity_points, self.session.color_points),
            "rendered_image_path": band_result["final_image_path"],
        }
        num_windows = len(band_result["bands"])
        iterations_used = num_windows + 1  # Stage 1: one call per window, Stage 2: one call

        if band_result["converged"] and band_result["selected_band_label"]:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.75,
                reason=f"Selected {band_result['selected_band_label']} from {num_windows} intensity "
                       f"window(s): {band_result['reasoning']}",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=iterations_used,
                artifacts=[band_result["final_image_path"]],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if num_windows else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.3,
            reason=f"Could not confidently match an intensity window to the goal: {band_result['reasoning']}",
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=[
                "adjust_opacity_transfer_function",
                "apply_clipping",
                "segment_target_structure",
            ],
            failure_type="capability_insufficient",
            iterations_used=iterations_used,
            artifacts=[band_result["final_image_path"]],
        )

    def _handle_band_result(self, band_result: dict) -> None:
        """Normalize one window selection into the shared on_iteration shape (see
        CameraSpecialist._handle_camera_iteration for the camera-side version of this same
        normalized dict). Reported as a single "iteration" (index 0) -- the two-stage
        pipeline underneath is still one logical selection, not a multi-round loop."""
        observation_by_label = {
            e["window"]: e["observation"]
            for e in band_result.get("band_evaluation", [])
            if isinstance(e, dict) and isinstance(e.get("window"), str)
        }

        view_images_by_label = band_result.get("view_images", {})

        candidates = []
        for window in band_result["bands"]:
            label = f"{WINDOW_LABEL_PREFIX}{window['low']}_{window['high']}"
            observation = observation_by_label.get(label)
            candidates.append({
                "label": label,
                "image_path": band_result["band_images"].get(label),
                "selected": label == band_result["selected_band_label"],
                "real_action": None,
                "observation": _summarize_observation(observation),
                "satisfies_goal": None,  # no per-candidate goal judgment in this pipeline -- see module docstring
                "range": f"{window['low']}-{window['high']} (peak {window['peak']})",
                "view_images": view_images_by_label.get(label, {}),  # {view_id: path}, for debugging/logging
            })

        self.on_iteration({
            "agent_id": self.agent_id,
            "iteration": 0,
            "current_image_path": band_result["final_image_path"],
            "candidates": candidates,
            "selected_label": band_result["selected_band_label"],
            "reasoning": band_result.get("reasoning") or "",
            "extra": {
                "selected_window": band_result["selected_band_label"],
                "converged": band_result["converged"],
            },
        })
