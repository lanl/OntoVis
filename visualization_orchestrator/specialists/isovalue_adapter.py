# =============================================================================
# TRANSFER FUNCTION AGENT — QUALITY REFINEMENT V2
# NEW IMPLEMENTATION: FIXED-WINDOW COARSE SEARCH + RANGE/OPACITY REFINEMENT
# VERSION CONSTANT: TF_REFINEMENT_IMPLEMENTATION_VERSION = "quality-refinement-v2"
#
# IMPORTANT: If this banner is missing, you are looking at the OLD implementation.
# This version must continue past coarse windowing and test discrete refinement actions.
# =============================================================================

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
TF_REFINEMENT_IMPLEMENTATION_VERSION = "quality-refinement-v2"

# Local discrete refinement defaults. These are intentionally module-level defaults so the
# existing constructor and callers do not need to change. Advanced callers may optionally
# override them through the existing `constraints` dictionary in run_until_complete.
DEFAULT_MAX_RANGE_ITERATIONS = 0
DEFAULT_MAX_OPACITY_ITERATIONS = 0
DEFAULT_OPACITY_STEP = 0.10
DEFAULT_RAMP_SHAPE_STEP = 0.50
MIN_RANGE_STEP = 1.0
MIN_WIDTH_STEP = 1.0
MIN_OPACITY_STEP = 0.025
MIN_RAMP_SHAPE_STEP = 0.25

CRITERION_STATUS_RANK = {"not_met": 0, "unknown": 1, "met": 2}
ARTIFACT_SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3}

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
        "abstain), uses that window as a coarse baseline, then locally refines range and "
        "opacity with discrete actions accepted only when they preserve every goal "
        "criterion and safely improve criteria or artifact quality."
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


# Goal-aware evaluation used by both the coarse fixed-window stage and the local refinement
# stages. The model is not allowed to rank candidates or choose an action. It only assigns a
# categorical status to each explicit success criterion from the already goal-blind visual
# observations. State transitions are then decided deterministically in Python.
CRITERION_EVALUATION_PROMPT_TEMPLATE = """Evaluate each candidate independently against each success criterion.

User goal:
{goal}

Criteria (use these opaque ids exactly):
{criteria_block}

Candidate observations:
{candidate_observations}

The observations were produced by a separate vision stage that did not know the goal.
Do not rank candidates, do not choose a candidate, and do not infer any visual evidence that
is absent from the supplied observations.

For every candidate and criterion, assign exactly one status:
- "met": the observation contains clear evidence that the criterion is satisfied.
- "not_met": the observation contains clear evidence that the criterion is violated or the
  required feature is visibly absent.
- "unknown": the supplied observation is insufficient or ambiguous.

Rules:
- Judge every candidate on its own evidence, never relative to another candidate.
- "unknown" is not the same as "met".
- Use only criterion ids and candidate ids supplied below.
- Evidence must quote or closely paraphrase the supplied observation, not the image itself.
- Do not provide an overall score, preference, decision, or selected candidate.

Respond with STRICT JSON ONLY:
{{
  "candidate_assessments": {{
    "<candidate_N>": {{
      "criteria": {{
        "<criterion_N>": {{
          "status": "met" | "not_met" | "unknown",
          "evidence": ["<supporting or contradictory observation text>"]
        }}
      }}
    }}
  }}
}}

Include every supplied candidate and every supplied criterion exactly once.
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalized_tf_state(
    state: dict,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
) -> dict:
    """Return a valid, canonical transfer-function search state.

    Existing fixed-window dictionaries remain valid because every refinement parameter has a
    default. Values are kept numeric and rounded only for stable cache keys/logging.
    """
    min_value, max_value = value_range
    low = _clamp(float(state["low"]), min_value, max_value)
    high = _clamp(float(state["high"]), min_value, max_value)
    if high <= low:
        high = min(max_value, low + 1.0)
        if high <= low:
            low = max(min_value, high - 1.0)

    peak = (low + high) / 2.0
    peak_opacity = _clamp(
        float(state.get("peak_opacity", DEFAULT_PEAK_OPACITY)), 0.01, 0.95
    )
    sustain_opacity = _clamp(
        float(state.get("sustain_opacity", DEFAULT_SUSTAIN_OPACITY)), 0.01, 0.98
    )
    sustain_opacity = max(peak_opacity, sustain_opacity)
    ramp_shape = _clamp(float(state.get("ramp_shape", 0.0)), -1.0, 1.0)
    transition_fraction = _clamp(
        float(state.get("transition_fraction", 0.5)), 0.0, 2.0
    )

    return {
        "low": round(low, 4),
        "high": round(high, 4),
        "peak": round(peak, 4),
        "peak_opacity": round(peak_opacity, 4),
        "sustain_opacity": round(sustain_opacity, 4),
        "ramp_shape": round(ramp_shape, 4),
        "transition_fraction": round(transition_fraction, 4),
    }


def _tf_state_key(state: dict) -> Tuple[float, ...]:
    canonical = _normalized_tf_state(state, (float("-inf"), float("inf")))
    return (
        canonical["low"],
        canonical["high"],
        canonical["peak_opacity"],
        canonical["sustain_opacity"],
        canonical["ramp_shape"],
        canonical["transition_fraction"],
    )


def build_opacity_ramp_for_band(
    band: dict,
    min_value: float = DEFAULT_VALUE_RANGE[0],
    max_value: float = DEFAULT_VALUE_RANGE[1],
    peak_opacity: float = DEFAULT_PEAK_OPACITY,
    sustain_opacity: float = DEFAULT_SUSTAIN_OPACITY,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float, float]]]:
    """Build a deterministic ramp from a fixed window or a refined TF state.

    Backward compatibility is preserved: callers may still pass only ``low/high/peak``.
    Local refinement adds optional ``peak_opacity``, ``sustain_opacity``, ``ramp_shape``, and
    ``transition_fraction`` fields. ``ramp_shape`` is represented using one extra linear
    knee between low and peak: positive values rise earlier/sharper, negative values rise
    later/softer.
    """
    state = _normalized_tf_state(
        {
            **band,
            "peak_opacity": band.get("peak_opacity", peak_opacity),
            "sustain_opacity": band.get("sustain_opacity", sustain_opacity),
        },
        (min_value, max_value),
    )
    low, high, peak = state["low"], state["high"], state["peak"]
    peak_opacity = state["peak_opacity"]
    sustain_opacity = state["sustain_opacity"]
    ramp_shape = state["ramp_shape"]

    transition = max(1.0, (peak - low) * state["transition_fraction"])
    ramp_start = max(min_value, low - transition)
    low_opacity = min(0.05, peak_opacity)
    knee = low + 0.5 * (peak - low)
    knee_fraction = _clamp(0.5 + 0.35 * ramp_shape, 0.10, 0.90)
    knee_opacity = low_opacity + (peak_opacity - low_opacity) * knee_fraction

    opacity_points = _strictly_increasing([
        (min_value, 0.0),
        (ramp_start, 0.0),
        (low, low_opacity),
        (knee, knee_opacity),
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
    state = _normalized_tf_state(band, (min_value, max_value))
    low, high, peak = state["low"], state["high"], state["peak"]
    transition = max(1.0, (peak - low) * state["transition_fraction"])
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


def _safe_file_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _render_tf_candidate_previews(
    session: CameraReasoningSession,
    candidates: List[dict],
    output_dir: str,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    multi_angle: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Render arbitrary refined TF states with the same goal-blind grayscale treatment.

    ``candidates`` entries contain ``candidate_id`` and ``state``. The session's previous
    rendering mode, transfer function, and camera state are restored even on failure.
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

    rendered: Dict[str, Dict[str, str]] = {}
    try:
        for entry in candidates:
            candidate_id = entry["candidate_id"]
            state = entry["state"]
            opacity_points, color_points = build_evaluation_ramp_for_window(
                state, min_value=value_range[0], max_value=value_range[1]
            )
            session.set_transfer_function(opacity_points, color_points)
            view_images: Dict[str, str] = {}
            file_prefix = _safe_file_component(entry.get("file_label") or candidate_id)
            for view_index, action in enumerate(actions_to_render):
                view_id = f"view_{view_index}"
                set_camera_state(camera, original_camera_state)
                if action is not None:
                    apply_action(action, camera, session._renderer)
                else:
                    session._renderer.ResetCameraClippingRange()
                image_path = str(out_dir / f"{file_prefix}_{view_id}.png")
                save_screenshot(session._render_window, image_path)
                view_images[view_id] = image_path
            rendered[candidate_id] = view_images
    finally:
        if original_use_volume_rendering:
            session.set_transfer_function(original_opacity_points, original_color_points)
        else:
            session.set_isovalue(original_isovalue)
        set_camera_state(camera, original_camera_state)
        session._renderer.ResetCameraClippingRange()

    return rendered


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


def _evaluate_candidate_criteria(
    goal: str,
    success_criteria: List[str],
    candidate_observations: List[dict],
    model: Optional[str],
) -> Tuple[Optional[dict], str]:
    """Text-only categorical criterion assessment; never asks the LLM to choose."""
    criteria_block = "\n".join(
        f"criterion_{i}: {criterion}" for i, criterion in enumerate(success_criteria)
    )
    observation_blocks = []
    for entry in candidate_observations:
        observation = entry.get("observation")
        if observation is None:
            observation = {"error": "no observation available"}
        observation_blocks.append(
            f"[{entry['candidate_id']}]\n{json.dumps(observation, indent=2)}"
        )

    prompt = CRITERION_EVALUATION_PROMPT_TEMPLATE.format(
        goal=goal,
        criteria_block=criteria_block,
        candidate_observations="\n\n".join(observation_blocks),
    )
    response_text = ask_chatgpt(prompt=prompt, model=model)
    return extract_json_object(response_text), response_text


def _normalize_assessments(
    parsed: Optional[dict],
    candidate_ids: List[str],
    success_criteria: List[str],
) -> Dict[str, dict]:
    """Normalize malformed/missing model fields to conservative ``unknown`` statuses."""
    raw_assessments = (parsed or {}).get("candidate_assessments") or {}
    normalized: Dict[str, dict] = {}
    for candidate_id in candidate_ids:
        raw_candidate = raw_assessments.get(candidate_id) or {}
        raw_criteria = raw_candidate.get("criteria") or {}
        criteria: Dict[str, dict] = {}
        for index, criterion_text in enumerate(success_criteria):
            criterion_id = f"criterion_{index}"
            raw_item = raw_criteria.get(criterion_id) or {}
            status = raw_item.get("status")
            if status not in CRITERION_STATUS_RANK:
                status = "unknown"
            evidence = raw_item.get("evidence")
            if not isinstance(evidence, list):
                evidence = []
            criteria[criterion_id] = {
                "criterion": criterion_text,
                "status": status,
                "evidence": [str(item) for item in evidence if item is not None],
            }
        normalized[candidate_id] = {"criteria": criteria}
    return normalized


def _criterion_status_map(assessment: dict) -> Dict[str, str]:
    return {
        criterion_id: item.get("status", "unknown")
        for criterion_id, item in (assessment.get("criteria") or {}).items()
    }


def _all_criteria_met(assessment: dict) -> bool:
    statuses = list(_criterion_status_map(assessment).values())
    return bool(statuses) and all(status == "met" for status in statuses)


def _candidate_verdicts_from_assessments(
    assessments: Dict[str, dict],
) -> Dict[str, dict]:
    """Maintain the old public ``candidate_verdicts`` return field."""
    verdicts: Dict[str, dict] = {}
    for candidate_id, assessment in assessments.items():
        criteria = assessment.get("criteria") or {}
        met = [item["criterion"] for item in criteria.values() if item["status"] == "met"]
        not_met = [
            item["criterion"] for item in criteria.values() if item["status"] != "met"
        ]
        verdicts[candidate_id] = {
            "verdict": "passes" if criteria and not not_met else "fails",
            "criteria_met": met,
            "criteria_not_met": not_met,
            "criterion_statuses": {
                item["criterion"]: item["status"] for item in criteria.values()
            },
        }
    return verdicts


ARTIFACT_FIELDS = ("isolated_fragments", "surface_noise", "occlusion")


def _artifact_severity_map(observation: Optional[dict]) -> Dict[str, int]:
    """Return comparable artifact severities; lower is better.

    Missing or malformed fields are treated conservatively as ``moderate``. A completely
    missing observation is handled separately by candidate selection and is never accepted
    as a refinement.
    """
    artifacts = (observation or {}).get("noise_and_artifacts") or {}
    severities: Dict[str, int] = {}
    for field in ARTIFACT_FIELDS:
        value = str(artifacts.get(field, "moderate")).strip().lower()
        severities[field] = ARTIFACT_SEVERITY_RANK.get(
            value, ARTIFACT_SEVERITY_RANK["moderate"]
        )
    return severities


def _artifact_penalty(observation: Optional[dict]) -> int:
    return sum(_artifact_severity_map(observation).values())


def _assessment_quality_tuple(
    assessment: dict,
    observation: Optional[dict],
    candidate_index: int,
) -> tuple:
    statuses = list(_criterion_status_map(assessment).values())
    score = sum(CRITERION_STATUS_RANK.get(status, 1) for status in statuses)
    met_count = sum(status == "met" for status in statuses)
    not_met_count = sum(status == "not_met" for status in statuses)
    return (score, met_count, -not_met_count, -_artifact_penalty(observation), -candidate_index)


def _select_coarse_baseline(
    candidate_ids: List[str],
    assessments: Dict[str, dict],
    observations_by_id: Dict[str, Optional[dict]],
) -> Optional[str]:
    """Choose a deterministic coarse starting point without asking the LLM to rank.

    At least one criterion must be ``met`` or ``unknown``. If every candidate is explicitly
    ``not_met`` on every criterion, the fixed sweep provides no defensible local-search
    anchor and the agent abstains.
    """
    viable = []
    for index, candidate_id in enumerate(candidate_ids):
        statuses = list(_criterion_status_map(assessments[candidate_id]).values())
        if observations_by_id.get(candidate_id) is None:
            continue
        if statuses and all(status == "not_met" for status in statuses):
            continue
        viable.append(
            (
                _assessment_quality_tuple(
                    assessments[candidate_id], observations_by_id[candidate_id], index
                ),
                candidate_id,
            )
        )
    return max(viable)[1] if viable else None


def _refinement_improvement(
    candidate_assessment: dict,
    current_assessment: dict,
    candidate_observation: Optional[dict],
    current_observation: Optional[dict],
) -> Optional[Tuple[int, int, int]]:
    """Measure a safe local refinement, or return ``None`` when it is not acceptable.

    Success criteria are hard constraints: a refinement may never lower any criterion's
    categorical rank. Artifact dimensions are also compared independently, so a candidate
    cannot hide worse occlusion behind lower surface noise. At least one criterion or
    artifact dimension must improve strictly.

    Returns ``(total_gain, criterion_gain, artifact_gain)`` for deterministic ranking.
    """
    if candidate_observation is None:
        return None

    candidate_statuses = _criterion_status_map(candidate_assessment)
    current_statuses = _criterion_status_map(current_assessment)
    if candidate_statuses.keys() != current_statuses.keys():
        return None

    criterion_gain = 0
    for criterion_id, current_status in current_statuses.items():
        candidate_rank = CRITERION_STATUS_RANK[candidate_statuses[criterion_id]]
        current_rank = CRITERION_STATUS_RANK[current_status]
        if candidate_rank < current_rank:
            return None
        criterion_gain += candidate_rank - current_rank

    candidate_artifacts = _artifact_severity_map(candidate_observation)
    current_artifacts = _artifact_severity_map(current_observation)
    artifact_gain = 0
    for field in ARTIFACT_FIELDS:
        candidate_severity = candidate_artifacts[field]
        current_severity = current_artifacts[field]
        if candidate_severity > current_severity:
            return None
        artifact_gain += current_severity - candidate_severity

    total_gain = criterion_gain + artifact_gain
    if total_gain <= 0:
        return None
    return total_gain, criterion_gain, artifact_gain


def _parameter_distance(a: dict, b: dict) -> float:
    keys = ("low", "high", "peak_opacity", "sustain_opacity", "ramp_shape")
    return sum(abs(float(a[key]) - float(b[key])) for key in keys)


def _choose_dominating_candidate(
    candidates: List[dict],
    assessments_by_key: Dict[Tuple[float, ...], dict],
    observations_by_key: Dict[Tuple[float, ...], Optional[dict]],
    current_state: dict,
) -> Optional[dict]:
    current_key = _tf_state_key(current_state)
    current_assessment = assessments_by_key[current_key]
    current_observation = observations_by_key.get(current_key)
    ranked = []
    for index, candidate in enumerate(candidates):
        key = _tf_state_key(candidate["state"])
        if key == current_key:
            continue
        assessment = assessments_by_key[key]
        improvement = _refinement_improvement(
            assessment,
            current_assessment,
            observations_by_key.get(key),
            current_observation,
        )
        if improvement is None:
            continue
        total_gain, criterion_gain, artifact_gain = improvement
        quality = _assessment_quality_tuple(
            assessment, observations_by_key.get(key), index
        )
        ranking_key = (
            total_gain,
            criterion_gain,
            artifact_gain,
            quality,
            -_parameter_distance(candidate["state"], current_state),
            -index,
        )
        ranked.append((ranking_key, candidate))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _replace_range(state: dict, low: float, high: float, value_range: Tuple[float, float]) -> dict:
    return _normalized_tf_state({**state, "low": low, "high": high}, value_range)


def _generate_range_candidates(
    current_state: dict,
    shift_step: float,
    width_step: float,
    value_range: Tuple[float, float],
) -> List[dict]:
    current = _normalized_tf_state(current_state, value_range)
    low, high = current["low"], current["high"]
    min_value, max_value = value_range
    width = high - low

    proposals = [
        ("CURRENT", current),
        ("SHIFT_LOWER", _replace_range(current, max(min_value, low - shift_step), max(min_value, low - shift_step) + width, value_range)),
        ("SHIFT_HIGHER", _replace_range(current, min(max_value - width, low + shift_step), min(max_value - width, low + shift_step) + width, value_range)),
        ("EXPAND_RANGE", _replace_range(current, max(min_value, low - width_step), min(max_value, high + width_step), value_range)),
    ]
    if high - low > 2.0 * width_step + 1.0:
        proposals.append(
            ("NARROW_RANGE", _replace_range(current, low + width_step, high - width_step, value_range))
        )

    deduplicated = []
    seen = set()
    for action, state in proposals:
        key = _tf_state_key(state)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append({"action": action, "state": state})
    return deduplicated


def _generate_opacity_candidates(
    current_state: dict,
    opacity_step: float,
    ramp_shape_step: float,
    value_range: Tuple[float, float],
) -> List[dict]:
    current = _normalized_tf_state(current_state, value_range)
    proposals = [
        ("CURRENT", current),
        (
            "INCREASE_OPACITY",
            _normalized_tf_state(
                {
                    **current,
                    "peak_opacity": current["peak_opacity"] + opacity_step,
                    "sustain_opacity": current["sustain_opacity"] + opacity_step,
                },
                value_range,
            ),
        ),
        (
            "DECREASE_OPACITY",
            _normalized_tf_state(
                {
                    **current,
                    "peak_opacity": current["peak_opacity"] - opacity_step,
                    "sustain_opacity": current["sustain_opacity"] - opacity_step,
                },
                value_range,
            ),
        ),
        (
            "SHARPEN_RAMP",
            _normalized_tf_state(
                {**current, "ramp_shape": current["ramp_shape"] + ramp_shape_step},
                value_range,
            ),
        ),
        (
            "SOFTEN_RAMP",
            _normalized_tf_state(
                {**current, "ramp_shape": current["ramp_shape"] - ramp_shape_step},
                value_range,
            ),
        ),
    ]

    deduplicated = []
    seen = set()
    for action, state in proposals:
        key = _tf_state_key(state)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append({"action": action, "state": state})
    return deduplicated


def _evaluate_refinement_states(
    session: CameraReasoningSession,
    candidates: List[dict],
    goal: str,
    success_criteria: List[str],
    model: Optional[str],
    output_dir: str,
    value_range: Tuple[float, float],
    multi_angle: bool,
    observation_cache: Dict[Tuple[float, ...], Optional[dict]],
    assessment_cache: Dict[Tuple[float, ...], dict],
    render_cache: Dict[Tuple[float, ...], Dict[str, str]],
    raw_responses: List[str],
    phase: str,
    iteration: int,
) -> int:
    """Populate render/observation/assessment caches and return the LLM-call count.

    Render paths are cached independently from observations. This lets the notebook callback
    display every range/opacity candidate even when a state's blind observation was reused
    from an earlier iteration.
    """
    llm_calls = 0
    local_id_to_key: Dict[str, Tuple[float, ...]] = {}
    missing_renders: List[dict] = []

    for index, candidate in enumerate(candidates):
        key = _tf_state_key(candidate["state"])
        candidate_id = f"candidate_{index}"
        candidate["candidate_id"] = candidate_id
        candidate["file_label"] = f"{phase}_{iteration}_{candidate['action']}_{index}"
        local_id_to_key[candidate_id] = key
        if key not in render_cache:
            missing_renders.append(candidate)

    if missing_renders:
        rendered = _render_tf_candidate_previews(
            session,
            missing_renders,
            output_dir,
            value_range=value_range,
            multi_angle=multi_angle,
        )
        for candidate in missing_renders:
            key = _tf_state_key(candidate["state"])
            render_cache[key] = rendered[candidate["candidate_id"]]

    for candidate in candidates:
        key = _tf_state_key(candidate["state"])
        if key in observation_cache:
            continue
        parsed, response_text = _observe_window_blind(render_cache[key], model)
        observation_cache[key] = parsed
        raw_responses.append(
            f"=== {phase} iteration {iteration} {candidate['action']} blind observation ===\n"
            f"{response_text}"
        )
        llm_calls += 1

    missing_assessments = []
    for candidate in candidates:
        key = _tf_state_key(candidate["state"])
        if key not in assessment_cache:
            missing_assessments.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "observation": observation_cache.get(key),
                }
            )

    if missing_assessments:
        parsed, response_text = _evaluate_candidate_criteria(
            goal, success_criteria, missing_assessments, model
        )
        normalized = _normalize_assessments(
            parsed,
            [entry["candidate_id"] for entry in missing_assessments],
            success_criteria,
        )
        for entry in missing_assessments:
            candidate_id = entry["candidate_id"]
            assessment_cache[local_id_to_key[candidate_id]] = normalized[candidate_id]
        raw_responses.append(
            f"=== {phase} iteration {iteration} criterion evaluation ===\n{response_text}"
        )
        llm_calls += 1

    return llm_calls


def _refinement_candidate_snapshot(
    candidates: List[dict],
    selected_state: dict,
    observation_cache: Dict[Tuple[float, ...], Optional[dict]],
    assessment_cache: Dict[Tuple[float, ...], dict],
    render_cache: Dict[Tuple[float, ...], Dict[str, str]],
) -> List[dict]:
    """Build the existing notebook callback shape for one refinement iteration."""
    selected_key = _tf_state_key(selected_state)
    snapshots: List[dict] = []
    for candidate in candidates:
        state = candidate["state"]
        key = _tf_state_key(state)
        observation = observation_cache.get(key)
        assessment = assessment_cache[key]
        views = render_cache.get(key, {})
        artifacts = _artifact_severity_map(observation)
        snapshots.append(
            {
                "label": candidate["action"],
                "image_path": next(iter(views.values()), None),
                "selected": key == selected_key,
                "real_action": candidate["action"],
                "observation": _summarize_observation(observation),
                "satisfies_goal": _all_criteria_met(assessment),
                "range": (
                    f"{state['low']:g}-{state['high']:g}; "
                    f"peak_opacity={state['peak_opacity']:.3f}; "
                    f"sustain_opacity={state['sustain_opacity']:.3f}; "
                    f"ramp_shape={state['ramp_shape']:.3f}"
                ),
                "view_images": views,
                "criterion_statuses": _criterion_status_map(assessment),
                "artifact_severities": artifacts,
                "tf_state": state,
            }
        )
    return snapshots


def _representative_refinement_image(
    state: dict,
    render_cache: Dict[Tuple[float, ...], Dict[str, str]],
) -> Optional[str]:
    views = render_cache.get(_tf_state_key(state), {})
    return next(iter(views.values()), None)


def _run_local_refinement(
    session: CameraReasoningSession,
    initial_state: dict,
    initial_observation: Optional[dict],
    initial_assessment: dict,
    goal: str,
    success_criteria: List[str],
    model: Optional[str],
    output_dir: str,
    value_range: Tuple[float, float],
    multi_angle: bool,
    max_range_iterations: int,
    max_opacity_iterations: int,
) -> dict:
    """Run range then opacity search using criterion-and-artifact Pareto dominance.

    Meeting every success criterion marks the current visualization as valid, but does not
    terminate refinement. Range and opacity phases still test local actions for lower noise,
    fewer isolated fragments, or lower occlusion without sacrificing any goal criterion.
    """
    current_state = _normalized_tf_state(initial_state, value_range)
    initial_key = _tf_state_key(current_state)
    observation_cache: Dict[Tuple[float, ...], Optional[dict]] = {
        initial_key: initial_observation
    }
    assessment_cache: Dict[Tuple[float, ...], dict] = {initial_key: initial_assessment}
    render_cache: Dict[Tuple[float, ...], Dict[str, str]] = {}
    raw_responses: List[str] = []
    history: List[dict] = []
    llm_calls = 0

    initial_width = max(1.0, current_state["high"] - current_state["low"])
    shift_step = max(MIN_RANGE_STEP, initial_width / 4.0)
    width_step = max(MIN_WIDTH_STEP, initial_width / 4.0)

    for iteration in range(max(0, max_range_iterations)):
        candidates = _generate_range_candidates(
            current_state, shift_step, width_step, value_range
        )
        llm_calls += _evaluate_refinement_states(
            session,
            candidates,
            goal,
            success_criteria,
            model,
            str(Path(output_dir) / "range"),
            value_range,
            multi_angle,
            observation_cache,
            assessment_cache,
            render_cache,
            raw_responses,
            "range",
            iteration,
        )
        selected = _choose_dominating_candidate(
            candidates, assessment_cache, observation_cache, current_state
        )
        if selected is not None:
            previous = current_state
            current_state = selected["state"]
            assessment = assessment_cache[_tf_state_key(current_state)]
            history.append(
                {
                    "phase": "range",
                    "iteration": iteration,
                    "action": selected["action"],
                    "accepted": True,
                    "before": previous,
                    "after": current_state,
                    "criterion_statuses": _criterion_status_map(assessment),
                    "artifact_severities": _artifact_severity_map(
                        observation_cache.get(_tf_state_key(current_state))
                    ),
                    "current_image_path": _representative_refinement_image(
                        current_state, render_cache
                    ),
                    "candidates": _refinement_candidate_snapshot(
                        candidates,
                        current_state,
                        observation_cache,
                        assessment_cache,
                        render_cache,
                    ),
                }
            )
            continue

        next_shift = max(MIN_RANGE_STEP, shift_step / 2.0)
        next_width = max(MIN_WIDTH_STEP, width_step / 2.0)
        history.append(
            {
                "phase": "range",
                "iteration": iteration,
                "action": "REDUCE_STEP" if (next_shift, next_width) != (shift_step, width_step) else "STOP_PHASE",
                "accepted": False,
                "before": current_state,
                "after": current_state,
                "shift_step": shift_step,
                "width_step": width_step,
                "criterion_statuses": _criterion_status_map(
                    assessment_cache[_tf_state_key(current_state)]
                ),
                "artifact_severities": _artifact_severity_map(
                    observation_cache.get(_tf_state_key(current_state))
                ),
                "current_image_path": _representative_refinement_image(
                    current_state, render_cache
                ),
                "candidates": _refinement_candidate_snapshot(
                    candidates,
                    current_state,
                    observation_cache,
                    assessment_cache,
                    render_cache,
                ),
            }
        )
        if (next_shift, next_width) == (shift_step, width_step):
            break
        shift_step, width_step = next_shift, next_width

    opacity_step = DEFAULT_OPACITY_STEP
    ramp_shape_step = DEFAULT_RAMP_SHAPE_STEP
    for iteration in range(max(0, max_opacity_iterations)):
        candidates = _generate_opacity_candidates(
            current_state, opacity_step, ramp_shape_step, value_range
        )
        llm_calls += _evaluate_refinement_states(
            session,
            candidates,
            goal,
            success_criteria,
            model,
            str(Path(output_dir) / "opacity"),
            value_range,
            multi_angle,
            observation_cache,
            assessment_cache,
            render_cache,
            raw_responses,
            "opacity",
            iteration,
        )
        selected = _choose_dominating_candidate(
            candidates, assessment_cache, observation_cache, current_state
        )
        if selected is not None:
            previous = current_state
            current_state = selected["state"]
            assessment = assessment_cache[_tf_state_key(current_state)]
            history.append(
                {
                    "phase": "opacity",
                    "iteration": iteration,
                    "action": selected["action"],
                    "accepted": True,
                    "before": previous,
                    "after": current_state,
                    "criterion_statuses": _criterion_status_map(assessment),
                    "artifact_severities": _artifact_severity_map(
                        observation_cache.get(_tf_state_key(current_state))
                    ),
                    "current_image_path": _representative_refinement_image(
                        current_state, render_cache
                    ),
                    "candidates": _refinement_candidate_snapshot(
                        candidates,
                        current_state,
                        observation_cache,
                        assessment_cache,
                        render_cache,
                    ),
                }
            )
            continue

        next_opacity = max(MIN_OPACITY_STEP, opacity_step / 2.0)
        next_ramp = max(MIN_RAMP_SHAPE_STEP, ramp_shape_step / 2.0)
        history.append(
            {
                "phase": "opacity",
                "iteration": iteration,
                "action": "REDUCE_STEP" if (next_opacity, next_ramp) != (opacity_step, ramp_shape_step) else "STOP_PHASE",
                "accepted": False,
                "before": current_state,
                "after": current_state,
                "opacity_step": opacity_step,
                "ramp_shape_step": ramp_shape_step,
                "criterion_statuses": _criterion_status_map(
                    assessment_cache[_tf_state_key(current_state)]
                ),
                "artifact_severities": _artifact_severity_map(
                    observation_cache.get(_tf_state_key(current_state))
                ),
                "current_image_path": _representative_refinement_image(
                    current_state, render_cache
                ),
                "candidates": _refinement_candidate_snapshot(
                    candidates,
                    current_state,
                    observation_cache,
                    assessment_cache,
                    render_cache,
                ),
            }
        )
        if (next_opacity, next_ramp) == (opacity_step, ramp_shape_step):
            break
        opacity_step, ramp_shape_step = next_opacity, next_ramp

    final_assessment = assessment_cache[_tf_state_key(current_state)]
    return {
        "state": current_state,
        "assessment": final_assessment,
        "converged": _all_criteria_met(final_assessment),
        "history": history,
        "raw_responses": raw_responses,
        "llm_calls": llm_calls,
        "stop_reason": (
            "Local range and opacity refinement finished; final success criteria are met."
            if _all_criteria_met(final_assessment)
            else "Local range and opacity refinement finished, but some success criteria remain unmet."
        ),
    }


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
    max_range_iterations: int = DEFAULT_MAX_RANGE_ITERATIONS,
    max_opacity_iterations: int = DEFAULT_MAX_OPACITY_ITERATIONS,
) -> dict:
    """Coarse fixed-window initialization followed by in-file discrete TF refinement.

    The public name and original arguments are retained. Existing callers therefore need no
    changes. With non-empty success criteria, the LLM only performs (1) goal-blind visual
    observation and (2) categorical per-criterion assessment. Python selects the coarse
    baseline, accepts only criterion-and-artifact Pareto improvements, reduces step sizes
    when no action improves safely, and runs both range and opacity phases to their step or
    iteration limits even when the coarse baseline already satisfies every success criterion.
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

    raw_responses: List[str] = []
    candidate_observations: List[dict] = []
    llm_calls = 0
    for candidate_id, label in zip(candidate_ids, window_labels):
        parsed, response_text = _observe_window_blind(window_view_images[label], model)
        candidate_observations.append({"candidate_id": candidate_id, "observation": parsed})
        raw_responses.append(
            f"=== {candidate_id} coarse blind observation, views={list(window_view_images[label])} ===\n"
            f"{response_text}"
        )
        llm_calls += 1

    observations_by_id = {
        entry["candidate_id"]: entry["observation"] for entry in candidate_observations
    }
    band_evaluation = [
        {"window": candidate_to_label[candidate_id], "observation": observations_by_id[candidate_id]}
        for candidate_id in candidate_ids
    ]

    # Preserve legacy behavior when no explicit criteria are available: use the original
    # strict selector and skip local refinement because deterministic dominance is undefined.
    if not success_criteria:
        selection_parsed, selection_raw = _select_candidate_from_observations(
            goal, success_criteria, candidate_observations, model
        )
        raw_responses.append(f"=== legacy coarse selection ===\n{selection_raw}")
        llm_calls += 1
        selected_candidate_id = (selection_parsed or {}).get("selected_candidate")
        decision = (selection_parsed or {}).get("decision")
        verdicts = (selection_parsed or {}).get("candidate_verdicts") or {}
        if decision != "selected" or selected_candidate_id not in candidate_to_window:
            final_image_path = session.render_and_save()
            return {
                "converged": False,
                "bands": windows,
                "band_images": window_images,
                "view_images": window_view_images,
                "view_action_map": view_action_map(multi_angle),
                "band_evaluation": band_evaluation,
                "candidate_verdicts": verdicts,
                "selected_band_label": None,
                "selected_tf_state": None,
                "refinement_history": [],
                "final_opacity_points": session.opacity_points,
                "final_color_points": session.color_points,
                "final_image_path": final_image_path,
                "reasoning": (selection_parsed or {}).get("explanation") or "No candidate selected.",
                "raw_response": "\n\n".join(raw_responses),
                "llm_calls": llm_calls,
            }
        initial_state = _normalized_tf_state(candidate_to_window[selected_candidate_id], value_range)
        opacity_points, color_points = build_opacity_ramp_for_band(
            initial_state, min_value=value_range[0], max_value=value_range[1]
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
            "selected_band_label": candidate_to_label[selected_candidate_id],
            "selected_tf_state": initial_state,
            "refinement_history": [],
            "final_opacity_points": opacity_points,
            "final_color_points": color_points,
            "final_image_path": final_image_path,
            "reasoning": (selection_parsed or {}).get("explanation", ""),
            "raw_response": "\n\n".join(raw_responses),
            "llm_calls": llm_calls,
        }

    evaluation_parsed, evaluation_raw = _evaluate_candidate_criteria(
        goal, success_criteria, candidate_observations, model
    )
    raw_responses.append(f"=== coarse criterion evaluation ===\n{evaluation_raw}")
    llm_calls += 1
    assessments = _normalize_assessments(
        evaluation_parsed, candidate_ids, success_criteria
    )
    verdicts = _candidate_verdicts_from_assessments(assessments)
    selected_candidate_id = _select_coarse_baseline(
        candidate_ids, assessments, observations_by_id
    )

    if selected_candidate_id is None:
        final_image_path = session.render_and_save()
        return {
            "converged": False,
            "bands": windows,
            "band_images": window_images,
            "view_images": window_view_images,
            "view_action_map": view_action_map(multi_angle),
            "band_evaluation": band_evaluation,
            "candidate_verdicts": verdicts,
            "selected_band_label": None,
            "selected_tf_state": None,
            "refinement_history": [],
            "final_opacity_points": session.opacity_points,
            "final_color_points": session.color_points,
            "final_image_path": final_image_path,
            "reasoning": (
                "Every coarse window was explicitly not_met for every success criterion; "
                "no defensible local-search anchor was available."
            ),
            "raw_response": "\n\n".join(raw_responses),
            "llm_calls": llm_calls,
        }

    selected_label = candidate_to_label[selected_candidate_id]
    initial_state = _normalized_tf_state(candidate_to_window[selected_candidate_id], value_range)
    initial_assessment = assessments[selected_candidate_id]
    initial_observation = observations_by_id[selected_candidate_id]

    refinement = _run_local_refinement(
        session=session,
        initial_state=initial_state,
        initial_observation=initial_observation,
        initial_assessment=initial_assessment,
        goal=goal,
        success_criteria=success_criteria,
        model=model,
        output_dir=str(Path(render_dir) / "refinement"),
        value_range=value_range,
        multi_angle=multi_angle,
        max_range_iterations=max_range_iterations,
        max_opacity_iterations=max_opacity_iterations,
    )
    llm_calls += refinement["llm_calls"]
    raw_responses.extend(refinement["raw_responses"])

    final_state = refinement["state"]
    opacity_points, color_points = build_opacity_ramp_for_band(
        final_state, min_value=value_range[0], max_value=value_range[1]
    )
    session.set_transfer_function(opacity_points, color_points)
    final_image_path = session.render_and_save()

    accepted_actions = [
        step["action"] for step in refinement["history"] if step.get("accepted")
    ]
    action_text = ", ".join(accepted_actions) if accepted_actions else "no local action accepted"
    reasoning = (
        f"Coarse baseline {selected_label}; {action_text}. "
        f"{refinement['stop_reason']}"
    )

    return {
        "converged": refinement["converged"],
        "bands": windows,
        "band_images": window_images,
        "view_images": window_view_images,
        "view_action_map": view_action_map(multi_angle),
        "band_evaluation": band_evaluation,
        "candidate_verdicts": verdicts,
        "selected_band_label": selected_label,
        "selected_tf_state": final_state,
        "final_criterion_statuses": _criterion_status_map(refinement["assessment"]),
        "refinement_history": refinement["history"],
        "final_opacity_points": opacity_points,
        "final_color_points": color_points,
        "final_image_path": final_image_path,
        "reasoning": reasoning,
        "raw_response": "\n\n".join(raw_responses),
        "llm_calls": llm_calls,
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
            max_range_iterations=int(
                constraints.get("tf_max_range_iterations", DEFAULT_MAX_RANGE_ITERATIONS)
            ),
            max_opacity_iterations=int(
                constraints.get("tf_max_opacity_iterations", DEFAULT_MAX_OPACITY_ITERATIONS)
            ),
        )

        if self.on_iteration:
            self._handle_band_result(band_result)

        state_patch = {
            "transfer_function": (self.session.opacity_points, self.session.color_points),
            "rendered_image_path": band_result["final_image_path"],
        }
        num_windows = len(band_result["bands"])
        iterations_used = int(band_result.get("llm_calls", num_windows + 1))

        if band_result["converged"] and band_result["selected_band_label"]:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.75,
                reason=(
                    f"Initialized from {band_result['selected_band_label']} and refined the "
                    f"transfer function: {band_result['reasoning']}"
                ),
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
            reason=(
                "Transfer-function search produced the best non-worsening state but did not "
                f"satisfy every criterion: {band_result['reasoning']}"
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
            artifacts=[band_result["final_image_path"]],
        )

    def _handle_band_result(self, band_result: dict) -> None:
        """Emit coarse-window and local-refinement iterations through the existing callback.

        No notebook changes are required: refinement candidates use the same candidate dict
        shape already handled by the demo callback (`label`, `range`, `observation`,
        `view_images`, and `selected`).
        """
        observation_by_label = {
            e["window"]: e["observation"]
            for e in band_result.get("band_evaluation", [])
            if isinstance(e, dict) and isinstance(e.get("window"), str)
        }
        view_images_by_label = band_result.get("view_images", {})

        coarse_candidates = []
        for window in band_result["bands"]:
            label = f"{WINDOW_LABEL_PREFIX}{window['low']}_{window['high']}"
            observation = observation_by_label.get(label)
            coarse_candidates.append(
                {
                    "label": label,
                    "image_path": band_result["band_images"].get(label),
                    "selected": label == band_result["selected_band_label"],
                    "real_action": None,
                    "observation": _summarize_observation(observation),
                    "satisfies_goal": None,
                    "range": f"{window['low']}-{window['high']} (peak {window['peak']})",
                    "view_images": view_images_by_label.get(label, {}),
                }
            )

        selected_label = band_result.get("selected_band_label")
        coarse_image = band_result.get("band_images", {}).get(selected_label)
        self.on_iteration(
            {
                "agent_id": self.agent_id,
                "iteration": 0,
                "current_image_path": coarse_image or band_result["final_image_path"],
                "candidates": coarse_candidates,
                "selected_label": selected_label,
                "reasoning": (
                    f"Selected {selected_label} as the coarse transfer-function baseline."
                    if selected_label
                    else band_result.get("reasoning") or "No coarse baseline selected."
                ),
                "extra": {
                    "stage": "coarse_window_selection",
                    "selected_window": selected_label,
                },
            }
        )

        history = band_result.get("refinement_history", [])
        for display_iteration, step in enumerate(history, start=1):
            accepted = bool(step.get("accepted"))
            action = step.get("action") or "CURRENT"
            selected_action = action if accepted else "CURRENT"
            if accepted:
                step_reasoning = (
                    f"Accepted {action}: it preserved every success criterion and "
                    "strictly improved criterion or artifact quality."
                )
            elif action == "REDUCE_STEP":
                step_reasoning = (
                    "No safe improving candidate was found at this resolution; reduced the "
                    "local-search step and kept CURRENT."
                )
            else:
                step_reasoning = (
                    "No safe improving candidate remained at the minimum step; kept CURRENT."
                )

            self.on_iteration(
                {
                    "agent_id": self.agent_id,
                    "iteration": display_iteration,
                    "current_image_path": (
                        step.get("current_image_path") or band_result["final_image_path"]
                    ),
                    "candidates": step.get("candidates", []),
                    "selected_label": selected_action,
                    "reasoning": step_reasoning,
                    "extra": {
                        "stage": f"{step.get('phase', 'refinement')}_refinement",
                        "phase_iteration": step.get("iteration"),
                        "action": action,
                        "accepted": accepted,
                        "before": step.get("before"),
                        "after": step.get("after"),
                        "criterion_statuses": step.get("criterion_statuses", {}),
                        "artifact_severities": step.get("artifact_severities", {}),
                    },
                }
            )
