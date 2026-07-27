"""Isovalue specialist: segments the volume's own intensity histogram into material BANDS
(valley-to-valley ranges, each containing one peak), asks the LLM which band's material
matches the goal (from cheap per-band preview renders), then DETERMINISTICALLY derives an
opacity/color transfer function from that band's own [low, high] range and applies it via
direct volume rendering -- instead of picking a single hand-tuned isovalue number by
comparing many similar candidate renders.

Why: a single isovalue is a hard binary threshold -- every voxel crossing it becomes
surface, with no way to distinguish a thin/isolated noise speck from a thick, continuous
real structure. A gradual opacity ramp across the whole target band lets THICK material
(many voxels deep) accumulate to full opacity along the viewing ray while THIN/isolated
noise (little depth) stays comparatively faint -- the same physics behind
examples/render_skull_transfer_function.py's hand-tuned head preset, but here the ramp's
shape is computed directly from this dataset's own histogram band boundaries
(build_opacity_ramp_for_band) instead of guessed constants.

Per-band previews (render_band_previews) render EACH band under its OWN derived opacity
ramp via direct volume rendering -- the same build_opacity_ramp_for_band call used for the
final applied result -- rather than a single isosurface at the band's peak. A peak is just
wherever the histogram happens to be tallest within a band; for a broad or lopsided band
(e.g. several small fragments merged into one large band dominated by a huge low-intensity
sub-range) the peak can sit nowhere near where that band's own distinctive material actually
renders, making a peak-only isosurface a poor stand-in for the band as a whole. Previewing
with the exact mechanism used for the final result also means there's no risk of the
selected band's actual look differing from whatever got judged. Multi-angle tiling
(render_band_previews) still applies -- there are only ~2-4 bands per dataset, versus up to
16 individual isovalue candidates previously, so this is also cheaper per comparison call.

Some datasets' target material never forms its own histogram peak at all -- a smooth,
unimodal intensity gradient where e.g. bone blends continuously into soft tissue rather than
forming a separate population of voxels (data/skull_256x256x256_uint8.raw and
data/foot_256x256x256_uint8.raw are both like this). For those, run_isovalue_band_selection
can run ONE bounded refinement round: if the LLM reports the band it picked still looks like
a mix of materials, that band gets split evenly into sub-ranges (_subdivide_band) and
compared again, so the model gets a chance to isolate a purer sub-range instead of silently
settling for "everything above background". See that function's docstring for the mechanism
in full.

Operates on an existing, already-`.initialize()`d CameraReasoningSession, via
CameraReasoningSession.set_transfer_function() -- used both for per-band previews (restored
afterward) and the final applied result -- which preserves camera state and can switch the
session between isosurface and volume-rendering mode in place, so a camera task that ran
before or after this one composes onto the same render correctly.
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
DEFAULT_MIN_BAND_FRACTION = 0.01  # drop bands with <1% of the volume's voxels (noise slivers)
DEFAULT_PEAK_OPACITY = 0.55
DEFAULT_SUSTAIN_OPACITY = 0.70
DEFAULT_REFINEMENT_SUBDIVISIONS = 3  # sub-ranges to split a band into for the refinement round

# Six views per band preview, computed RELATIVE to the session's camera state at the moment
# the sweep runs (via camera_actions.apply_action) rather than fixed absolute world axes --
# see examples/render_head_iso_candidates.ipynb's six_view_cameras, which used absolute axes
# tuned specifically for vis_male_128x256x256_uint8.raw and wouldn't transfer to other
# datasets, so these labels mean "relative to wherever the camera already was".
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
        "Segments the volume's intensity histogram into material bands, asks the LLM "
        "which band matches the goal from cheap per-band previews, then deterministically "
        "derives an opacity ramp from that band's own range and renders it via direct "
        "volume rendering."
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

BAND_SELECTION_PROMPT_TEMPLATE = """You are selecting which MATERIAL BAND best matches the goal, by comparing rendered
previews of each band -- not by reasoning about the numbers alone.

Goal: {goal}

Success criteria this selection must satisfy:
{success_criteria_block}

Each band is a contiguous RANGE of scalar intensities from the volume's own intensity
histogram, roughly corresponding to one material (e.g. background, soft tissue, bone). The
image shown for each band is rendered with that band's own opacity ramp already applied --
the SAME rendering the final result will use if this band is selected -- so judge each
preview as an accurate representation of that band's actual appearance.{multi_angle_note}

You are given one image per band, each preceded by "[{label_prefix}<low>_<high>]".

Bands available this call: {band_list}

Evaluate every band listed above exactly once -- do not shortlist, group, or skip any.

Additionally, judge whether the band you SELECT looks like ONE relatively uniform material,
or a MIX of more than one structure/density blended together (e.g. the target is present but
not cleanly separated from something else, such as bone blended with soft tissue because this
band spans both). Set "refine_further" to true if narrowing the selected band's own intensity
range further would likely isolate the target more cleanly; set it to false if the selected
band already looks like one clean, well-isolated material.

Respond with STRICT JSON ONLY, no prose outside the JSON:
{{
  "band_evaluation": [
    {{"band": "{label_prefix}<low>_<high>", "observation": "<what material/structure this band shows>", "satisfies_goal": true | false}}
  ],
  "selected_band": "<MUST be exactly one of the bands listed above>",
  "goal_satisfied": true | false,
  "refine_further": true | false,
  "reasoning": "<why the selected band is the best match for the goal, referencing the other bands>"
}}
"""

MULTI_ANGLE_NOTE = (
    " Each band's image is a single tiled grid combining 6 different camera angles "
    "(front/right/back/left/top/bottom), each sub-panel labeled with its angle name in the "
    "top-left corner. Judge using ALL sub-panels together -- a structure that looks clean "
    "in one sub-panel can still be fragmented or absent in another."
)

BAND_LABEL_PREFIX = "BAND_"


def _compute_smoothed_histogram(
    raw_path: str,
    scalar_type: str,
    num_bins: int,
    value_range: Tuple[float, float],
    smoothing_window: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read raw_path once and return (raw histogram, smoothed histogram, total voxel count)
    -- shared by compute_histogram_bands and run_isovalue_band_selection's refinement round
    (_subdivide_band) so a refinement doesn't need its own separate pass over the volume."""
    intensities = np.fromfile(raw_path, dtype=np.dtype(scalar_type))
    total_voxels = intensities.size
    hist, _ = np.histogram(intensities, bins=num_bins, range=value_range)
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed_hist = np.convolve(hist, kernel, mode="same")
    return hist, smoothed_hist, total_voxels


def _bands_from_histogram(
    hist: np.ndarray,
    smoothed_hist: np.ndarray,
    total_voxels: int,
    num_bins: int,
    min_band_fraction: float,
) -> List[dict]:
    """Segment an already-computed histogram into material BANDS using local minima
    ("valleys") as boundaries -- each band roughly corresponds to one material (background,
    soft tissue, bone, ...), bounded by valleys and containing one local maximum ("peak"). A
    valley sits between two materials, where relatively few voxels share that exact
    intensity; a peak sits in the middle of a homogeneous material. See
    compute_histogram_bands (the public entry point that also computes the histogram).

    Bands below `min_band_fraction` of the whole volume's voxels are MERGED into their
    larger neighbor (repeatedly, until every remaining band clears the threshold) rather
    than simply dropped -- a real material can still form a thin shell with a small overall
    voxel count (e.g. bone in a skull scan) fragmented by several closely-spaced, noisy
    valleys; dropping those slivers outright would silently delete that whole material
    instead of just cleaning up spurious fragmentation within it.

    Returns bands sorted by intensity, each {"low": int, "high": int, "peak": int,
    "voxel_fraction": float}.
    """
    diff = np.diff(smoothed_hist)
    sign = np.sign(diff)
    sign[sign == 0] = 1
    sign_changes = np.diff(sign)
    local_minima = sorted(int(v) for v in (np.flatnonzero(sign_changes > 0) + 1))

    # Boundaries are treated as HALF-OPEN [low, high) ranges internally (num_bins, not
    # num_bins - 1, as the final edge) so adjacent bands never share a bin -- summing
    # voxel_count across all bands must exactly reproduce the volume's total voxel count,
    # with no double-counting at shared boundaries.
    boundaries = sorted({0, num_bins, *local_minima})

    def _band_from_range(low: int, high: int) -> dict:
        segment = smoothed_hist[low:high]
        peak = low + int(np.argmax(segment))
        voxel_count = float(hist[low:high].sum())
        return {"low": low, "high": high, "peak": peak, "voxel_count": voxel_count}

    bands = [
        _band_from_range(low, high)
        for low, high in zip(boundaries[:-1], boundaries[1:])
        if high > low
    ]

    # Merge bands below threshold using greedy agglomerative merging: at each step, merge
    # whichever ADJACENT PAIR (among pairs involving an under-threshold band) has the
    # SMALLEST combined voxel count, not just "the bigger of a small band's two neighbors".
    # A naive "always merge into the bigger neighbor" rule lets one dominant band (e.g.
    # background) swallow small fragments one at a time before they ever get a chance to
    # consolidate with each other -- e.g. many small, noisy bone-density fragments would
    # each merge individually into a much larger background band instead of first combining
    # into one sensible "bone" band. Preferring the cheapest (smallest-combined) merge each
    # round lets genuinely related small fragments coalesce first.
    min_voxel_count = min_band_fraction * total_voxels
    while len(bands) > 1:
        under_threshold = [i for i, b in enumerate(bands) if b["voxel_count"] < min_voxel_count]
        if not under_threshold:
            break

        best = None  # (combined_voxel_count, low_index, high_index)
        for i in under_threshold:
            for target_index in (i - 1, i + 1):
                if 0 <= target_index < len(bands):
                    pair = tuple(sorted((i, target_index)))
                    combined = bands[pair[0]]["voxel_count"] + bands[pair[1]]["voxel_count"]
                    if best is None or combined < best[0]:
                        best = (combined, pair[0], pair[1])

        _, lo, hi = best
        merged = _band_from_range(bands[lo]["low"], bands[hi]["high"])
        bands = bands[:lo] + [merged] + bands[hi + 1:]

    return [
        {"low": b["low"], "high": b["high"], "peak": b["peak"],
         "voxel_fraction": b["voxel_count"] / total_voxels if total_voxels else 0.0}
        for b in bands
    ]


def compute_histogram_bands(
    raw_path: str,
    scalar_type: str = "uint8",
    num_bins: int = DEFAULT_NUM_BINS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    min_band_fraction: float = DEFAULT_MIN_BAND_FRACTION,
) -> List[dict]:
    """Segment the volume's own intensity histogram into material bands. Reads the raw
    scalar values directly from `raw_path` (shape doesn't matter for a histogram, so no
    `dimensions` argument is needed here). See _bands_from_histogram for the valley-finding
    and merge logic, and build_opacity_ramp_for_band for how a band's own [low, high] range
    becomes an opacity ramp instead of a single arbitrary threshold.

    Returns bands sorted by intensity, each {"low": int, "high": int, "peak": int,
    "voxel_fraction": float}.
    """
    hist, smoothed_hist, total_voxels = _compute_smoothed_histogram(
        raw_path, scalar_type, num_bins, value_range, smoothing_window
    )
    return _bands_from_histogram(hist, smoothed_hist, total_voxels, num_bins, min_band_fraction)


def _subdivide_band(
    band: dict,
    hist: np.ndarray,
    smoothed_hist: np.ndarray,
    total_voxels: int,
    num_parts: int = DEFAULT_REFINEMENT_SUBDIVISIONS,
) -> List[dict]:
    """Split ONE band's [low, high) range evenly into `num_parts` contiguous sub-ranges,
    each with its own peak (local maximum of the smoothed histogram WITHIN that sub-range)
    and voxel_fraction -- used for the refinement round in run_isovalue_band_selection, when
    a band has no internal valley to split at naturally (see that function's docstring for
    why: some datasets' target material never separates into its own histogram peak at all,
    e.g. a smooth CT attenuation gradient). An even split by intensity still lets the LLM
    zero in on a higher- or lower-intensity part of an otherwise undifferentiated band,
    since intensity often correlates with density even without a discrete peak marking a
    material boundary.

    Returns [band] unchanged if the band is too narrow to split into `num_parts` distinct
    integer sub-ranges.
    """
    low, high = band["low"], band["high"]
    width = high - low
    if width < num_parts:
        return [band]

    edges = [low + round(i * width / num_parts) for i in range(num_parts + 1)]
    edges[-1] = high

    sub_bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        segment = smoothed_hist[a:b]
        peak = a + int(np.argmax(segment))
        voxel_count = float(hist[a:b].sum())
        sub_bands.append({
            "low": a, "high": b, "peak": peak,
            "voxel_fraction": voxel_count / total_voxels if total_voxels else 0.0,
        })
    return sub_bands


def _strictly_increasing(points: List[tuple]) -> List[tuple]:
    """Drop any point whose first element doesn't strictly exceed the previous kept point's
    -- vtkPiecewiseFunction/vtkColorTransferFunction expect strictly increasing scalar
    positions; degenerate bands (e.g. peak coinciding with low) could otherwise produce
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
    """Deterministically derive an opacity/color transfer function from ONE histogram
    band's own [low, high] range and peak -- instead of hand-tuned constants.

    Shape: opacity stays 0 below the band, ramps up GRADUALLY starting at `low` (never a
    hard step) to `peak_opacity` by the band's own peak, then a slightly higher
    `sustain_opacity` from `high` onward. Gradual ramps let THICK, continuous material (the
    real target, many voxels deep along the viewing ray) accumulate to full opacity via
    depth, while THIN/isolated structures at the same intensity (noise) stay comparatively
    faint since they have little depth to accumulate over -- see the isovalue specialist's
    module docstring.

    Color: near-black below the band (invisible anyway), transitioning to a neutral warm
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


def render_band_previews(
    session: CameraReasoningSession,
    bands: List[dict],
    output_dir: str,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    multi_angle: bool = True,
) -> List[Tuple[dict, str]]:
    """Render one preview per band, using THAT band's own derived opacity ramp
    (build_opacity_ramp_for_band) via direct volume rendering -- NOT a single isosurface at
    the band's peak.

    A peak is just wherever the histogram happens to be tallest within a band -- for a
    broad or lopsided band (e.g. several small fragments merged into one large band
    dominated by a huge low-intensity sub-range) the peak can sit nowhere near where the
    band's own most distinctive material actually shows up, making a peak-only isosurface a
    poor stand-in for what selecting that band would actually produce. Rendering each
    preview with the SAME ramp-building logic used for the final applied result also means
    the preview IS accurate -- there's no risk of the selected band's actual look differing
    from whatever was judged.

    Restores the session's transfer function/isovalue (whichever mode it was in) AND camera
    state to whatever they were before this call afterward, even if a render fails partway
    through.

    Returns [(band, image_path), ...] in the same order as `bands`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_use_volume_rendering = session.use_volume_rendering
    original_isovalue = session.isovalue
    original_opacity_points = session.opacity_points
    original_color_points = session.color_points
    camera = session._renderer.GetActiveCamera()
    original_camera_state = get_camera_state(camera)

    views_to_render = MULTI_ANGLE_VIEWS if multi_angle else MULTI_ANGLE_VIEWS[:1]

    rendered: List[Tuple[dict, str]] = []
    try:
        for band in bands:
            opacity_points, color_points = build_opacity_ramp_for_band(
                band, min_value=value_range[0], max_value=value_range[1]
            )
            session.set_transfer_function(opacity_points, color_points)
            candidate_views: List[Tuple[str, str]] = []
            for view_label, action in views_to_render:
                set_camera_state(camera, original_camera_state)
                if action is not None:
                    apply_action(action, camera, session._renderer)
                else:
                    session._renderer.ResetCameraClippingRange()
                image_path = str(out_dir / f"band_{band['low']}_{band['high']}_{view_label}.png")
                save_screenshot(session._render_window, image_path)
                candidate_views.append((view_label, image_path))

            if multi_angle:
                grid_path = str(out_dir / f"band_{band['low']}_{band['high']}_grid.png")
                _combine_views_into_grid(candidate_views, grid_path)
                rendered.append((band, grid_path))
            else:
                rendered.append((band, candidate_views[0][1]))
    finally:
        if original_use_volume_rendering:
            session.set_transfer_function(original_opacity_points, original_color_points)
        else:
            session.set_isovalue(original_isovalue)
        set_camera_state(camera, original_camera_state)
        session._renderer.ResetCameraClippingRange()

    return rendered


def _run_band_comparison(
    session: CameraReasoningSession,
    goal: str,
    success_criteria: List[str],
    bands: List[dict],
    model: Optional[str],
    multi_angle: bool,
    render_dir: str,
    value_range: Tuple[float, float],
) -> Tuple[Dict[str, str], Optional[dict], str]:
    """Render one preview per band and make ONE LLM call comparing them -- the render +
    prompt + call + parse logic shared by run_isovalue_band_selection's initial top-level
    comparison and its (optional) refinement-round comparison over sub-bands, so that logic
    only lives in one place.

    Returns (band_images: {label: path}, parsed_response_or_None, raw_response_text).
    """
    rendered = render_band_previews(
        session, bands, render_dir, value_range=value_range, multi_angle=multi_angle
    )
    band_labels = [f"{BAND_LABEL_PREFIX}{band['low']}_{band['high']}" for band in bands]
    band_images = {label: path for label, (_, path) in zip(band_labels, rendered)}
    reference_items = [(label, band_images[label], "") for label in band_labels]

    prompt = BAND_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        success_criteria_block="\n".join(f"- {c}" for c in success_criteria) or "- (none specified)",
        multi_angle_note=MULTI_ANGLE_NOTE if multi_angle else "",
        label_prefix=BAND_LABEL_PREFIX,
        band_list=", ".join(band_labels),
    )
    response_text = ask_chatgpt(prompt=prompt, reference_items=reference_items, model=model)
    parsed = extract_json_object(response_text)
    return band_images, parsed, response_text


def run_isovalue_band_selection(
    session: CameraReasoningSession,
    goal: str,
    success_criteria: List[str],
    model: Optional[str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
    value_range: Tuple[float, float] = DEFAULT_VALUE_RANGE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    min_band_fraction: float = DEFAULT_MIN_BAND_FRACTION,
    output_dir: Optional[str] = None,
    multi_angle: bool = True,
    allow_refinement: bool = True,
) -> dict:
    """Segment the volume's histogram into bands, render one preview per band, ask the LLM
    which band matches `goal` in a single batched comparison call, then deterministically
    derive and apply an opacity ramp from the selected band's own range.

    If the LLM reports the selected band still looks like a MIX of materials
    ("refine_further" in BAND_SELECTION_PROMPT_TEMPLATE), ONE additional round splits that
    band into DEFAULT_REFINEMENT_SUBDIVISIONS even sub-ranges (_subdivide_band) and repeats
    the comparison among just those -- bounded to a single round (not recursive), to keep
    cost predictable. This matters for datasets whose target material never separates into
    its own histogram band at all -- a smooth, unimodal intensity gradient with no interior
    valley (e.g. bone in data/skull_256x256x256_uint8.raw and
    data/foot_256x256x256_uint8.raw never forms its own peak; it blends continuously into
    soft tissue). Without refinement, the initial comparison has no way to discover a purer
    sub-range exists, since it was never shown one -- it can only report the best of what it
    was given, even if that's "everything above background" rather than the target material
    specifically. Pass allow_refinement=False to disable this and always accept the initial
    top-level selection as final.

    Returns {"converged": bool, "bands": [...], "band_images": {label: path},
    "band_evaluation": [...], "selected_band_label": str | None,
    "initial_selected_band_label": str | None (the top-level pick, even if later refined),
    "refined": bool, "refinement_bands": [...] | None, "refinement_band_images":
    {label: path} | None, "refinement_band_evaluation": [...] | None,
    "final_opacity_points"/"final_color_points": the applied ramp or None,
    "final_image_path": str, "reasoning": str, "raw_response": str}.
    """
    hist, smoothed_hist, total_voxels = _compute_smoothed_histogram(
        session.raw_path, session.scalar_type, num_bins, value_range, smoothing_window
    )
    bands = _bands_from_histogram(hist, smoothed_hist, total_voxels, num_bins, min_band_fraction)
    if not bands:
        # Degenerate histogram (e.g. perfectly flat) -- nothing to compare against; fall
        # back to a single band spanning the whole range rather than guessing.
        bands = [{"low": int(value_range[0]), "high": int(value_range[1]), "peak": int(session.isovalue), "voxel_fraction": 1.0}]

    render_dir = output_dir or str(Path(session.output_dir) / "screenshots" / "isovalue_bands")
    band_images, parsed, response_text = _run_band_comparison(
        session, goal, success_criteria, bands, model, multi_angle, render_dir, value_range
    )

    band_labels = [f"{BAND_LABEL_PREFIX}{band['low']}_{band['high']}" for band in bands]
    label_to_band = dict(zip(band_labels, bands))
    selected_label = parsed.get("selected_band") if parsed else None

    if not parsed or selected_label not in label_to_band:
        # Invalid/unresolvable response -- report failure rather than guessing a band.
        return {
            "converged": False,
            "bands": bands,
            "band_images": band_images,
            "band_evaluation": (parsed or {}).get("band_evaluation", []),
            "selected_band_label": None,
            "initial_selected_band_label": None,
            "refined": False,
            "refinement_bands": None,
            "refinement_band_images": None,
            "refinement_band_evaluation": None,
            "final_opacity_points": session.opacity_points,
            "final_color_points": session.color_points,
            "final_image_path": session.render_and_save(),
            "reasoning": "Model response was not valid JSON or selected an unknown band.",
            "raw_response": response_text,
        }

    initial_selected_label = selected_label
    final_band = label_to_band[selected_label]
    final_parsed = parsed
    final_response_text = response_text

    refined = False
    refinement_bands = None
    refinement_band_images = None
    refinement_band_evaluation = None

    if allow_refinement and parsed.get("refine_further"):
        sub_bands = _subdivide_band(final_band, hist, smoothed_hist, total_voxels)
        if len(sub_bands) > 1:
            refine_dir = str(Path(render_dir) / "refine")
            sub_band_images, sub_parsed, sub_response_text = _run_band_comparison(
                session, goal, success_criteria, sub_bands, model, multi_angle, refine_dir, value_range
            )
            sub_labels = [f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}" for b in sub_bands]
            sub_label_to_band = dict(zip(sub_labels, sub_bands))
            sub_selected_label = sub_parsed.get("selected_band") if sub_parsed else None

            refinement_bands = sub_bands
            refinement_band_images = sub_band_images
            refinement_band_evaluation = (sub_parsed or {}).get("band_evaluation", [])

            if sub_parsed and sub_selected_label in sub_label_to_band:
                final_band = sub_label_to_band[sub_selected_label]
                final_parsed = sub_parsed
                final_response_text = sub_response_text
                selected_label = sub_selected_label
                refined = True
            # else: refinement call was invalid/unresolvable -- silently keep the
            # round-1 selection rather than failing outright.

    opacity_points, color_points = build_opacity_ramp_for_band(
        final_band, min_value=value_range[0], max_value=value_range[1]
    )
    session.set_transfer_function(opacity_points, color_points)
    final_image_path = session.render_and_save()

    reasoning = final_parsed.get("reasoning", "")
    if refined:
        reasoning = f"{parsed.get('reasoning', '')} Refined further: {reasoning}"

    return {
        "converged": bool(final_parsed.get("goal_satisfied", True)),
        "bands": bands,
        "band_images": band_images,
        "band_evaluation": parsed.get("band_evaluation", []),
        "selected_band_label": selected_label,
        "initial_selected_band_label": initial_selected_label,
        "refined": refined,
        "refinement_bands": refinement_bands,
        "refinement_band_images": refinement_band_images,
        "refinement_band_evaluation": refinement_band_evaluation,
        "final_opacity_points": opacity_points,
        "final_color_points": color_points,
        "final_image_path": final_image_path,
        "reasoning": reasoning,
        "raw_response": final_response_text,
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
        min_band_fraction: float = DEFAULT_MIN_BAND_FRACTION,
        multi_angle: bool = True,
        allow_refinement: bool = True,
        on_iteration: Optional[Callable[[dict], None]] = None,
    ):
        self.session = session
        self.model = model
        self.num_bins = num_bins
        self.value_range = value_range
        self.smoothing_window = smoothing_window
        self.min_band_fraction = min_band_fraction
        # Render each band's preview as a 6-angle tiled grid instead of 1 flat view
        # (default) -- still just one image attachment per band in the comparison call.
        self.multi_angle = multi_angle
        # When the LLM reports the selected band still looks like a mix of materials, allow
        # ONE extra round narrowing it into sub-ranges (see run_isovalue_band_selection's
        # docstring). Set False to always accept the initial top-level selection as final.
        self.allow_refinement = allow_refinement
        # Called once (twice if a refinement round ran) with a normalized dict after band
        # selection completes (see _handle_band_result) -- e.g. a notebook display callback.
        # Optional.
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
            num_bins=self.num_bins,
            value_range=self.value_range,
            smoothing_window=self.smoothing_window,
            min_band_fraction=self.min_band_fraction,
            multi_angle=self.multi_angle,
            allow_refinement=self.allow_refinement,
        )

        if self.on_iteration:
            self._handle_band_result(band_result)

        state_patch = {
            "transfer_function": (self.session.opacity_points, self.session.color_points),
            "rendered_image_path": band_result["final_image_path"],
        }
        num_bands = len(band_result["bands"])
        refined_note = " (refined into sub-ranges)" if band_result["refined"] else ""

        if band_result["converged"] and band_result["selected_band_label"]:
            return AgentExecutionResult(
                agent_id=self.agent_id,
                status="success",
                goal_satisfied=True,
                state_patch=state_patch,
                confidence=0.75,
                reason=f"Selected {band_result['selected_band_label']} from {num_bands} histogram "
                       f"band(s){refined_note}: {band_result['reasoning']}",
                satisfied_criteria=list(success_criteria),
                unsatisfied_criteria=[],
                suggested_capabilities=[],
                iterations_used=2 if band_result["refined"] else 1,
                artifacts=[band_result["final_image_path"]],
            )

        return AgentExecutionResult(
            agent_id=self.agent_id,
            status="partial" if num_bands else "failed",
            goal_satisfied=False,
            state_patch=state_patch,
            confidence=0.3,
            reason=f"Could not confidently match a histogram band to the goal{refined_note}: "
                   f"{band_result['reasoning']}",
            satisfied_criteria=[],
            unsatisfied_criteria=list(success_criteria),
            suggested_capabilities=[
                "adjust_opacity_transfer_function",
                "apply_clipping",
                "segment_target_structure",
            ],
            failure_type="capability_insufficient",
            iterations_used=2 if band_result["refined"] else 1,
            artifacts=[band_result["final_image_path"]],
        )

    def _handle_band_result(self, band_result: dict) -> None:
        """Normalize each band-selection round into the shared on_iteration shape (see
        CameraSpecialist._handle_camera_iteration for the camera-side version of this same
        normalized dict). Reports iteration 0 for the top-level band comparison, and --
        only when a refinement round actually ran -- iteration 1 for the sub-band
        comparison."""
        self._emit_band_round(
            iteration=0,
            bands=band_result["bands"],
            band_images=band_result["band_images"],
            band_evaluation=band_result["band_evaluation"],
            selected_label=band_result["initial_selected_band_label"],
            reasoning=band_result.get("reasoning") or "",
            image_path=band_result["final_image_path"],
            extra={"converged": band_result["converged"], "refined": band_result["refined"]},
        )
        if band_result.get("refinement_bands"):
            self._emit_band_round(
                iteration=1,
                bands=band_result["refinement_bands"],
                band_images=band_result["refinement_band_images"] or {},
                band_evaluation=band_result["refinement_band_evaluation"] or [],
                selected_label=band_result["selected_band_label"] if band_result["refined"] else None,
                reasoning=band_result.get("reasoning") or "",
                image_path=band_result["final_image_path"],
                extra={"converged": band_result["converged"], "refined": band_result["refined"]},
            )

    def _emit_band_round(
        self,
        iteration: int,
        bands: List[dict],
        band_images: Dict[str, str],
        band_evaluation: List[dict],
        selected_label: Optional[str],
        reasoning: str,
        image_path: str,
        extra: dict,
    ) -> None:
        evaluation_by_label = {
            e["band"]: e
            for e in band_evaluation
            if isinstance(e, dict) and isinstance(e.get("band"), str)
        }

        candidates = []
        for band in bands:
            label = f"{BAND_LABEL_PREFIX}{band['low']}_{band['high']}"
            evaluation = evaluation_by_label.get(label, {})
            candidates.append({
                "label": label,
                "image_path": band_images.get(label),
                "selected": label == selected_label,
                "real_action": None,
                "observation": evaluation.get("observation"),
                "satisfies_goal": evaluation.get("satisfies_goal"),
                "range": f"{band['low']}-{band['high']} (peak {band['peak']})",
            })

        self.on_iteration({
            "agent_id": self.agent_id,
            "iteration": iteration,
            "current_image_path": image_path,
            "candidates": candidates,
            "selected_label": selected_label,
            "reasoning": reasoning,
            "extra": extra,
        })
