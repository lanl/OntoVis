"""
Blind visual-rollout camera-action selection: a structurally label-blind variant
of visual_rollout_agent.py.

Problem this solves: even when a candidate image is rendered from an actual
camera action, if the LLM sees the real action name anywhere (e.g.
"[AZIMUTH_LEFT_COARSE]" as the image's label), it can reason from the words
"left"/"right"/"up"/"down" instead of the actual visual differences between the
current render, candidate renders, and target. Telling the model "don't rely on
the label" is not reliable if the label is still visible.

This module removes the label entirely: every candidate is rendered as before
(reusing render_candidate_rollouts from visual_rollout_agent.py, unmodified),
then assigned a random opaque ID (e.g. "CANDIDATE_K7P4Q") that is regenerated and
reshuffled every iteration and carries no information about direction or scale.
The LLM only ever sees opaque IDs + images + one uniform, non-descriptive caption.
Python alone holds the private candidate_id -> real movement family mapping, and
resolves the final selection into a real, VALID_ACTIONS-checked action name
(AZIMUTH_LEFT_COARSE, AZIMUTH_LEFT_MEDIUM, AZIMUTH_LEFT_FINE, ...) only AFTER the
visual decision has been made — the LLM never outputs or even sees a real action
name.

Scope: candidates are the four movement families that have COARSE/MEDIUM/FINE
magnitude variants — AZIMUTH_LEFT, AZIMUTH_RIGHT, ELEVATION_UP, ELEVATION_DOWN —
rendered ONLY at COARSE (their full-magnitude outcome); MEDIUM and FINE are never
directly rendered, only reasoned about as smaller interpolations along the same
visually observed trajectory. Plus fixed, single-outcome candidates with no
magnitude variants at all — AZIMUTH_RIGHT_180, ELEVATION_UP_180, and STOP (see
FIXED_BLIND_ACTIONS) — whose magnitude cannot be varied, so Pass 3 is skipped
for them entirely. STOP is now a genuine blind candidate: it gets an opaque ID
and a "rendered" image that is simply the current render (render_candidate_rollouts
already special-cases this), so Pass 1 diagnoses it exactly like any other
candidate and Pass 2's normal ranking can select it. Pass 2 ALSO still has a
separate, independent fixed-threshold STOP check that can short-circuit before
ranking even runs (see select_candidate_from_diagnosis) — both paths can
produce "STOP", which is fine, they're not mutually exclusive. UNDO_LAST is not
selectable by either LLM pass in this refactor (see "UNDO_LAST" below).

------------------------------------------------------------------------------
Three-pass architecture
------------------------------------------------------------------------------

Earlier versions of this module asked one LLM call to both diagnose every
candidate AND synthesize/select a winner AND recommend a magnitude, all in a
single free-form response. That allowed internally inconsistent outputs: the
model could correctly describe Candidate A as facing the target side and
Candidate B as facing away from it, and then nevertheless "select" Candidate B
during a later free-form synthesis paragraph. The old parser only ever looked
for a "Selected candidate:" line and had no way to catch that contradiction.

This module now enforces a strict three-pass pipeline so the visual diagnosis is
authoritative and a later free-form step can never override it:

  Pass 1 (LLM):     diagnose_blind_candidates()
                     Structured JSON diagnosis of the target, the current render,
                     and every opaque candidate. This pass NEVER selects a
                     candidate, never recommends a magnitude, and never outputs
                     STOP/UNDO_LAST — its only job is an honest, structured,
                     human-readable description of what is visible.

                     Pass 1 is additionally REFERENCE-GROUNDED (see "Reference
                     grounding" below): a bank of reference-view images is
                     attached alongside the candidates, and for the target,
                     the current render, and every candidate, Pass 1 must
                     report by eye which reference view (if any) it most
                     resembles, a categorical match quality, and a required
                     explanation of what's visually different when the match
                     isn't exact. This is an LLM judgment, not a Python
                     precomputation — an earlier revision used deterministic
                     CLIP image-embedding retrieval instead, but that proved
                     unreliable on this dataset (see "Reference grounding").

  Pass 2 (Python):  select_candidate_from_diagnosis()
                     Pure Python, no LLM call. Every candidate is eligible;
                     deterministically selects the best-ranked one from Pass
                     1's structured diagnosis by numeric ranking
                     (similarity_score, then confidence, then
                     reference_match_confidence, then stable order) — or
                     deterministically returns STOP if the current render
                     already satisfies fixed similarity/confidence thresholds
                     and the best-ranked candidate doesn't offer a meaningful
                     improvement. There is no longer a categorical
                     target_progress eligibility gate — an earlier revision of
                     this pipeline had one, but it was an LLM trajectory
                     judgment that could block selection outright even when a
                     candidate's similarity_score showed real improvement, so
                     it was removed in favor of ranking by the numeric score
                     alone (always the LLM's own estimate — no image
                     embeddings anywhere in this module).

                     If EVERY candidate's reference_match_quality is
                     "unclear" (a narrow reference bank has nothing to say
                     about any of them), numeric ranking is skipped in favor
                     of an exploration fallback: continue the same movement
                     direction as the last applied action if one of this
                     iteration's candidates continues it, else pick one
                     arbitrarily — see "Reference-bank exploration fallback"
                     in select_candidate_from_diagnosis's docstring.

  Pass 3 (LLM):     estimate_selected_candidate_magnitude()
                     Only called for a scalable-family candidate (never for
                     STOP, and never for a FIXED_BLIND_ACTIONS candidate, which
                     has no magnitude to choose). Receives ONLY the current
                     render, the target, and the already-selected candidate's
                     render — the other candidates are not shown, so this pass
                     cannot reconsider or replace Python's selection. It answers
                     one question only: full / reduced / minimal magnitude.

Final action construction (Python) then privately resolves the Pass 2 selection
+ Pass 3 magnitude into a real, VALID_ACTIONS-checked action name, exactly as
before.

This is purely additive: visual_rollout_agent.py, prompt_writer.py, session.py,
camera_actions.py, camera_state.py, and action_parser.py are all unmodified. It
reuses:
  - visual_rollout_agent.render_candidate_rollouts (unmodified, candidate rendering)
  - camera_actions.VALID_ACTIONS (unmodified, final validation)
  - chatgpt_client.ask_chatgpt's `reference_items` param (unmodified)
  - CameraReasoningSession.process_chatgpt_response (unmodified action-application
    path — fed a synthetic "Next action:\\n<ACTION>" response once Python has
    privately resolved and validated the real action name)

------------------------------------------------------------------------------
Reference grounding
------------------------------------------------------------------------------

An earlier revision of this module replaced LLM-judged reference matching with
deterministic CLIP image-embedding retrieval (cosine similarity via a
GraphImageMatcher), on the theory that a plain nearest-neighbor lookup would be
more consistent than an LLM's eye. In practice, on this dataset (18 skull
isosurface renders that all share nearly identical color/composition/texture
and differ only in fine geometric rotation), CLIP-ViT-B/32 embeddings turned
out NOT to be reliable — verified directly by comparing a candidate render
against its claimed closest reference image side by side: the two did not
actually look alike, while the top-2/top-3 CLIP scores for that same candidate
were within noise of each other (~0.91-0.95 cosine similarity band, no clear
separation). So this module has been reverted to LLM-judged visual matching.

Reference grounding is now performed by Pass 1 itself, by eye, against a bank
of attached reference-view images (built by `_build_reference_bank_items()`
from `reference_image_paths` (a plain node_id -> image_path dict) +
`node_descriptions`). Every reference image is attached to the Pass 1 request,
each immediately followed by its own deterministic description, labeled
"[REFERENCE_<node_id>]". For the target (if a target image was given), the
current render, and every candidate, Pass 1 must report which reference image
(if any) looks most similar, a categorical `reference_match_quality`
(exact/close/partial/distant/unclear — see the prompt template), and a
required `reference_match_differences` explanation of what's visually
different when the match isn't exact. This directly answers "is it really
similar, or only the closest of a bad set of options" rather than presenting a
single point-estimate confidence number as fact.

`reference_image_paths`/`node_descriptions` being unavailable (or a node's
image missing on disk) simply drops that reference from the attached bank; if
the bank ends up empty, Pass 1 is told explicitly that no reference bank was
provided and must answer null/"unclear"/an explanatory note for every image.
Deliberately decoupled from any particular dataset format — a CameraSpatialGraph
bank (camera_graph.json) and a flat photo bank with no graph/edges concept at
all can both be used, as long as the caller reduces it to node_id -> path.

This module no longer uses image embeddings (CLIP or otherwise) anywhere. An
earlier revision also used a CLIP embedding matcher to override `similarity_score`
with direct cosine similarity against a target IMAGE — that has been removed
too, per explicit request to drop image-embedding dependence entirely.
`similarity_score` is now always the LLM's own judgment, for every field, in
every case.

------------------------------------------------------------------------------
UNDO_LAST
------------------------------------------------------------------------------

Neither Pass 1 nor Pass 3 is asked to select UNDO_LAST. Detecting "the previous
move was a clear reversal" reliably would require either genuine stored
quantitative evidence from the previous iteration (e.g. a measured similarity
regression) or another LLM judgment call layered on top of this pipeline — this
refactor does not invent that heuristic. `_construct_final_action` still accepts
"UNDO_LAST" as a pass-through value for API/backward compatibility (mirroring
STOP), but nothing in the current three-pass pipeline ever produces it
automatically.
"""
import json
import random
import re
import shutil
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from .camera_actions import VALID_ACTIONS
from .chatgpt_client import ask_chatgpt
from .visual_rollout_agent import render_candidate_rollouts

# The four movement families that have COARSE/MEDIUM/FINE variants — eligible for
# blind magnitude interpolation (full/reduced/minimal -> COARSE/MEDIUM/FINE).
# (ROLL/ZOOM/PAN have no such 3-tier structure and are out of scope for this module.)
SCALABLE_MOVEMENT_FAMILIES = {"AZIMUTH_LEFT", "AZIMUTH_RIGHT", "ELEVATION_UP", "ELEVATION_DOWN"}

# Complete, single-outcome actions that also get a blind candidate + rendered
# image, but have no magnitude variants to interpolate, so Pass 3 (magnitude
# estimation) is skipped entirely for these. Their movement_family IS the final
# action, verbatim. STOP's "render" is just the current image (see
# render_candidate_rollouts in visual_rollout_agent.py, which already special-
# cases STOP this way, reused unmodified) — the LLM diagnoses it exactly like
# any other candidate, so Pass 2's ranking can naturally favor stopping when
# the STOP-candidate (identical to current) already looks close enough to the
# target, in addition to Pass 2's separate fixed-threshold STOP check.
FIXED_BLIND_ACTIONS = {"AZIMUTH_RIGHT_180", "ELEVATION_UP_180", "STOP"}

BLIND_CANDIDATE_ACTION_SUBSET = {f"{family}_COARSE" for family in SCALABLE_MOVEMENT_FAMILIES} | FIXED_BLIND_ACTIONS

SCALE_MAP = {"full": "COARSE", "reduced": "MEDIUM", "minimal": "FINE"}

# Uniform, non-descriptive caption used for every blind candidate — identical
# text for all of them, so it carries no distinguishing information.
BLIND_CANDIDATE_DESCRIPTION = "A full-magnitude candidate render generated from the current camera state."

_CANDIDATE_ID_ALPHABET = string.ascii_uppercase + string.digits

_MAGNITUDE_VALUES = {"full", "reduced", "minimal"}

# The reference-quality categories Pass 1 must choose from when reporting how
# closely an image matches its claimed closest reference view (see module
# docstring, "Reference grounding"). Mapped to a numeric confidence purely for
# Pass 2's existing numeric ranking tie-breaker — the LLM never outputs a raw
# float itself, only the category.
REFERENCE_MATCH_QUALITY_LEVELS = {"exact", "close", "partial", "distant", "unclear"}
REFERENCE_MATCH_QUALITY_CONFIDENCE = {
    "exact": 1.0,
    "close": 0.8,
    "partial": 0.5,
    "distant": 0.2,
    "unclear": 0.0,
}


class BlindSelectionError(ValueError):
    """Raised when a diagnosis, selection, or magnitude response can't be
    resolved to a valid, consistent action — never silently defaulted to STOP,
    the first candidate, or an arbitrary/unvalidated action."""


@dataclass
class BlindCandidateBatch:
    """LLM-visible blind candidates + the private mapping back to real actions.

    visible_items: (candidate_id, neutral_image_path, neutral_description) triples,
        in shuffled order — safe to hand straight to ask_chatgpt's reference_items.
    candidate_mapping: candidate_id -> real movement family (e.g. "AZIMUTH_LEFT").
        NEVER sent to the LLM; only used locally after both LLM passes return.
    """

    visible_items: List[Tuple[str, str, str]]
    candidate_mapping: Dict[str, str]


def _movement_family_from_action(action: str) -> str:
    if action in FIXED_BLIND_ACTIONS:
        return action  # already a complete, non-scalable action name
    if action.endswith("_COARSE"):
        return action[: -len("_COARSE")]
    raise ValueError(f"unsupported blind candidate action: {action!r}")


def _movement_family_from_applied_action(action: Optional[str]) -> Optional[str]:
    """Best-effort movement-family extraction from a REAL, already-applied
    action name (e.g. the most recent entry in session._action_history), for
    the "continue the same direction" exploration fallback in
    select_candidate_from_diagnosis (see "Reference-bank exploration fallback"
    in the module docstring). Unlike _movement_family_from_action (which only
    accepts THIS iteration's blind candidate actions, COARSE-only, and raises
    on anything else), this accepts any already-applied action, including
    MEDIUM/FINE magnitude variants — and returns None (never raises) for
    STOP/UNDO_LAST or an unrecognized action, since there's no "direction" to
    continue in that case.
    """
    if not action or action in ("STOP", "UNDO_LAST"):
        return None
    if action in FIXED_BLIND_ACTIONS:
        return action
    for suffix in ("_COARSE", "_MEDIUM", "_FINE"):
        if action.endswith(suffix):
            return action[: -len(suffix)]
    return None


def _generate_opaque_candidate_id(existing: Set[str]) -> str:
    while True:
        suffix = "".join(random.choices(_CANDIDATE_ID_ALPHABET, k=5))
        candidate_id = f"CANDIDATE_{suffix}"
        if candidate_id not in existing:
            return candidate_id


def prepare_blind_candidates(rollouts: List[dict], temp_dir: str) -> BlindCandidateBatch:
    """Turn real-action rollouts (from render_candidate_rollouts) into a blind
    candidate batch: random opaque IDs (regenerated every call), images copied to
    neutral temp filenames (never the original action-named filename), one
    uniform neutral caption for every candidate, and shuffled order.

    The returned candidate_mapping is private — the caller must not forward it to
    the LLM in any form (prompt, attachment, response schema, or history).
    """
    out_dir = Path(temp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = list(rollouts)
    random.shuffle(entries)

    used_ids: Set[str] = set()
    visible_items: List[Tuple[str, str, str]] = []
    candidate_mapping: Dict[str, str] = {}

    for entry in entries:
        movement_family = _movement_family_from_action(entry["action"])
        candidate_id = _generate_opaque_candidate_id(used_ids)
        used_ids.add(candidate_id)

        neutral_path = out_dir / f"{candidate_id.lower()}.png"
        shutil.copy(entry["image_path"], neutral_path)

        visible_items.append((candidate_id, str(neutral_path), BLIND_CANDIDATE_DESCRIPTION))
        candidate_mapping[candidate_id] = movement_family

    return BlindCandidateBatch(visible_items=visible_items, candidate_mapping=candidate_mapping)


def _find_candidate_image_path(batch: BlindCandidateBatch, candidate_id: str) -> str:
    """Look up the neutral image path for one candidate ID in a batch. Raises
    BlindSelectionError if the ID isn't present (should be unreachable once Pass
    1 validation has already run, but kept defensive since this is also used
    directly by the Pass 3 caller)."""
    for cid, path, _ in batch.visible_items:
        if cid == candidate_id:
            return path
    raise BlindSelectionError(
        f"candidate_id {candidate_id!r} not found in batch; valid IDs: {sorted(batch.candidate_mapping)}"
    )


# ------------------------------------------------------------------
# Reference bank: attach every reference-view image to Pass 1 so the LLM can
# visually judge the match itself (see module docstring, "Reference
# grounding"). No image embeddings are used anywhere in this module —
# `reference_image_paths` is a plain node_id -> image_path dict, deliberately
# decoupled from any particular graph/dataset format (e.g. a CameraSpatialGraph
# bank with camera_graph.json, or a flat photo bank with no graph/edges concept
# at all — both just need a node_id -> path mapping).
# ------------------------------------------------------------------

REFERENCE_LABEL_PREFIX = "REFERENCE_"

# Reference images larger than this on their longest side are downscaled (and
# cached) before being attached — some banks (e.g. reference_views_medical/,
# real photos rather than small VTK renders) are large enough that attaching
# all of them plus the candidates in one Pass 1 call can exceed the model's
# context window. 800 matches the size of the known-working VTK-render banks
# (reference_views_relative/, 800x800), so those pass through unchanged.
REFERENCE_BANK_MAX_DIMENSION = 800
_REFERENCE_BANK_RESIZE_CACHE_DIRNAME = ".blind_rollout_resized_cache"


def _resized_reference_image_path(image_path: str, max_dimension: int = REFERENCE_BANK_MAX_DIMENSION) -> str:
    """Return image_path unchanged if it's already <= max_dimension on its
    longest side; otherwise downscale it (preserving aspect ratio) and return
    a path to a cached copy instead. The cache lives in a sibling
    ".blind_rollout_resized_cache/" directory next to the source image, keyed
    by filename + max_dimension, and is reused across iterations/runs as long
    as the source file's mtime hasn't changed since the cached copy was made —
    resizing 15 reference images on every single Pass 1 call would otherwise
    add real latency for no benefit, since the bank doesn't change mid-run.
    """
    from PIL import Image as PILImage

    source = Path(image_path)
    with PILImage.open(source) as probe:
        width, height = probe.size
    if max(width, height) <= max_dimension:
        return image_path

    cache_dir = source.parent / _REFERENCE_BANK_RESIZE_CACHE_DIRNAME
    cache_path = cache_dir / f"{source.stem}_{max_dimension}{source.suffix}"
    if cache_path.exists() and cache_path.stat().st_mtime >= source.stat().st_mtime:
        return str(cache_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with PILImage.open(source) as img:
        scale = max_dimension / max(width, height)
        resized = img.convert("RGB").resize(
            (round(width * scale), round(height * scale)), PILImage.LANCZOS
        )
        resized.save(cache_path)
    return str(cache_path)


def _build_reference_bank_items(
    reference_image_paths: Optional[Dict[str, str]],
    node_descriptions: Optional[Dict[str, str]],
) -> List[Tuple[str, str, str]]:
    """Build (label, image_path, description) triples for every reference view
    that has both a known description and an existing image file on disk,
    sorted by node_id for a stable prompt order. Returns [] if no
    reference_image_paths/node_descriptions were supplied, or nothing resolves
    to an existing image — Pass 1 is told explicitly when the bank is empty.

    Oversized images are downscaled (see _resized_reference_image_path) before
    being included, to avoid blowing the model's context window when many
    reference images are attached alongside the candidates in one Pass 1 call.
    """
    if not reference_image_paths or not node_descriptions:
        return []

    items: List[Tuple[str, str, str]] = []
    for node_id, description in sorted(node_descriptions.items()):
        image_path = reference_image_paths.get(node_id)
        if not image_path or not Path(image_path).exists():
            continue
        items.append((f"{REFERENCE_LABEL_PREFIX}{node_id}", _resized_reference_image_path(image_path), description))
    return items


# ------------------------------------------------------------------
# Neutralized action history (never exposes real action names to either prompt)
# ------------------------------------------------------------------

def _summarize_history_neutrally(action_history: Optional[List[Dict]], lookback: int = 5) -> str:
    """Convert real-action-name history into a neutral behavior summary — never
    prints a real action name, so the blind prompts can't be biased by it.
    """
    if not action_history:
        return "Recent behavior:\n  (no history yet)"

    recent = action_history[-lookback:]
    actions = [h["action"] for h in recent]

    lines = []
    if len(actions) >= 2 and actions[-1] == actions[-2]:
        lines.append("- The last two steps repeated the same movement.")
    if len(set(actions)) < len(actions):
        lines.append("- Some recent steps have repeated.")
    if "UNDO_LAST" in actions:
        lines.append("- A recent step undid an earlier change.")
    if not lines:
        lines.append("- Recent steps have each produced a new camera state (no obvious repetition detected).")

    return "Recent behavior:\n" + "\n".join(lines)


def _format_target_image_line(target_image_path: Optional[str]) -> str:
    if target_image_path and Path(target_image_path).exists():
        return "  A target reference image is also attached, immediately after the current image.\n"
    return ""


def _format_reference_bank_block(reference_bank_items: List[Tuple[str, str, str]]) -> str:
    """Describe the attached reference bank (or its absence) for the Pass 1
    prompt. The images themselves are attached separately via reference_items
    (see diagnose_blind_candidates) — this text just orients the model to what
    it's looking at and lists the exact valid IDs it may answer with.
    """
    if not reference_bank_items:
        return (
            "No reference bank was provided for this iteration — no reference images are\n"
            "attached. For every image below, set closest_reference_id to null,\n"
            "reference_match_quality to \"unclear\", and briefly note in\n"
            "reference_match_differences that no reference bank was available."
        )

    node_ids = [label[len(REFERENCE_LABEL_PREFIX):] for label, _, _ in reference_bank_items]
    return (
        f"{len(reference_bank_items)} reference-view images are attached below, each preceded by\n"
        f"its own label (e.g. \"[{reference_bank_items[0][0]}]\") and immediately followed by its own\n"
        "short, deterministic description. These are canonical camera viewpoints of the same\n"
        "object, NOT candidates — do not select one as your answer, only compare against them.\n"
        f"Valid reference IDs (answer with the ID only, no \"{REFERENCE_LABEL_PREFIX}\" prefix): "
        + ", ".join(node_ids)
    )


# ------------------------------------------------------------------
# Pass 1 prompt: reference-grounded visual diagnosis only (no selection, no
# magnitude, no STOP, no UNDO_LAST)
# ------------------------------------------------------------------

BLIND_VISUAL_DIAGNOSIS_PROMPT_TEMPLATE = """You are visually diagnosing a set of rendered camera viewpoints by comparing
them to a target viewpoint — never by reasoning about labels, identifiers, or
action-name words.

This is a DIAGNOSIS STAGE ONLY. Do not select a candidate. Do not recommend a
magnitude. Do not output STOP. Do not output UNDO_LAST. The candidate selection
will be performed deterministically by Python from your structured diagnosis
below — your only job is to produce an honest, structured, human-readable visual
diagnosis of what is actually visible in each image.

REFERENCE VIEW BANK

{reference_bank_block}

You are given, in order:

1. The CURRENT rendered image before any movement.
2. The target view: {target_description}
   {target_image_line}3. The reference bank described above (if any).
4. Several candidate images, each preceded by an opaque identifier such as
   "[CANDIDATE_K7P4Q]".

Candidate IDs are meaningless, randomly generated, regenerated every iteration,
and shown in random order. Do not infer anything from identifier text. It
contains no information about direction, magnitude, or movement type. Do not
mention or guess at hidden action names or movement families anywhere in your
diagnosis.

Each candidate image shows the full-magnitude outcome of one hidden camera
movement. Use only the visible current-to-candidate transformation to describe
what changed.

{neutral_history_block}

GLOBAL VIEWPOINT REASONING (used for judging candidate/target similarity)

To judge similarity to the target, compare global viewpoint evidence such as:

* which broad object surfaces are visible;
* relative exposure of frontal, lateral, rear, upper, or lower surfaces;
* the whole-object silhouette;
* major occlusion relationships;
* relative projected positions of distant landmarks;
* how frontal or edge-on major surfaces appear;
* the balance between opposing visible sides;
* coherent silhouette changes across the whole object;
* movement of multiple landmarks as a coherent group;
* foreshortening of major object axes.

Prioritize coherent changes across the whole object over isolated changes in one
small feature. Do NOT reward a candidate because:

* one tooth, cavity, or edge is clearer or sharper;
* the lighting or rendering color happens to be similar;
* the object instance has more similar local anatomy;
* one isolated landmark happens to look alike;
* more detail is visible or the image is easier to interpret.

Similarity judgments must prioritize camera viewpoint over appearance. When the
target image comes from another source, rendering style, object instance,
lighting condition, or dataset, ignore lighting, texture, color, exact object
shape, rendering style, image quality, and missing/damaged local anatomy — use
only viewpoint-relevant geometry.

Respond with STRICT JSON ONLY — no prose before or after the JSON, no Markdown
commentary outside the JSON itself. You may wrap the JSON in a single Markdown
code fence if you prefer, but nothing else may appear in your response.

Use exactly this JSON structure:

{{
  "target": {{
    "view_description": "What is visually apparent about the target's viewpoint.",
    "closest_reference_id": "view_XXX or null",
    "reference_match_quality": "exact | close | partial | distant | unclear",
    "reference_match_differences": "What is visually different from the matched reference, if anything."
  }},
  "current": {{
    "view_description": "What is visually apparent about the current render's viewpoint.",
    "difference_from_target": "Direct visual comparison with the target.",
    "closest_reference_id": "view_XXX or null",
    "reference_match_quality": "exact | close | partial | distant | unclear",
    "reference_match_differences": "What is visually different from the matched reference, if anything.",
    "similarity_score": 0,
    "confidence": 0.0
  }},
  "candidates": [
    {{
      "candidate_id": "CANDIDATE_XXXXX",
      "view_description": "What is visually apparent about this candidate's viewpoint.",
      "comparison_to_target": "Direct visual comparison between this candidate and the target.",
      "closest_reference_id": "view_XXX or null",
      "reference_match_quality": "exact | close | partial | distant | unclear",
      "reference_match_differences": "What is visually different from the matched reference, if anything.",
      "similarity_score": 0,
      "confidence": 0.0
    }}
  ]
}}

Field requirements:

* `view_description` (target, current, and every candidate) must be non-empty
  and written so a human reviewing this diagnosis later can understand your
  reasoning without seeing the images.
* `closest_reference_id` (target, current, and every candidate): the reference
  bank ID (e.g. "view_007", no "{reference_label_prefix}" prefix) whose image looks most
  similar to THIS image's viewpoint, or null if nothing in the bank resembles
  it well enough to call it a meaningful match, or if no reference bank was
  provided, or (target only) if no target image was attached for you to judge
  against the bank yourself.
* `reference_match_quality`: one of "exact" (essentially indistinguishable),
  "close" (very similar, only minor differences), "partial" (recognizably
  related but a clearly different angle), "distant" (only weak resemblance —
  the least-dissimilar option available, not a real match), or "unclear" (use
  only when closest_reference_id is null).
* `reference_match_differences`: REQUIRED, non-empty. Describe concretely what
  is visually different between this image and its matched reference (e.g.
  "rotated further right than the reference, exposing more posterior
  surface"). If reference_match_quality is "exact", say so explicitly (e.g.
  "No meaningful difference observed") instead of leaving this vague. If
  closest_reference_id is null, briefly say why (e.g. "no reference bank was
  provided" or "no target image was attached").
* `similarity_score` (target is not scored; current and every candidate) must
  be an integer from 0 to 100, where 100 means the viewpoint visually matches
  the target's side/orientation as closely as possible and 0 means it is
  maximally different. This is always your own honest estimate — nothing
  overrides it.
* `confidence` (current and every candidate) must be a number from 0.0 to 1.0.
* Every candidate ID shown to you must appear exactly once in "candidates".
* Do not invent candidate IDs that were not shown to you.
* Do not omit any candidate ID that was shown to you.
* Do not duplicate any candidate ID.
* Do not include a `facing_direction`, `vertical_orientation`, or
  `target_orientation_match` field anywhere.
"""


# ------------------------------------------------------------------
# Pass 3 prompt: magnitude estimation only, for the already-selected candidate
# ------------------------------------------------------------------

BLIND_MAGNITUDE_PROMPT_TEMPLATE = """A candidate viewpoint has already been selected by a deterministic selection
stage that ran after an earlier visual diagnosis. You are NOT allowed to
replace it, reject it, select a different candidate, output STOP, or output
UNDO_LAST. Your only task is to determine the movement MAGNITUDE along the
already-selected visual trajectory.

You are given, in order:

1. The CURRENT rendered image before any movement.
2. The target view: {target_description}
   {target_image_line}3. The SELECTED candidate image, preceded by the identifier
   "[{selected_candidate_id}]". This is the only candidate you will see — it has
   already been chosen; do not reconsider that choice.

The selected candidate image shows the full-magnitude (COARSE) outcome of the
selected hidden camera movement. MEDIUM and FINE outcomes were NOT directly
rendered — you must reason about them only as smaller interpolations along the
same visually observed current-to-candidate trajectory. Never claim that a
MEDIUM or FINE result was directly observed.

{neutral_history_block}

Magnitude definitions:

full — the candidate endpoint closely matches the target, the complete observed
global viewpoint change is useful, and there is no clear overshoot.

Current ---------------- Candidate ~= Target

reduced — the movement direction is correct, but the target appears visually
between the current render and the candidate endpoint, meaning the full
candidate clearly overshoots the target.

Current -------- Target -------- Candidate

minimal — the current render is already visually close to the target, and only
a small interpolation along the selected trajectory is needed; the full
candidate is substantially farther than necessary.

Current -- Target ---------------- Candidate

Rules:

* Magnitude reasoning must use global viewpoint geometry, not local feature
  clarity. Local feature clarity alone must not determine magnitude.
* Do not use magnitude to second-guess whether the candidate's direction was
  correct — that decision has already been made and is final.
* The selected candidate ID in your response must exactly match
  "{selected_candidate_id}".
* Never output a real camera-action name — only the opaque candidate ID already
  given to you.

Respond with STRICT JSON ONLY — no prose before or after the JSON, no Markdown
commentary outside the JSON itself. You may wrap the JSON in a single Markdown
code fence if you prefer, but nothing else may appear in your response.

Use exactly this JSON structure:

{{
  "selected_candidate": "{selected_candidate_id}",
  "magnitude_recommendation": "full | reduced | minimal",
  "magnitude_reasoning": "Human-readable explanation of where the target lies relative to current and candidate.",
  "expected_visual_result": "Concrete and falsifiable prediction about the broad viewpoint change."
}}
"""


# ------------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL | re.IGNORECASE)


def _parse_json_response(response_text: str) -> dict:
    """Parse a strict-JSON LLM response, tolerating a single surrounding
    Markdown code fence (```json ... ``` or ``` ... ```). Never uses eval().
    Raises BlindSelectionError on anything that isn't valid JSON after fence
    stripping, or that doesn't parse to a JSON object.
    """
    text = response_text.strip()

    fence_match = _CODE_FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BlindSelectionError(f"response was not valid JSON: {exc}\nRaw response:\n{response_text}") from exc

    if not isinstance(parsed, dict):
        raise BlindSelectionError(
            f"response JSON must be an object at the top level; got {type(parsed).__name__}"
        )

    return parsed


# ------------------------------------------------------------------
# Pass 1 validation
# ------------------------------------------------------------------

def _require_str(container: dict, key: str, context: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BlindSelectionError(f"{context}: field {key!r} must be a non-empty string; got {value!r}")
    return value


def _require_enum(container: dict, key: str, allowed: Set[str], context: str) -> str:
    value = container.get(key)
    if value not in allowed:
        raise BlindSelectionError(f"{context}: field {key!r} must be one of {sorted(allowed)}; got {value!r}")
    return value


def _require_score(container: dict, key: str, context: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BlindSelectionError(f"{context}: field {key!r} must be an integer 0-100; got {value!r}")
    if not (0 <= value <= 100):
        raise BlindSelectionError(f"{context}: field {key!r} must be in range [0, 100]; got {value!r}")
    return value


def _require_confidence(container: dict, key: str, context: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlindSelectionError(f"{context}: field {key!r} must be a number 0.0-1.0; got {value!r}")
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise BlindSelectionError(f"{context}: field {key!r} must be in range [0.0, 1.0]; got {value!r}")
    return value


def _validate_view_description(container: dict, context: str) -> dict:
    return {"view_description": _require_str(container, "view_description", context)}


def _validate_reference_match(
    container: dict, context: str, valid_node_ids: Set[str], node_descriptions: Dict[str, str]
) -> dict:
    """Validate the LLM's own reference-bank judgment for one image:
    `closest_reference_id`/`reference_match_quality`/`reference_match_differences`
    (see module docstring, "Reference grounding" — this replaced deterministic
    CLIP-embedding matching after it proved unreliable on this dataset).

    `closest_reference_id` must be null, or a string matching one of the node
    IDs actually attached this iteration (a "{REFERENCE_LABEL_PREFIX}" prefix
    is tolerated and stripped, in case the model echoes the label verbatim).
    `reference_match_quality` must be a known category, and must be "unclear"
    exactly when closest_reference_id is null (and never "unclear" otherwise)
    — an inconsistent pairing is treated as an invalid response, same as any
    other malformed field, rather than silently repaired.
    `reference_match_differences` is always required and non-empty.

    Also computes `reference_match_confidence` (a fixed numeric mapped from
    `reference_match_quality`, see REFERENCE_MATCH_QUALITY_CONFIDENCE) and
    `closest_reference_description` (looked up locally, not restated by the
    LLM) purely so downstream code (Pass 2 ranking, trace/console formatting)
    that expects those fields keeps working unchanged.
    """
    raw_id = container.get("closest_reference_id")
    closest_reference_id: Optional[str]
    if raw_id is None:
        closest_reference_id = None
    elif isinstance(raw_id, str):
        candidate_id = raw_id.strip()
        if candidate_id.startswith(REFERENCE_LABEL_PREFIX):
            candidate_id = candidate_id[len(REFERENCE_LABEL_PREFIX):]
        if candidate_id not in valid_node_ids:
            raise BlindSelectionError(
                f"{context}: field 'closest_reference_id' {raw_id!r} is not among this iteration's "
                f"attached reference IDs: {sorted(valid_node_ids)}"
            )
        closest_reference_id = candidate_id
    else:
        raise BlindSelectionError(f"{context}: field 'closest_reference_id' must be a string or null; got {raw_id!r}")

    quality = _require_enum(container, "reference_match_quality", REFERENCE_MATCH_QUALITY_LEVELS, context)
    if (closest_reference_id is None) != (quality == "unclear"):
        raise BlindSelectionError(
            f"{context}: 'reference_match_quality' ({quality!r}) is inconsistent with "
            f"'closest_reference_id' ({closest_reference_id!r}) — quality must be \"unclear\" if and "
            "only if closest_reference_id is null"
        )

    differences = _require_str(container, "reference_match_differences", context)

    return {
        "closest_reference_id": closest_reference_id,
        "closest_reference_description": (
            node_descriptions.get(closest_reference_id, "unclear") if closest_reference_id else "unclear"
        ),
        "reference_match_quality": quality,
        "reference_match_differences": differences,
        "reference_match_confidence": REFERENCE_MATCH_QUALITY_CONFIDENCE[quality],
        "reference_match_evidence": f"LLM visual comparison against the attached reference bank (quality={quality}).",
    }


def _validate_visual_diagnosis(
    diagnosis: dict,
    valid_candidate_ids: Set[str],
    valid_reference_node_ids: Set[str],
    node_descriptions: Dict[str, str],
) -> dict:
    """Validate the Pass 1 structured diagnosis. Raises BlindSelectionError on
    any missing field, out-of-range score, invalid enum value, or candidate-set
    mismatch (missing, duplicate, or unknown candidate IDs). Never repairs a
    semantically invalid diagnosis — only well-formed, complete diagnoses pass.

    Reference-bank matching (closest_reference_id/reference_match_quality/
    reference_match_differences) is now the LLM's own visual judgment,
    validated by `_validate_reference_match` against `valid_reference_node_ids`
    (the set of node IDs actually attached this iteration) — see module
    docstring, "Reference grounding".

    `similarity_score` (current and every candidate) is always the LLM's own
    judgment — this module has no image-embedding fallback or override
    anywhere.

    Returns a normalized copy (candidate IDs upper-cased for consistent lookup;
    all other fields unchanged) rather than mutating the input.
    """
    if "target" not in diagnosis or not isinstance(diagnosis["target"], dict):
        raise BlindSelectionError("diagnosis missing required object field 'target'")
    if "current" not in diagnosis or not isinstance(diagnosis["current"], dict):
        raise BlindSelectionError("diagnosis missing required object field 'current'")
    if "candidates" not in diagnosis or not isinstance(diagnosis["candidates"], list):
        raise BlindSelectionError("diagnosis missing required array field 'candidates'")

    target = diagnosis["target"]
    validated_target = {
        **_validate_view_description(target, "target"),
        **_validate_reference_match(target, "target", valid_reference_node_ids, node_descriptions),
    }

    current = diagnosis["current"]
    validated_current = {
        **_validate_view_description(current, "current"),
        **_validate_reference_match(current, "current", valid_reference_node_ids, node_descriptions),
        "difference_from_target": _require_str(current, "difference_from_target", "current"),
        "similarity_score": _require_score(current, "similarity_score", "current"),
        "confidence": _require_confidence(current, "confidence", "current"),
    }

    seen_ids: Set[str] = set()
    validated_candidates = []
    for index, candidate in enumerate(diagnosis["candidates"]):
        if not isinstance(candidate, dict):
            raise BlindSelectionError(f"candidates[{index}] must be an object; got {type(candidate).__name__}")

        context = f"candidates[{index}]"
        raw_id = candidate.get("candidate_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise BlindSelectionError(f"{context}: field 'candidate_id' must be a non-empty string; got {raw_id!r}")
        candidate_id = raw_id.strip().upper()

        if candidate_id not in valid_candidate_ids:
            raise BlindSelectionError(
                f"{context}: candidate_id {candidate_id!r} is not among this iteration's valid "
                f"candidate IDs: {sorted(valid_candidate_ids)}"
            )
        if candidate_id in seen_ids:
            raise BlindSelectionError(f"{context}: duplicate candidate_id {candidate_id!r}")
        seen_ids.add(candidate_id)

        validated_candidates.append(
            {
                "candidate_id": candidate_id,
                **_validate_view_description(candidate, context),
                **_validate_reference_match(candidate, context, valid_reference_node_ids, node_descriptions),
                "comparison_to_target": _require_str(candidate, "comparison_to_target", context),
                "similarity_score": _require_score(candidate, "similarity_score", context),
                "confidence": _require_confidence(candidate, "confidence", context),
            }
        )

    missing_ids = valid_candidate_ids - seen_ids
    if missing_ids:
        raise BlindSelectionError(f"diagnosis is missing required candidate IDs: {sorted(missing_ids)}")

    return {
        "target": validated_target,
        "current": validated_current,
        "candidates": validated_candidates,
    }


# ------------------------------------------------------------------
# Pass 1 (called every iteration): ask the LLM for a structured, reference-
# grounded visual diagnosis only — no selection, no magnitude, no STOP/UNDO_LAST
# ------------------------------------------------------------------

def candidate_diagnosis_to_current(candidate_diagnosis: dict) -> dict:
    """Reshape a selected candidate's validated diagnosis into "current" shape.

    Used to carry a candidate's own diagnosis forward as next iteration's
    "current" once it's actually applied, instead of asking the LLM to
    re-diagnose the same rendered image again from scratch next iteration —
    which could disagree with what it already said about it as a candidate
    this iteration (drops "candidate_id", renames "comparison_to_target" to
    "difference_from_target"; every other field is identical between the two
    schemas — see _validate_visual_diagnosis).
    """
    current = {k: v for k, v in candidate_diagnosis.items() if k not in ("candidate_id", "comparison_to_target")}
    current["difference_from_target"] = candidate_diagnosis["comparison_to_target"]
    return current


def diagnose_blind_candidates(
    current_image_path: str,
    batch: BlindCandidateBatch,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_image_paths: Optional[Dict[str, str]] = None,
    node_descriptions: Optional[Dict[str, str]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    frozen_target: Optional[dict] = None,
    carried_current: Optional[dict] = None,
) -> dict:
    """Ask the LLM to produce a structured visual diagnosis of the target, the
    current render, and every opaque candidate — with no candidate selection,
    magnitude recommendation, or STOP/UNDO_LAST anywhere in this pass.

    Reference grounding is the LLM's own visual judgment: every reference view
    (built by `_build_reference_bank_items` from `reference_image_paths`
    (node_id -> image_path) + `node_descriptions`) is attached as an image
    alongside the candidates, and Pass 1 must report, per image, which
    reference (if any) it resembles, how closely (`reference_match_quality`),
    and what's different when it's not an exact match
    (`reference_match_differences`) — see module docstring, "Reference
    grounding". No image embeddings are used anywhere in this module;
    `similarity_score` is always the LLM's own estimate too.

    `frozen_target`, if given, REPLACES the LLM's own "target" diagnosis this
    call with the fixed value from the run's first iteration — the model is
    still asked to describe the target (the prompt/schema don't change), but
    its answer for "target" is discarded, so a static target image can never
    drift to a different view_description/closest_reference_id purely from
    LLM sampling noise across iterations.

    `carried_current`, if given, similarly REPLACES the LLM's own "current"
    diagnosis with the previous iteration's already-selected candidate
    diagnosis (via candidate_diagnosis_to_current) — the render shown as
    "current" this iteration is the exact image that candidate was, so we
    reuse its own diagnosis instead of risking an inconsistent fresh redo of
    the same image.

    Returns {"diagnosis": <validated diagnosis dict>, "raw_response": str}.
    Raises BlindSelectionError if the response isn't valid JSON or doesn't
    match the required schema, or doesn't cover exactly this iteration's
    candidate ID set (missing, duplicate, or invented IDs).
    """
    node_descriptions = node_descriptions or {}
    reference_bank_items = _build_reference_bank_items(reference_image_paths, node_descriptions)
    valid_reference_node_ids = {label[len(REFERENCE_LABEL_PREFIX):] for label, _, _ in reference_bank_items}

    prompt = BLIND_VISUAL_DIAGNOSIS_PROMPT_TEMPLATE.format(
        target_description=target_description,
        target_image_line=_format_target_image_line(target_image_path),
        reference_bank_block=_format_reference_bank_block(reference_bank_items),
        reference_label_prefix=REFERENCE_LABEL_PREFIX,
        neutral_history_block=_summarize_history_neutrally(action_history),
    )

    # Reference-bank images (if any) are attached first, followed by the blind
    # candidates — both via reference_items, distinguished by label prefix.
    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path,
        target_image_path=target_image_path,
        reference_items=reference_bank_items + batch.visible_items,
        model=model,
    )

    try:
        raw_diagnosis = _parse_json_response(response_text)
        validated_diagnosis = _validate_visual_diagnosis(
            raw_diagnosis, set(batch.candidate_mapping), valid_reference_node_ids, node_descriptions,
        )
    except BlindSelectionError:
        print("[blind_visual_rollout_agent] ERROR: Pass 1 visual diagnosis failed validation.")
        print(f"[blind_visual_rollout_agent] Currently valid candidate IDs: {sorted(batch.candidate_mapping)}")
        print(f"[blind_visual_rollout_agent] Valid reference IDs this iteration: {sorted(valid_reference_node_ids)}")
        print("[blind_visual_rollout_agent] Raw response was:")
        print(response_text)
        raise

    if frozen_target is not None:
        validated_diagnosis["target"] = frozen_target
    if carried_current is not None:
        validated_diagnosis["current"] = carried_current

    return {"diagnosis": validated_diagnosis, "raw_response": response_text}


def diagnose_blind_candidates_sequentially(
    current_image_path: str,
    batch: BlindCandidateBatch,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_image_paths: Optional[Dict[str, str]] = None,
    node_descriptions: Optional[Dict[str, str]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    frozen_target: Optional[dict] = None,
    carried_current: Optional[dict] = None,
) -> dict:
    """Same contract and return shape as diagnose_blind_candidates(), but calls the LLM
    once PER CANDIDATE instead of once for the whole batch -- each call shows the model
    exactly one candidate image (plus the reference bank, current, and target), so there
    is nothing for it to mislabel a candidate's score/description against. Costs N LLM
    calls instead of 1 (N = number of candidates this iteration) and gives up side-by-side
    candidate comparison, in exchange for eliminating cross-candidate attribution errors
    -- e.g. reporting a left/right judgment that actually belonged to a different candidate
    in the same batch.

    Reuses diagnose_blind_candidates() unmodified for each single-candidate call (a
    BlindCandidateBatch containing just that one candidate), so all existing validation
    logic applies unchanged, just with N=1 each time.

    The first call's target/current diagnosis is frozen (via frozen_target/carried_current)
    and reused for every subsequent call this iteration, so the combined diagnosis ends up
    with exactly one consistent target/current view rather than N slightly different ones
    -- this is the same freezing mechanism run_blind_visual_rollout_alignment_loop already
    uses ACROSS iterations, just applied within one iteration's candidate calls too.

    Returns {"diagnosis": <combined diagnosis dict, candidates in batch.visible_items
    order>, "raw_response": <every call's raw response, concatenated for trace/debugging
    visibility>}.
    """
    combined_candidates = []
    raw_responses = []
    iteration_frozen_target = frozen_target
    iteration_carried_current = carried_current

    for candidate_id, image_path, description in batch.visible_items:
        single_batch = BlindCandidateBatch(
            visible_items=[(candidate_id, image_path, description)],
            candidate_mapping={candidate_id: batch.candidate_mapping[candidate_id]},
        )
        result = diagnose_blind_candidates(
            current_image_path,
            single_batch,
            target_description,
            target_image_path=target_image_path,
            reference_image_paths=reference_image_paths,
            node_descriptions=node_descriptions,
            action_history=action_history,
            model=model,
            frozen_target=iteration_frozen_target,
            carried_current=iteration_carried_current,
        )
        raw_responses.append(f"=== {candidate_id} ===\n{result['raw_response']}")
        combined_candidates.extend(result["diagnosis"]["candidates"])

        if iteration_frozen_target is None:
            iteration_frozen_target = result["diagnosis"]["target"]
        if iteration_carried_current is None:
            iteration_carried_current = result["diagnosis"]["current"]

    combined_diagnosis = {
        "target": iteration_frozen_target,
        "current": iteration_carried_current,
        "candidates": combined_candidates,
    }
    return {"diagnosis": combined_diagnosis, "raw_response": "\n\n".join(raw_responses)}


# ------------------------------------------------------------------
# Pass 2 (pure Python, no LLM call): deterministic candidate selection from
# Pass 1's structured diagnosis
# ------------------------------------------------------------------

def select_candidate_from_diagnosis(
    diagnosis: dict,
    *,
    stop_similarity_threshold: int = 92,
    stop_improvement_margin: int = 3,
    minimum_confidence: float = 0.5,
    candidate_movement_families: Optional[Dict[str, str]] = None,
    last_movement_family: Optional[str] = None,
) -> dict:
    """Deterministically select a candidate (or STOP) from a validated Pass 1
    diagnosis. Never calls an LLM.

    Every candidate is eligible for selection — there is no longer a
    categorical target_progress gate (removed: it was an LLM trajectory
    judgment that could block selection outright even when a candidate's
    similarity_score showed genuine improvement toward the target). Ranking is
    purely by similarity_score (always the LLM's own estimate — no
    image-embedding involvement anywhere in this module), then confidence,
    then reference_match_confidence, then the candidate's position in
    diagnosis["candidates"] as a stable final tie-breaker.

    STOP: returned only when the current render's similarity_score is at least
    stop_similarity_threshold AND its confidence is at least minimum_confidence
    AND the best-ranked candidate doesn't improve on the current render's
    similarity_score by at least stop_improvement_margin.

    Reference-bank exploration fallback: if EVERY candidate's
    reference_match_quality is "unclear" (the reference bank has nothing to
    say about any of this iteration's candidates — e.g. a narrow bank like
    reference_views_medical/skull, which only covers 3 base viewpoints), the
    normal similarity_score ranking above is skipped in favor of a sweep
    strategy: continue the same movement direction as the last applied action
    (`last_movement_family`, matched against `candidate_movement_families`,
    both optional) if a candidate for it exists among this iteration's
    scalable-family candidates, else arbitrarily pick a scalable-family
    candidate. This repeats every iteration the bank stays uninformative,
    since each iteration's own last-applied action becomes the next
    iteration's continuation direction — sweeping through view-space with a
    consistent direction, rather than the LLM's own uncalibrated
    similarity_score guesses or randomly changing direction, until some
    candidate's viewpoint finally lands close enough to a known reference for
    the LLM to report a real match again. Only fixed one-shot actions
    (STOP/UNDO_LAST/FIXED_BLIND_ACTIONS) are excluded from consideration as a
    continuation direction, since repeating them can't sweep anywhere new.
    Requires both `candidate_movement_families` and `last_movement_family`;
    either being unavailable (e.g. the caller didn't supply them, or the first
    iteration has no history yet) just means arbitrary selection is used
    instead of continuation.

    Since every candidate is always eligible and diagnosis["candidates"] is
    never empty, this always returns a decision — it no longer raises
    BlindSelectionError for "no eligible candidate" (Pass 1/Pass 3 validation
    failures can still raise elsewhere in the pipeline).
    """
    candidate_movement_families = candidate_movement_families or {}
    current = diagnosis["current"]
    candidates = diagnosis["candidates"]

    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (
            pair[1]["similarity_score"],
            pair[1]["confidence"],
            pair[1]["reference_match_confidence"],
            -pair[0],
        ),
        reverse=True,
    )
    best = ranked[0][1]

    current_satisfies_stop = (
        current["similarity_score"] >= stop_similarity_threshold
        and current["confidence"] >= minimum_confidence
    )
    best_improvement = best["similarity_score"] - current["similarity_score"]
    no_meaningful_improvement = best_improvement < stop_improvement_margin

    if current_satisfies_stop and no_meaningful_improvement:
        reason = (
            f"STOP: current render similarity_score={current['similarity_score']} "
            f"(>= threshold {stop_similarity_threshold}) with confidence="
            f"{current['confidence']} (>= minimum {minimum_confidence}); closest reference "
            f"{current['closest_reference_id']!r} ({current['closest_reference_description']!r}), "
            f"match_quality={current['reference_match_quality']} [{current['reference_match_differences']}]; "
            f"best candidate only improves similarity by {best_improvement} "
            f"(< margin {stop_improvement_margin})."
            f" difference_from_target: {current['difference_from_target']}"
        )
        return {
            "decision": "stop",
            "selected_candidate": "STOP",
            "selection_reason": reason,
            "selected_diagnosis": current,
        }

    all_unclear = all(c["reference_match_quality"] == "unclear" for c in candidates)
    if all_unclear:
        scalable_pool = [
            c for c in candidates
            if candidate_movement_families.get(c["candidate_id"]) in SCALABLE_MOVEMENT_FAMILIES
        ]
        pool = scalable_pool or candidates

        continuation = None
        if last_movement_family in SCALABLE_MOVEMENT_FAMILIES:
            for c in pool:
                if candidate_movement_families.get(c["candidate_id"]) == last_movement_family:
                    continuation = c
                    break

        chosen = continuation if continuation is not None else pool[0]

        if continuation is not None:
            reason = (
                f"All {len(candidates)} candidates have unclear reference-bank grounding; "
                f"continuing the same direction as the last applied movement "
                f"({last_movement_family!r}) rather than re-picking arbitrarily, to keep "
                f"sweeping toward a view the reference bank recognizes. Selected "
                f"{chosen['candidate_id']} (similarity_score={chosen['similarity_score']}, "
                f"confidence={chosen['confidence']})."
            )
        else:
            reason = (
                f"All {len(candidates)} candidates have unclear reference-bank grounding, and no "
                "prior movement direction was available to continue (first iteration, or the "
                f"last action had no continuable direction); arbitrarily picking "
                f"{chosen['candidate_id']} (similarity_score={chosen['similarity_score']}, "
                f"confidence={chosen['confidence']}) to keep exploring toward a view the "
                "reference bank recognizes."
            )

        return {
            "decision": "candidate",
            "selected_candidate": chosen["candidate_id"],
            "selection_reason": reason,
            "selected_diagnosis": chosen,
        }

    excluded_summary = [
        f"{c['candidate_id']} ranked lower (similarity_score={c['similarity_score']}, "
        f"confidence={c['confidence']}, reference_match_confidence={c['reference_match_confidence']})"
        for c in candidates
        if c["candidate_id"] != best["candidate_id"]
    ]

    reason = (
        f"Selected {best['candidate_id']}: closest_reference={best['closest_reference_id']!r} "
        f"({best['closest_reference_description']!r}), match_quality={best['reference_match_quality']} "
        f"[{best['reference_match_differences']}], similarity_score={best['similarity_score']}, "
        f"confidence={best['confidence']}. comparison_to_target: {best['comparison_to_target']}"
    )
    if excluded_summary:
        reason += " | " + "; ".join(excluded_summary)

    return {
        "decision": "candidate",
        "selected_candidate": best["candidate_id"],
        "selection_reason": reason,
        "selected_diagnosis": best,
    }


# ------------------------------------------------------------------
# Pass 3 validation
# ------------------------------------------------------------------

def _validate_magnitude_result(result: dict, expected_candidate_id: str) -> dict:
    """Validate the Pass 3 structured magnitude response. Raises
    BlindSelectionError if the echoed candidate doesn't exactly match the
    Python-selected candidate (Pass 3 must never be allowed to swap the
    selection), the magnitude isn't one of full/reduced/minimal, or either
    explanation string is missing/empty.
    """
    selected_candidate = result.get("selected_candidate")
    if not isinstance(selected_candidate, str):
        raise BlindSelectionError(f"magnitude response field 'selected_candidate' must be a string; got {selected_candidate!r}")
    if selected_candidate.strip().upper() != expected_candidate_id:
        raise BlindSelectionError(
            f"Pass 3 echoed candidate {selected_candidate!r}, which does not match the Python-selected "
            f"candidate {expected_candidate_id!r}. Pass 3 is not permitted to change the selection."
        )

    magnitude = _require_enum(result, "magnitude_recommendation", _MAGNITUDE_VALUES, "magnitude")
    reasoning = _require_str(result, "magnitude_reasoning", "magnitude")
    expected_visual_result = _require_str(result, "expected_visual_result", "magnitude")

    return {
        "selected_candidate": expected_candidate_id,
        "magnitude_recommendation": magnitude,
        "magnitude_reasoning": reasoning,
        "expected_visual_result": expected_visual_result,
    }


# ------------------------------------------------------------------
# Pass 3 (called only for scalable-family selections): ask the LLM to estimate
# magnitude for the already-selected candidate only
# ------------------------------------------------------------------

def estimate_selected_candidate_magnitude(
    current_image_path: str,
    selected_candidate_id: str,
    selected_candidate_image_path: str,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    diagnosis: Optional[dict] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Ask the LLM to estimate magnitude (full/reduced/minimal) for the
    already-selected candidate only. Only the current render, the target, and
    the selected candidate's own render are shown — no other candidates, and no
    reference-view images — so this pass cannot reconsider or replace Python's
    Pass 2 selection, and the reference-grounding decision from Pass 1 (already
    baked into that selection) is not re-litigated here.

    `diagnosis` (the Pass 1 diagnosis dict for this one candidate, if available)
    is accepted for future extensibility but is not currently interpolated into
    the prompt; the prompt is self-contained from the images alone.

    Returns {"magnitude": <validated magnitude dict>, "raw_response": str}.
    Raises BlindSelectionError if the response isn't valid JSON, doesn't match
    the schema, or echoes a different candidate ID than selected_candidate_id.
    """
    prompt = BLIND_MAGNITUDE_PROMPT_TEMPLATE.format(
        target_description=target_description,
        target_image_line=_format_target_image_line(target_image_path),
        selected_candidate_id=selected_candidate_id,
        neutral_history_block=_summarize_history_neutrally(action_history),
    )

    all_reference_items = [
        (selected_candidate_id, selected_candidate_image_path, BLIND_CANDIDATE_DESCRIPTION)
    ]
    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path,
        target_image_path=target_image_path,
        reference_items=all_reference_items,
        model=model,
    )

    try:
        raw_result = _parse_json_response(response_text)
        validated_result = _validate_magnitude_result(raw_result, selected_candidate_id)
    except BlindSelectionError:
        print(f"[blind_visual_rollout_agent] ERROR: Pass 3 magnitude estimation failed validation "
              f"for candidate {selected_candidate_id!r}.")
        print("[blind_visual_rollout_agent] Raw response was:")
        print(response_text)
        raise

    return {"magnitude": validated_result, "raw_response": response_text}


def _build_fixed_magnitude_result(selected_diagnosis: dict) -> dict:
    """Build the internal magnitude result for a FIXED_BLIND_ACTIONS candidate,
    without calling Pass 3 (there is no smaller/larger version of a fixed,
    single-outcome movement to interpolate to). Never exposes the real fixed
    action name — only reuses the Pass 1 diagnosis text already produced for
    this candidate, which is itself action-name-free.
    """
    return {
        "selected_candidate": selected_diagnosis["candidate_id"],
        "magnitude_recommendation": "not_applicable",
        "magnitude_reasoning": (
            "The selected candidate represents a fixed, single-outcome movement with no magnitude "
            "variants to interpolate between; magnitude estimation was skipped."
        ),
        "expected_visual_result": selected_diagnosis["comparison_to_target"],
    }


# ------------------------------------------------------------------
# Final private action construction
# ------------------------------------------------------------------

def _construct_final_action(
    selected_candidate: str,
    magnitude_recommendation: Optional[str],
    candidate_mapping: Dict[str, str],
) -> str:
    """Privately resolve the deterministic selection + magnitude result into a
    real, validated action name. Raises BlindSelectionError (never silently
    falls back to STOP, the first candidate, or an arbitrary/unvalidated
    action) if anything is invalid.
    """
    if selected_candidate in ("STOP", "UNDO_LAST"):
        return selected_candidate

    if selected_candidate not in candidate_mapping:
        raise BlindSelectionError(
            f"selected_candidate {selected_candidate!r} is not among this iteration's valid "
            f"candidate IDs: {sorted(candidate_mapping)}"
        )

    movement_family = candidate_mapping[selected_candidate]
    if movement_family in FIXED_BLIND_ACTIONS:
        # No magnitude variants exist for this one — Pass 3 is skipped entirely
        # for these candidates (see _build_fixed_magnitude_result).
        return movement_family

    if magnitude_recommendation not in SCALE_MAP:
        raise BlindSelectionError(
            f"invalid magnitude_recommendation {magnitude_recommendation!r} for candidate "
            f"{selected_candidate!r}; expected one of {sorted(SCALE_MAP)}"
        )

    final_action = f"{movement_family}_{SCALE_MAP[magnitude_recommendation]}"
    if final_action not in VALID_ACTIONS:
        raise BlindSelectionError(f"constructed action {final_action!r} is not in VALID_ACTIONS")
    return final_action


# ------------------------------------------------------------------
# Orchestration wrapper (called every iteration): runs all three passes and
# returns a rich, trace-friendly result
# ------------------------------------------------------------------

def select_action_from_blind_candidates(
    current_image_path: str,
    batch: BlindCandidateBatch,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_image_paths: Optional[Dict[str, str]] = None,
    node_descriptions: Optional[Dict[str, str]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    stop_similarity_threshold: int = 92,
    stop_improvement_margin: int = 3,
    minimum_confidence: float = 0.5,
    frozen_target: Optional[dict] = None,
    carried_current: Optional[dict] = None,
    sequential_diagnosis: bool = False,
) -> dict:
    """Run the full three-pass pipeline (LLM visual diagnosis, reference-
    grounded via an attached reference-view bank the LLM judges by eye ->
    deterministic Python selection -> LLM magnitude estimation) and privately
    resolve + validate the real action. Raises BlindSelectionError on any
    invalid/inconsistent response at any pass — never applies an arbitrary
    action.

    `reference_image_paths`/`node_descriptions` are used only by Pass 1, to
    build the attached reference-view bank (see diagnose_blind_candidates and
    module docstring, "Reference grounding"). They are NOT used by Pass 3,
    which only ever sees the current render, the target, and the
    already-selected candidate — no reference-view images or grounding text
    at all.

    `frozen_target`/`carried_current` are passed straight through to
    diagnose_blind_candidates (or diagnose_blind_candidates_sequentially) — see their
    docstrings.

    `sequential_diagnosis`, if True, runs Pass 1 via
    diagnose_blind_candidates_sequentially() instead of diagnose_blind_candidates() --
    one LLM call per candidate instead of one call for the whole batch. Costs more calls
    but eliminates cross-candidate attribution errors (the model mislabeling which
    candidate a score/description belongs to when several similar images are shown at
    once) -- see that function's docstring.

    Returns a rich dict:
        {"visual_diagnosis", "visual_diagnosis_raw_response",
         "selection", "magnitude", "magnitude_raw_response",
         "selected_candidate", "magnitude_recommendation", "final_action"}
    """
    diagnose_fn = diagnose_blind_candidates_sequentially if sequential_diagnosis else diagnose_blind_candidates
    diagnosis_result = diagnose_fn(
        current_image_path,
        batch,
        target_description,
        target_image_path=target_image_path,
        reference_image_paths=reference_image_paths,
        node_descriptions=node_descriptions,
        action_history=action_history,
        model=model,
        frozen_target=frozen_target,
        carried_current=carried_current,
    )
    diagnosis = diagnosis_result["diagnosis"]

    last_action = action_history[-1]["action"] if action_history else None
    try:
        selection_result = select_candidate_from_diagnosis(
            diagnosis,
            stop_similarity_threshold=stop_similarity_threshold,
            stop_improvement_margin=stop_improvement_margin,
            minimum_confidence=minimum_confidence,
            candidate_movement_families=batch.candidate_mapping,
            last_movement_family=_movement_family_from_applied_action(last_action),
        )
    except BlindSelectionError:
        print("[blind_visual_rollout_agent] ERROR: Pass 2 could not select a candidate.")
        print("[blind_visual_rollout_agent] Full Pass 1 diagnosis:")
        print(json.dumps(diagnosis, indent=2))
        raise

    magnitude_raw_response = None
    if selection_result["decision"] == "stop":
        magnitude_result = {
            "selected_candidate": "STOP",
            "magnitude_recommendation": "not_applicable",
            "magnitude_reasoning": selection_result["selection_reason"],
            "expected_visual_result": "The viewpoint remains unchanged.",
        }
        final_action = "STOP"
    else:
        selected_candidate_id = selection_result["selected_candidate"]
        selected_candidate_path = _find_candidate_image_path(batch, selected_candidate_id)

        if batch.candidate_mapping[selected_candidate_id] in FIXED_BLIND_ACTIONS:
            magnitude_result = _build_fixed_magnitude_result(selection_result["selected_diagnosis"])
        else:
            magnitude_pass = estimate_selected_candidate_magnitude(
                current_image_path,
                selected_candidate_id,
                selected_candidate_path,
                target_description,
                target_image_path=target_image_path,
                diagnosis=selection_result["selected_diagnosis"],
                action_history=action_history,
                model=model,
            )
            magnitude_result = magnitude_pass["magnitude"]
            magnitude_raw_response = magnitude_pass["raw_response"]

        final_action = _construct_final_action(
            selected_candidate_id,
            magnitude_result["magnitude_recommendation"],
            batch.candidate_mapping,
        )

    return {
        "visual_diagnosis": diagnosis,
        "visual_diagnosis_raw_response": diagnosis_result["raw_response"],
        "selection": selection_result,
        "magnitude": magnitude_result,
        "magnitude_raw_response": magnitude_raw_response,
        "selected_candidate": selection_result["selected_candidate"],
        "magnitude_recommendation": magnitude_result["magnitude_recommendation"],
        "final_action": final_action,
    }


# ------------------------------------------------------------------
# Console visibility for the validated reference-grounding fields — a
# consolidated summary alongside the full raw LLM JSON already printed above,
# since the raw response is easy to skim past when it's mixed in with every
# other diagnosis field.
# ------------------------------------------------------------------

def _format_visual_diagnosis_grounding(visual_diagnosis: dict) -> str:
    def _line(label: str, entry: dict) -> str:
        return (
            f"  {label}: closest_reference={entry['closest_reference_id']!r} "
            f"(quality={entry['reference_match_quality']}) "
            f"similarity_score={entry.get('similarity_score', 'n/a')} "
            f"-- {entry['closest_reference_description']}\n"
            f"    differences: {entry['reference_match_differences']}"
        )

    lines = [_line("target ", visual_diagnosis["target"]), _line("current", visual_diagnosis["current"])]
    for candidate in visual_diagnosis["candidates"]:
        lines.append(_line(f"[{candidate['candidate_id']}]", candidate))
    return "\n".join(lines)


# ------------------------------------------------------------------
# Trace saving (debugging — the private mapping is included here only, a local
# file, never sent to or logged as if visible to the LLM)
# ------------------------------------------------------------------

def save_blind_rollout_trace(
    trace_dir: str,
    iteration: int,
    current_image_path: str,
    batch: BlindCandidateBatch,
    selection: dict,
    applied_action: Optional[str],
) -> str:
    trace = {
        "iteration": iteration,
        "current_image_path": current_image_path,
        "blind_candidates": [
            {"candidate_id": cid, "image_path": path} for cid, path, _ in batch.visible_items
        ],
        "private_candidate_mapping": batch.candidate_mapping,
        "pass_1_visual_diagnosis": {
            "parsed": selection["visual_diagnosis"],
            "raw_response": selection["visual_diagnosis_raw_response"],
        },
        "pass_2_deterministic_selection": selection["selection"],
        "pass_3_magnitude_estimation": {
            "parsed": selection["magnitude"],
            "raw_response": selection["magnitude_raw_response"],
        },
        "final_action": selection["final_action"],
        "applied_action": applied_action,
    }
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"blind_rollout_{iteration:03d}.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    return str(trace_path)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run_blind_visual_rollout_alignment_loop(
    session,
    max_iterations: int = 10,
    trace_dir: Optional[str] = None,
    reference_image_paths: Optional[Dict[str, str]] = None,
    node_descriptions: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    on_iteration_end: Optional[Callable[[str], None]] = None,
    stop_similarity_threshold: int = 92,
    stop_improvement_margin: int = 3,
    minimum_confidence: float = 0.5,
    skip_on_indecision: bool = True,
    sequential_diagnosis: bool = False,
) -> dict:
    """Run the blind visual-rollout camera-action-selection loop against an
    already-`.initialize()`d CameraReasoningSession.

    `sequential_diagnosis`, if True, diagnoses candidates one LLM call at a time instead
    of all together in one call (see diagnose_blind_candidates_sequentially's docstring)
    -- more LLM calls per iteration, but avoids the model mislabeling which candidate a
    score/description belongs to when several similar images are shown at once.

    Every iteration: render the blind-eligible COARSE/fixed candidates (4
    scalable + len(FIXED_BLIND_ACTIONS) fixed, currently 7 total — including a
    STOP candidate whose "render" is just the current image)
    (render_candidate_rollouts, reused unmodified), assign them random opaque
    IDs and shuffle (prepare_blind_candidates), then run the three-pass pipeline
    (select_action_from_blind_candidates: LLM visual diagnosis, reference-
    grounded via an attached reference-view bank the LLM itself judges by eye
    -> deterministic Python selection -> LLM magnitude estimation), privately
    resolve + validate the real action, then apply it via the existing
    session.process_chatgpt_response() path. Stops early on STOP, or after
    max_iterations.

    `reference_image_paths` (a plain node_id -> image_path dict — no image
    embeddings/CLIP involved anywhere in this module, and deliberately
    decoupled from any particular dataset format; a CameraSpatialGraph bank or
    a flat photo bank with no graph/edges concept both just need to be reduced
    to this mapping) and `node_descriptions` (node_id -> deterministic
    description) together build the attached reference-view bank: for the
    target (if a target image is set), the current render, and every
    candidate, Pass 1 visually reports which reference view (if any) it
    resembles, how closely, and what's different when it isn't an exact match
    (see module docstring, "Reference grounding"). Passing
    reference_image_paths=None is allowed but disables the reference bank
    entirely for this run (every image's `closest_reference_id` will be
    None/"unclear").

    `skip_on_indecision` (default True) controls what happens when Pass 1 or
    Pass 3 return an invalid/inconsistent response (e.g. malformed JSON, a
    missing/duplicate candidate ID, or Pass 3 echoing the wrong candidate) —
    Pass 2 itself now always produces a decision (every candidate is eligible,
    see select_candidate_from_diagnosis), so this no longer triggers on
    indecision at the selection stage. When it does trigger: the full Pass 1
    diagnosis (if one was produced) is printed, NO action is applied this
    iteration (the camera state is left untouched — this is never a silent
    fallback to an arbitrary action), and the loop moves on to the next
    iteration with fresh candidates. Set to False to instead let
    BlindSelectionError propagate and abort the whole run.

    Target/current consistency across iterations: the target image never
    changes mid-run, but asking the LLM to redescribe it fresh every
    iteration risked a different view_description/closest_reference_id
    purely from sampling noise. So Pass 1's target diagnosis is frozen after
    the FIRST iteration (frozen_target_diagnosis below) and reused for every
    iteration after that — the LLM is still asked, its answer is just
    discarded once we have one. Similarly, once iteration i actually applies
    a (non-STOP) action, the render that becomes iteration i+1's "current" is
    exactly the candidate that was selected — so instead of re-diagnosing
    that same image from scratch (which could disagree with what was already
    said about it as a candidate), its own diagnosis is carried forward via
    candidate_diagnosis_to_current() and reused as "current" next iteration.
    """
    if trace_dir is None:
        trace_dir = str(Path(session.output_dir) / "blind_rollout_traces")

    applied_actions = []
    frozen_target_diagnosis: Optional[dict] = None
    carried_current_diagnosis: Optional[dict] = None
    for i in range(max_iterations):
        current_image_path = session.render_and_save()
        print(f"\n[blind_visual_rollout_agent] Iteration {i}", flush=True)

        rollouts = render_candidate_rollouts(session, current_image_path, action_subset=BLIND_CANDIDATE_ACTION_SUBSET)
        blind_temp_dir = str(Path(session.output_dir) / "screenshots" / "blind_candidates" / f"step_{session._step:03d}")
        batch = prepare_blind_candidates(rollouts, blind_temp_dir)
        print(
            f"[blind_visual_rollout_agent] Prepared {len(batch.visible_items)} blind candidates "
            f"(shuffled, private mapping not logged): {[cid for cid, _, _ in batch.visible_items]}",
            flush=True,
        )

        try:
            selection = select_action_from_blind_candidates(
                current_image_path,
                batch,
                session.target_description,
                target_image_path=session.target_image_path,
                reference_image_paths=reference_image_paths,
                node_descriptions=node_descriptions,
                action_history=session._action_history,
                model=model,
                stop_similarity_threshold=stop_similarity_threshold,
                stop_improvement_margin=stop_improvement_margin,
                minimum_confidence=minimum_confidence,
                frozen_target=frozen_target_diagnosis,
                carried_current=carried_current_diagnosis,
                sequential_diagnosis=sequential_diagnosis,
            )
        except BlindSelectionError as exc:
            if not skip_on_indecision:
                raise
            print(f"[blind_visual_rollout_agent] SKIPPING iteration {i} — no action applied: {exc}", flush=True)
            continue

        if frozen_target_diagnosis is None:
            frozen_target_diagnosis = selection["visual_diagnosis"]["target"]
            print(
                "[blind_visual_rollout_agent] Froze target diagnosis for the rest of this run "
                f"(closest_reference_id={frozen_target_diagnosis['closest_reference_id']!r}).",
                flush=True,
            )

        print("[blind_visual_rollout_agent] PASS 1 — VISUAL DIAGNOSIS", flush=True)
        print(selection["visual_diagnosis_raw_response"], flush=True)

        print("\n[blind_visual_rollout_agent] PASS 1 — REFERENCE GROUNDING (computed by Python via image "
              "embeddings; not part of the LLM's own response above)", flush=True)
        print(_format_visual_diagnosis_grounding(selection["visual_diagnosis"]), flush=True)

        print("\n[blind_visual_rollout_agent] PASS 2 — DETERMINISTIC SELECTION", flush=True)
        print(f"  decision: {selection['selection']['decision']}", flush=True)
        print(f"  selected_candidate: {selection['selected_candidate']}", flush=True)
        print(f"  selection_reason: {selection['selection']['selection_reason']}", flush=True)

        print("\n[blind_visual_rollout_agent] PASS 3 — MAGNITUDE ESTIMATION", flush=True)
        if selection["magnitude_raw_response"] is not None:
            print(selection["magnitude_raw_response"], flush=True)
        else:
            print(f"  (skipped: {selection['magnitude']['magnitude_reasoning']})", flush=True)
        print(f"  magnitude_recommendation: {selection['magnitude_recommendation']}", flush=True)

        print("\n[blind_visual_rollout_agent] FINAL RESOLVED ACTION", flush=True)
        print(f"  final_action: {selection['final_action']}", flush=True)

        applied_action = None
        if dry_run:
            print("[blind_visual_rollout_agent] Dry run — action NOT applied.", flush=True)
        else:
            applied_action = session.process_chatgpt_response(f"Next action:\n{selection['final_action']}")
            applied_actions.append(applied_action)

            if applied_action != "STOP":
                carried_current_diagnosis = candidate_diagnosis_to_current(
                    selection["selection"]["selected_diagnosis"]
                )
                print(
                    "[blind_visual_rollout_agent] Carrying selected candidate's diagnosis forward "
                    "as next iteration's 'current' (will not be re-diagnosed from scratch).",
                    flush=True,
                )

        if trace_dir:
            trace_path = save_blind_rollout_trace(
                trace_dir, i, current_image_path, batch, selection, applied_action
            )
            print(f"[blind_visual_rollout_agent] Saved trace: {trace_path}", flush=True)
            if on_iteration_end:
                on_iteration_end(trace_path)

        if not dry_run and applied_action == "STOP":
            break

    return {"mode": "blind_visual_rollout_alignment", "applied_actions": applied_actions, "dry_run": dry_run}