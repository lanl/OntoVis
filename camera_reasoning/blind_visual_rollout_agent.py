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
visually observed trajectory. Plus two fixed, single-outcome candidates with no
magnitude variants at all — AZIMUTH_RIGHT_180, ELEVATION_UP_180 (see
FIXED_BLIND_ACTIONS) — whose magnitude cannot be varied, so Pass 3 is skipped for
them entirely. STOP remains directly selectable (deterministically, by Python —
see "Three-pass architecture" below); UNDO_LAST is not selectable by either LLM
pass in this refactor (see "UNDO_LAST" below).

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

                     As of this revision, Pass 1 is additionally REFERENCE-
                     GROUNDED (see "Reference grounding" below): rather than
                     inventing its own left/right/frontal/etc. labels, it must
                     visually match each image (target, current, and every
                     candidate) to the single closest supplied reference view
                     and copy that reference's deterministic, camera-metadata-
                     derived description verbatim as the primary orientation
                     anchor for its diagnosis.

  Pass 2 (Python):  select_candidate_from_diagnosis()
                     Pure Python, no LLM call. Deterministically selects a
                     candidate from Pass 1's structured diagnosis using an
                     eligibility rule (target_progress == "toward") and a
                     numeric ranking (similarity_score, then confidence, then
                     reference_match_confidence, then stable order) — or
                     deterministically returns STOP if the current render
                     already satisfies fixed similarity/confidence thresholds
                     and no candidate offers a meaningful improvement. A
                     candidate diagnosed as "away" can never be selected, no
                     matter how high its similarity_score is: the categorical
                     diagnosis is authoritative over the numeric score.

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

The reference views supplied via `reference_items` (typically the calibrated
camera-relative graph bank in reference_views_relative/) each carry a
deterministic description generated OFFLINE from known camera metadata — not
guessed by an LLM. Because that description is anchored to an actual calibrated
camera position, it is a far more reliable orientation label than anything an
LLM could invent by eyeballing a single render in isolation.

Pass 1 is required to use these descriptions as its primary orientation anchor,
via this reasoning order for the target, the current render, and every
candidate:

    image
      -> visually compare against every supplied reference image
      -> pick the single closest reference (by viewpoint geometry, not
         appearance, filename, or ID text)
      -> copy that reference's description EXACTLY (never paraphrased)
      -> describe how the image differs from that reference, if at all
      -> use this reference-grounded diagnosis when judging target progress

`closest_reference_description` must be either the verbatim catalog string for
`closest_reference_id`, or exactly "unclear" when `closest_reference_id` is
`None` (used only when no supplied reference is a reasonable match). The
validator (`_validate_visual_diagnosis`) enforces the exact-match requirement
and never silently repairs a mismatched or invented description — it raises
BlindSelectionError instead.

Reference matching in this module is visual-only: the model is never told how
the references were generated, their camera positions, azimuth/elevation
values, parent/child relationships, or the movement family used to render them.
The reference catalog it receives is exactly the (reference_id -> description)
pairs already being sent as `reference_items` to `ask_chatgpt` — build with
`_build_reference_catalog()`.

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
# image, but have no magnitude variants to interpolate — there is no "reduced" or
# "minimal" 180-degree rotation, so Pass 3 (magnitude estimation) is skipped
# entirely for these. Their movement_family IS the final action, verbatim.
FIXED_BLIND_ACTIONS = {"AZIMUTH_RIGHT_180", "ELEVATION_UP_180"}

BLIND_CANDIDATE_ACTION_SUBSET = {f"{family}_COARSE" for family in SCALABLE_MOVEMENT_FAMILIES} | FIXED_BLIND_ACTIONS

SCALE_MAP = {"full": "COARSE", "reduced": "MEDIUM", "minimal": "FINE"}

# Uniform, non-descriptive caption used for every blind candidate — identical
# text for all of them, so it carries no distinguishing information.
BLIND_CANDIDATE_DESCRIPTION = "A full-magnitude candidate render generated from the current camera state."

_CANDIDATE_ID_ALPHABET = string.ascii_uppercase + string.digits

_TARGET_PROGRESS_VALUES = {"toward", "away", "ambiguous", "no_change"}
_MAGNITUDE_VALUES = {"full", "reduced", "minimal"}

# reference_id -> deterministic, camera-metadata-derived description. Built from
# the same (label, image_path, description) triples already passed as
# `reference_items` to ask_chatgpt — see _build_reference_catalog().
ReferenceCatalog = Dict[str, str]


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
# Reference catalog (deterministic, camera-metadata-derived descriptions used
# to ground Pass 1's orientation diagnosis)
# ------------------------------------------------------------------

def _build_reference_catalog(
    reference_items: Optional[List[Tuple[str, str, str]]],
) -> ReferenceCatalog:
    """Build a validated {reference_id: description} catalog from the same
    (reference_id, image_path, description) triples already passed as
    `reference_items` to ask_chatgpt.

    The description text is passed through completely unchanged — it is never
    regenerated or paraphrased by an LLM here; it is expected to already be the
    deterministic, camera-metadata-derived description produced offline (e.g.
    by graph_view_description_generator.py).

    Raises ValueError if any reference_id is empty/duplicate, any description is
    empty, or any image_path does not exist on disk.
    """
    catalog: ReferenceCatalog = {}
    if not reference_items:
        return catalog

    for reference_id, image_path, description in reference_items:
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError(f"reference_items contains an empty/invalid reference_id: {reference_id!r}")
        if reference_id in catalog:
            raise ValueError(f"reference_items contains a duplicate reference_id: {reference_id!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"reference_items[{reference_id!r}] has an empty description")
        if not Path(image_path).exists():
            raise ValueError(f"reference_items[{reference_id!r}] image_path does not exist: {image_path!r}")

        catalog[reference_id] = description

    return catalog


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


def _format_reference_items_line(reference_items: Optional[List[Tuple[str, str, str]]]) -> str:
    if reference_items:
        return (
            "  Labeled reference-view images are also attached above the candidates below, each in the\n"
            "   form \"[REFERENCE_VIEW] reference_id: <id> description: <deterministic description>\"\n"
            "   followed by its image. These descriptions were generated offline from known, calibrated\n"
            "   camera positions — they are NOT LLM guesses and are NOT camera-action instructions.\n"
        )
    return ""


def _format_reference_catalog_block(reference_catalog: ReferenceCatalog) -> str:
    if not reference_catalog:
        return "No reference views were supplied for this iteration."
    lines = ["Supplied reference catalog (reference_id -> deterministic description):"]
    for reference_id, description in reference_catalog.items():
        lines.append(f'  - {reference_id}: "{description}"')
    return "\n".join(lines)


# ------------------------------------------------------------------
# Pass 1 prompt: reference-grounded visual diagnosis only (no selection, no
# magnitude, no STOP, no UNDO_LAST)
# ------------------------------------------------------------------

BLIND_VISUAL_DIAGNOSIS_PROMPT_TEMPLATE = """You are visually diagnosing a set of rendered camera viewpoints by comparing
them to a target viewpoint and to a supplied set of reference views — never by
reasoning about labels, identifiers, or action-name words.

This is a DIAGNOSIS STAGE ONLY. Do not select a candidate. Do not recommend a
magnitude. Do not output STOP. Do not output UNDO_LAST. The candidate selection
will be performed deterministically by Python from your structured diagnosis
below — your only job is to produce an honest, structured, human-readable visual
diagnosis of what is actually visible in each image.

REFERENCE GROUNDING — READ CAREFULLY

The supplied reference views are orientation anchors. Each reference image has a
deterministic description generated from its known, calibrated camera position.
Treat that description as authoritative for the reference image — it was NOT
guessed by an LLM, and it is more reliable than any orientation label you could
invent yourself from a single render.

{reference_catalog_block}

Before diagnosing the target, the current render, or ANY candidate, follow this
exact procedure for that image:

1. Inspect the image.
2. Compare it visually against every supplied reference image.
3. Identify the single reference view with the most similar GLOBAL viewpoint
   geometry (not appearance, not lighting, not local anatomy).
4. Copy that reference's supplied description EXACTLY, character for character,
   into `closest_reference_description`. Do not paraphrase it, reword it,
   summarize it, or "improve" it.
5. Explain the visual evidence for the match in `reference_match_evidence`.
6. Describe any visible difference between the analyzed image and the matched
   reference in `view_description` (e.g. "closest to the left lateral
   reference, but slightly more oblique").
7. Use this reference-grounded diagnosis — not an independently invented
   left/right/frontal/posterior/superior/inferior/oblique label — as the basis
   for comparing the candidate with the target.

Do NOT infer orientation from candidate IDs, reference IDs, filenames, node
order, graph position, parent IDs, known camera actions, or any assumption
about how the references were generated. Reference matching must use image
content only. A reference ID such as "view_003" is an opaque identifier; its
number carries no semantic orientation information.

Do NOT independently invent a contradictory orientation label after selecting a
reference. The selected reference's description is the primary orientation
anchor for that image — your own `view_description` may only note deviations
from it, never replace it with a different, self-invented direction claim.

If no supplied reference is a reasonable match for an image, set
`closest_reference_id` to null and `closest_reference_description` to exactly
"unclear" — but only when no reference is reasonably similar. If one reference
is clearly the closest even though it is not an exact match, select it and
describe the difference in `view_description` rather than falling back to null.

You are given, in order:

1. The CURRENT rendered image before any movement.
2. The target view: {target_description}
   {target_image_line}{reference_items_line}  3. Several candidate images, each preceded by an opaque identifier such as
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

GLOBAL VIEWPOINT REASONING (used for reference matching AND for judging
candidate/target similarity)

To determine which reference is closest, and to judge similarity to the target,
compare global viewpoint evidence such as:

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
small feature. Do NOT choose a closest reference, or reward a candidate, because:

* one tooth, cavity, or edge is clearer or sharper;
* the lighting or rendering color happens to be similar;
* the object instance has more similar local anatomy;
* one isolated landmark happens to look alike;
* more detail is visible or the image is easier to interpret.

Reference matching and target-progress judgments must prioritize camera
viewpoint over appearance. When the target image comes from another source,
rendering style, object instance, lighting condition, or dataset, ignore
lighting, texture, color, exact object shape, rendering style, image quality,
and missing/damaged local anatomy — use only viewpoint-relevant geometry.

TARGET-PROGRESS RULES

* When a candidate's closest reference is the SAME reference as the target's
  closest reference, this is strong evidence of target progress (`toward`) —
  but you must still check whether the candidate is actually closer to the
  target than the current render already was; matching the same broad
  reference class does not automatically mean the candidate improved if the
  current render was already at least as close.
* When a candidate's closest reference is geometrically OPPOSITE to the
  target's closest reference orientation, this is strong evidence that the
  candidate moves away (`away`).
* A candidate that visibly faces the opposite direction from the target should
  normally be diagnosed with target_progress "away".
* Do not force positive evidence for every candidate — if there is no evidence
  that a candidate helps, state that plainly instead of rationalizing it.
* Reference-view images are visual anchors only. They show what certain views
  look like, not which action should be selected.

Respond with STRICT JSON ONLY — no prose before or after the JSON, no Markdown
commentary outside the JSON itself. You may wrap the JSON in a single Markdown
code fence if you prefer, but nothing else may appear in your response.

Use exactly this JSON structure:

{{
  "target": {{
    "closest_reference_id": "view_000",
    "closest_reference_description": "Exact copy of that reference's supplied description.",
    "reference_match_confidence": 0.0,
    "reference_match_evidence": "Visual evidence supporting the reference match.",
    "view_description": "How the target compares to the matched reference, including any deviation."
  }},
  "current": {{
    "closest_reference_id": "view_001",
    "closest_reference_description": "Exact copy of that reference's supplied description.",
    "reference_match_confidence": 0.0,
    "reference_match_evidence": "Visual evidence supporting the reference match.",
    "view_description": "How the current render compares to the matched reference, including any deviation.",
    "difference_from_target": "Direct visual comparison with the target.",
    "similarity_score": 0,
    "confidence": 0.0
  }},
  "candidates": [
    {{
      "candidate_id": "CANDIDATE_XXXXX",
      "closest_reference_id": "view_002",
      "closest_reference_description": "Exact copy of that reference's supplied description.",
      "reference_match_confidence": 0.0,
      "reference_match_evidence": "Visual evidence supporting the reference match.",
      "view_description": "How this candidate compares to the matched reference, including any deviation.",
      "comparison_to_target": "Direct comparison between this candidate's reference-grounded diagnosis and the target's.",
      "target_progress": "toward | away | ambiguous | no_change",
      "similarity_score": 0,
      "confidence": 0.0
    }}
  ]
}}

Field requirements:

* `closest_reference_id` must be either one of the supplied reference IDs, or
  `null` if (and only if) no reference is reasonably similar.
* `closest_reference_description` must be the EXACT supplied description string
  for `closest_reference_id` when it is not null, or exactly "unclear" when
  `closest_reference_id` is null. Never paraphrase the canonical description.
* `reference_match_confidence` (target, current, and every candidate) must be a
  number from 0.0 to 1.0.
* `reference_match_evidence` and `view_description` must be non-empty and
  written so a human reviewing this diagnosis later can understand your
  reasoning without seeing the images.
* `similarity_score` (target is not scored; current and every candidate) must
  be an integer from 0 to 100, where 100 means the viewpoint visually matches
  the target's side/orientation as closely as possible and 0 means it is
  maximally different.
* `confidence` (current and every candidate) must be a number from 0.0 to 1.0.
* Every candidate ID shown to you must appear exactly once in "candidates".
* Do not invent candidate IDs that were not shown to you.
* Do not omit any candidate ID that was shown to you.
* Do not duplicate any candidate ID.
* Do not include a `facing_direction`, `vertical_orientation`, or
  `target_orientation_match` field anywhere — orientation must come only from
  `closest_reference_description`, `reference_match_evidence`, and
  `view_description`.
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


def _validate_reference_grounding(
    container: dict,
    context: str,
    valid_reference_catalog: ReferenceCatalog,
) -> dict:
    """Validate the reference-grounding fields (closest_reference_id,
    closest_reference_description, reference_match_confidence,
    reference_match_evidence, view_description) shared by target/current/every
    candidate. Raises BlindSelectionError on an invented reference ID, a
    mismatched/paraphrased description, or an inconsistent null/"unclear" pair
    — never silently corrects any of these.
    """
    if "closest_reference_id" not in container:
        raise BlindSelectionError(f"{context}: missing required field 'closest_reference_id'")
    raw_reference_id = container["closest_reference_id"]

    if raw_reference_id is None:
        closest_reference_id = None
        expected_description = "unclear"
    else:
        if not isinstance(raw_reference_id, str) or not raw_reference_id.strip():
            raise BlindSelectionError(
                f"{context}: field 'closest_reference_id' must be null or a non-empty string; "
                f"got {raw_reference_id!r}"
            )
        if raw_reference_id not in valid_reference_catalog:
            raise BlindSelectionError(
                f"{context}: closest_reference_id {raw_reference_id!r} is not among the supplied "
                f"reference catalog: {sorted(valid_reference_catalog)}"
            )
        closest_reference_id = raw_reference_id
        expected_description = valid_reference_catalog[raw_reference_id]

    closest_reference_description = _require_str(container, "closest_reference_description", context)
    if closest_reference_description != expected_description:
        raise BlindSelectionError(
            f"{context}: 'closest_reference_description' must exactly equal the catalog description "
            f"for closest_reference_id={closest_reference_id!r}. Expected {expected_description!r}, "
            f"got {closest_reference_description!r}. Paraphrasing the canonical description is not allowed."
        )

    return {
        "closest_reference_id": closest_reference_id,
        "closest_reference_description": closest_reference_description,
        "reference_match_confidence": _require_confidence(container, "reference_match_confidence", context),
        "reference_match_evidence": _require_str(container, "reference_match_evidence", context),
        "view_description": _require_str(container, "view_description", context),
    }


def _validate_visual_diagnosis(
    diagnosis: dict,
    valid_candidate_ids: Set[str],
    valid_reference_catalog: ReferenceCatalog,
) -> dict:
    """Validate the Pass 1 structured diagnosis. Raises BlindSelectionError on
    any missing field, out-of-range score, invalid enum value, candidate-set
    mismatch (missing, duplicate, or unknown candidate IDs), invented reference
    ID, or mismatched/paraphrased reference description. Never repairs a
    semantically invalid diagnosis — only well-formed, complete diagnoses pass.

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
    validated_target = _validate_reference_grounding(target, "target", valid_reference_catalog)

    current = diagnosis["current"]
    validated_current = {
        **_validate_reference_grounding(current, "current", valid_reference_catalog),
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
                **_validate_reference_grounding(candidate, context, valid_reference_catalog),
                "comparison_to_target": _require_str(candidate, "comparison_to_target", context),
                "target_progress": _require_enum(
                    candidate, "target_progress", _TARGET_PROGRESS_VALUES, context
                ),
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

def diagnose_blind_candidates(
    current_image_path: str,
    batch: BlindCandidateBatch,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Ask the LLM to produce a structured, reference-grounded visual diagnosis
    of the target, the current render, and every opaque candidate — with no
    candidate selection, magnitude recommendation, or STOP/UNDO_LAST anywhere in
    this pass.

    `reference_items` (label, image_path, description) triples are used both as
    ask_chatgpt attachments AND as the authoritative reference catalog the model
    must ground its diagnosis in (see module docstring, "Reference grounding").
    The descriptions are passed through unchanged — never regenerated here.

    Returns {"diagnosis": <validated diagnosis dict>, "raw_response": str}.
    Raises BlindSelectionError if the response isn't valid JSON, doesn't match
    the required schema, doesn't cover exactly this iteration's candidate ID set
    (missing, duplicate, or invented IDs), or grounds any image to an invented
    or mismatched/paraphrased reference.
    """
    reference_catalog = _build_reference_catalog(reference_items)

    prompt = BLIND_VISUAL_DIAGNOSIS_PROMPT_TEMPLATE.format(
        target_description=target_description,
        target_image_line=_format_target_image_line(target_image_path),
        reference_items_line=_format_reference_items_line(reference_items),
        reference_catalog_block=_format_reference_catalog_block(reference_catalog),
        neutral_history_block=_summarize_history_neutrally(action_history),
    )

    all_reference_items = list(reference_items or []) + batch.visible_items
    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path,
        target_image_path=target_image_path,
        reference_items=all_reference_items,
        model=model,
    )

    try:
        raw_diagnosis = _parse_json_response(response_text)
        validated_diagnosis = _validate_visual_diagnosis(
            raw_diagnosis, set(batch.candidate_mapping), reference_catalog
        )
    except BlindSelectionError:
        print("[blind_visual_rollout_agent] ERROR: Pass 1 visual diagnosis failed validation.")
        print(f"[blind_visual_rollout_agent] Currently valid candidate IDs: {sorted(batch.candidate_mapping)}")
        print(f"[blind_visual_rollout_agent] Supplied reference catalog: {reference_catalog}")
        print("[blind_visual_rollout_agent] Raw response was:")
        print(response_text)
        raise

    return {"diagnosis": validated_diagnosis, "raw_response": response_text}


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
) -> dict:
    """Deterministically select a candidate (or STOP) from a validated Pass 1
    diagnosis. Never calls an LLM.

    Eligibility: only candidates diagnosed with target_progress == "toward" are
    eligible for selection. Candidates diagnosed "away", "ambiguous", or
    "no_change" are never selected, regardless of similarity_score — a high
    numeric score can never override the categorical diagnosis. This applies
    equally when the diagnosis is reference-grounded: a candidate whose closest
    reference is geometrically opposite to the target's closest reference must
    still be diagnosed "away" by Pass 1, and a high similarity_score cannot
    rescue it here either.

    Ranking among eligible candidates: highest similarity_score first, then
    highest confidence, then highest reference_match_confidence, then the
    candidate's position in diagnosis["candidates"] as a stable final
    tie-breaker.

    STOP: returned only when the current render's similarity_score is at least
    stop_similarity_threshold AND its confidence is at least minimum_confidence
    AND no eligible ("toward") candidate improves on the current render's
    similarity_score by at least stop_improvement_margin. STOP is never returned
    merely because there are no eligible candidates unless the current render
    itself actually satisfies these thresholds.

    Raises BlindSelectionError if there is no eligible candidate and the current
    render does not satisfy the STOP criteria either — never falls back to an
    "ambiguous" or "away" candidate.
    """
    current = diagnosis["current"]
    candidates = diagnosis["candidates"]

    eligible = [c for c in candidates if c["target_progress"] == "toward"]
    eligible_ranked = sorted(
        enumerate(eligible),
        key=lambda pair: (
            pair[1]["similarity_score"],
            pair[1]["confidence"],
            pair[1]["reference_match_confidence"],
            -pair[0],
        ),
        reverse=True,
    )
    best_eligible = eligible_ranked[0][1] if eligible_ranked else None

    current_satisfies_stop = (
        current["similarity_score"] >= stop_similarity_threshold
        and current["confidence"] >= minimum_confidence
    )
    best_improvement = (
        (best_eligible["similarity_score"] - current["similarity_score"]) if best_eligible else 0
    )
    no_meaningful_improvement = best_improvement < stop_improvement_margin

    if current_satisfies_stop and no_meaningful_improvement:
        reason = (
            f"STOP: current render similarity_score={current['similarity_score']} "
            f"(>= threshold {stop_similarity_threshold}) with confidence="
            f"{current['confidence']} (>= minimum {minimum_confidence}); closest reference "
            f"{current['closest_reference_id']!r} ({current['closest_reference_description']!r}); "
            + (
                f"best eligible candidate only improves similarity by "
                f"{best_improvement} (< margin {stop_improvement_margin})."
                if best_eligible
                else "no candidate was diagnosed as making progress toward the target."
            )
            + f" difference_from_target: {current['difference_from_target']}"
        )
        return {
            "decision": "stop",
            "selected_candidate": "STOP",
            "selection_reason": reason,
            "selected_diagnosis": current,
        }

    if best_eligible is None:
        raise BlindSelectionError(
            "no candidate was diagnosed with target_progress='toward', and the current render does "
            f"not satisfy the STOP criteria (similarity_score={current['similarity_score']}, "
            f"confidence={current['confidence']}, thresholds: similarity>="
            f"{stop_similarity_threshold}, confidence>={minimum_confidence}). Refusing to select an "
            "'away' or 'ambiguous' candidate as a fallback."
        )

    excluded_summary = []
    for candidate in candidates:
        if candidate["candidate_id"] == best_eligible["candidate_id"]:
            continue
        if candidate["target_progress"] != "toward":
            excluded_summary.append(
                f"{candidate['candidate_id']} excluded (target_progress={candidate['target_progress']!r}, "
                f"closest_reference={candidate['closest_reference_id']!r}, "
                f"similarity_score={candidate['similarity_score']})"
            )
        else:
            excluded_summary.append(
                f"{candidate['candidate_id']} ranked lower (target_progress=toward, "
                f"similarity_score={candidate['similarity_score']}, confidence={candidate['confidence']}, "
                f"reference_match_confidence={candidate['reference_match_confidence']})"
            )

    reason = (
        f"Selected {best_eligible['candidate_id']}: target_progress=toward, "
        f"closest_reference={best_eligible['closest_reference_id']!r} "
        f"({best_eligible['closest_reference_description']!r}), "
        f"similarity_score={best_eligible['similarity_score']}, confidence={best_eligible['confidence']}, "
        f"reference_match_confidence={best_eligible['reference_match_confidence']}. "
        f"comparison_to_target: {best_eligible['comparison_to_target']}"
    )
    if excluded_summary:
        reason += " | " + "; ".join(excluded_summary)

    return {
        "decision": "candidate",
        "selected_candidate": best_eligible["candidate_id"],
        "selection_reason": reason,
        "selected_diagnosis": best_eligible,
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
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
    stop_similarity_threshold: int = 92,
    stop_improvement_margin: int = 3,
    minimum_confidence: float = 0.5,
) -> dict:
    """Run the full three-pass pipeline (LLM reference-grounded diagnosis ->
    deterministic Python selection -> LLM magnitude estimation) and privately
    resolve + validate the real action. Raises BlindSelectionError on any
    invalid/inconsistent response at any pass — never applies an arbitrary
    action.

    `reference_items` is passed through to Pass 1 both as ask_chatgpt
    attachments and as the authoritative reference catalog Pass 1 must ground
    its diagnosis in. It is NOT passed to Pass 3, which only ever sees the
    current render, the target, and the already-selected candidate.

    Returns a rich dict:
        {"visual_diagnosis", "visual_diagnosis_raw_response",
         "selection", "magnitude", "magnitude_raw_response",
         "selected_candidate", "magnitude_recommendation", "final_action"}
    """
    diagnosis_result = diagnose_blind_candidates(
        current_image_path,
        batch,
        target_description,
        target_image_path=target_image_path,
        reference_items=reference_items,
        action_history=action_history,
        model=model,
    )
    diagnosis = diagnosis_result["diagnosis"]

    try:
        selection_result = select_candidate_from_diagnosis(
            diagnosis,
            stop_similarity_threshold=stop_similarity_threshold,
            stop_improvement_margin=stop_improvement_margin,
            minimum_confidence=minimum_confidence,
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
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    on_iteration_end: Optional[Callable[[str], None]] = None,
    stop_similarity_threshold: int = 92,
    stop_improvement_margin: int = 3,
    minimum_confidence: float = 0.5,
    skip_on_indecision: bool = True,
) -> dict:
    """Run the blind visual-rollout camera-action-selection loop against an
    already-`.initialize()`d CameraReasoningSession.

    Every iteration: render the 6 blind-eligible COARSE/fixed candidates
    (render_candidate_rollouts, reused unmodified), assign them random opaque
    IDs and shuffle (prepare_blind_candidates), then run the three-pass pipeline
    (select_action_from_blind_candidates: LLM reference-grounded diagnosis ->
    deterministic Python selection -> LLM magnitude estimation), privately
    resolve + validate the real action, then apply it via the existing
    session.process_chatgpt_response() path. Stops early on STOP, or after
    max_iterations.

    `reference_items` should be the calibrated reference-view graph bank (e.g.
    reference_views_relative/) — Pass 1 uses it both as visual attachments and
    as the authoritative reference catalog it must ground its diagnosis in (see
    module docstring, "Reference grounding"). Passing an empty/None
    reference_items is allowed but defeats the purpose of this revision: Pass 1
    will have no reference catalog to ground against and every image's
    `closest_reference_id` will necessarily be null/"unclear".

    `skip_on_indecision` (default True) controls what happens when Pass 2 can't
    select any candidate for this iteration (no candidate diagnosed "toward" and
    the current render doesn't satisfy the STOP thresholds either) or when Pass 1
    /Pass 3 return an invalid/inconsistent response: the full Pass 1 diagnosis
    (if one was produced) is printed, NO action is applied this iteration (the
    camera state is left untouched — this is never a silent fallback to an
    arbitrary action), and the loop moves on to the next iteration with fresh
    candidates. Set to False to instead let BlindSelectionError propagate and
    abort the whole run, e.g. for unattended runs where an indecisive iteration
    should be treated as a hard failure rather than skipped.
    """
    if trace_dir is None:
        trace_dir = str(Path(session.output_dir) / "blind_rollout_traces")

    applied_actions = []
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
                reference_items=reference_items,
                action_history=session._action_history,
                model=model,
                stop_similarity_threshold=stop_similarity_threshold,
                stop_improvement_margin=stop_improvement_margin,
                minimum_confidence=minimum_confidence,
            )
        except BlindSelectionError as exc:
            if not skip_on_indecision:
                raise
            print(f"[blind_visual_rollout_agent] SKIPPING iteration {i} — no action applied: {exc}", flush=True)
            continue

        print("[blind_visual_rollout_agent] PASS 1 — VISUAL DIAGNOSIS", flush=True)
        print(selection["visual_diagnosis_raw_response"], flush=True)

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
