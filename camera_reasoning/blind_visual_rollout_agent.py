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
It selects an opaque ID plus a qualitative magnitude ("full"/"reduced"/"minimal").
Python alone holds the private candidate_id -> real movement family mapping, and
resolves the LLM's selection into a real, VALID_ACTIONS-checked action name
(AZIMUTH_LEFT_COARSE, AZIMUTH_LEFT_MEDIUM, AZIMUTH_LEFT_FINE, ...) only AFTER the
visual decision has been made — the LLM never outputs or even sees a real action
name.

Scope: candidates are the four movement families that have COARSE/MEDIUM/FINE
magnitude variants — AZIMUTH_LEFT, AZIMUTH_RIGHT, ELEVATION_UP, ELEVATION_DOWN —
rendered ONLY at COARSE (their full-magnitude outcome); MEDIUM and FINE are never
directly rendered, only reasoned about as smaller interpolations along the same
visually observed trajectory. Plus two fixed, single-outcome candidates with no
magnitude variants at all — AZIMUTH_RIGHT_180, ELEVATION_UP_180 (see
FIXED_BLIND_ACTIONS) — whose magnitude recommendation is accepted but ignored,
since there is no smaller/larger version of a 180-degree rotation to interpolate
to. STOP and UNDO_LAST remain directly selectable keywords (not blind candidates
— STOP is already visible as the current image, and UNDO_LAST has no rendered
comparison of its own), preserving both exactly as before.

This module is purely additive: visual_rollout_agent.py, prompt_writer.py,
session.py, camera_actions.py, camera_state.py, and action_parser.py are all
unmodified. It reuses:
  - visual_rollout_agent.render_candidate_rollouts (unmodified, candidate rendering)
  - camera_actions.VALID_ACTIONS (unmodified, final validation)
  - chatgpt_client.ask_chatgpt's `reference_items` param (unmodified)
  - CameraReasoningSession.process_chatgpt_response (unmodified action-application
    path — fed a synthetic "Next action:\\n<ACTION>" response once Python has
    privately resolved and validated the real action name)
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
# "minimal" 180-degree rotation, so whatever magnitude the LLM gives for one of
# these is accepted but ignored (mirrors how STOP/UNDO_LAST tolerate any
# magnitude value). Their movement_family IS the final action, verbatim.
FIXED_BLIND_ACTIONS = {"AZIMUTH_RIGHT_180", "ELEVATION_UP_180"}

BLIND_CANDIDATE_ACTION_SUBSET = {f"{family}_COARSE" for family in SCALABLE_MOVEMENT_FAMILIES} | FIXED_BLIND_ACTIONS

SCALE_MAP = {"full": "COARSE", "reduced": "MEDIUM", "minimal": "FINE"}

# Uniform, non-descriptive caption used for every blind candidate — identical
# text for all of them, so it carries no distinguishing information.
BLIND_CANDIDATE_DESCRIPTION = "A full-magnitude candidate render generated from the current camera state."

_CANDIDATE_ID_ALPHABET = string.ascii_uppercase + string.digits


class BlindSelectionError(ValueError):
    """Raised when the LLM's selection can't be resolved to a valid action —
    never silently defaulted to STOP or an arbitrary candidate."""


@dataclass
class BlindCandidateBatch:
    """LLM-visible blind candidates + the private mapping back to real actions.

    visible_items: (candidate_id, neutral_image_path, neutral_description) triples,
        in shuffled order — safe to hand straight to ask_chatgpt's reference_items.
    candidate_mapping: candidate_id -> real movement family (e.g. "AZIMUTH_LEFT").
        NEVER sent to the LLM; only used locally after the response comes back.
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


# ------------------------------------------------------------------
# Neutralized action history (never exposes real action names to the blind prompt)
# ------------------------------------------------------------------

def _summarize_history_neutrally(action_history: Optional[List[Dict]], lookback: int = 5) -> str:
    """Convert real-action-name history into a neutral behavior summary — never
    prints a real action name, so the blind prompt can't be biased by it.
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


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------

BLIND_SELECTION_PROMPT_TEMPLATE = """You are selecting the next camera movement by comparing rendered candidate images
to the current render and the target — never by reasoning about labels,
identifiers, or action-name words.

Your PRIMARY objective is to move the camera viewpoint toward the target side,
orientation, and viewing direction.

Do not confuse improved feature visibility with improved target alignment.

A feature may appear clearer because of reduced overlap, better framing, larger
projected size, or a more favorable local angle even when the camera has moved
toward the wrong side of the object. Therefore, a candidate MUST NOT be judged
as better merely because one structure becomes clearer or more recognizable.

Use this strict decision hierarchy:

1. GLOBAL VIEWPOINT DIRECTION
   Determine whether the current-to-candidate trajectory moves the camera toward
   the target side and viewing orientation.

2. GLOBAL ENDPOINT PROXIMITY
   Determine whether the candidate endpoint is globally closer to the target
   viewpoint than the current render.

3. MAGNITUDE
   Determine whether the target lies near the full candidate endpoint or
   somewhere between the current render and candidate endpoint.

4. LOCAL FEATURE EVIDENCE
   Only after the first three checks, use clearer landmarks, structures, or
   occlusion changes as supporting evidence or as a tie-breaker between
   candidates that already move in the correct global direction.

Local feature clarity can support a candidate only when the visibility change is
consistent with moving toward the target side. Local clarity alone is never
sufficient evidence of target progress.

You are given, in order:

1. The CURRENT rendered image before any movement.
2. The target view: {target_description}
   {target_image_line}{reference_items_line}  3. Several candidate images, each preceded by an opaque identifier such as
   "[CANDIDATE_K7P4Q]".

Candidate IDs are meaningless, randomly generated, regenerated every iteration,
and shown in random order. Do not infer anything from identifier text. It
contains no information about direction, magnitude, or movement type.

Each candidate image shows the full-magnitude outcome of one hidden camera
movement. Use only the visible current-to-candidate transformation to infer the
movement trajectory.

MEDIUM and FINE outcomes are not directly rendered. Treat them only as smaller
interpolations along the same visually observed trajectory. Never claim that a
MEDIUM or FINE result was directly observed.

{neutral_history_block}

GLOBAL VIEWPOINT REASONING

To determine whether a candidate moves toward the target side, compare global
viewpoint evidence such as:

* which broad side or surface of the object becomes more exposed;
* which broad side or surface becomes hidden;
* changes in the balance between opposing visible sides;
* whether a target-facing surface becomes more frontal or more edge-on;
* silhouette changes across the whole object;
* global occlusion and depth-order changes;
* movement of multiple landmarks as a coherent group;
* foreshortening of major object axes;
* relative projected positions of distant landmarks;
* whether the candidate approaches the target's overall orientation.

Prioritize coherent changes across the whole object over isolated changes in one
small feature.

When the target image comes from another source, rendering style, object
instance, lighting condition, or dataset, compare viewpoint geometry rather than
appearance. Ignore differences in texture, color, lighting, rendering quality,
anatomical detail, and exact object shape unless they directly reveal viewpoint.

FEATURE-CLARITY WARNING

Do not select a candidate for reasons such as:

* one landmark is clearer;
* one structure is larger;
* an edge is sharper;
* a region has less overlap;
* more detail is visible;
* the image appears cleaner or easier to interpret.

These observations are relevant only if they indicate that the camera is
exposing the same broad side or orientation visible in the target.

For example:

* A feature becoming clearer while the target-facing side becomes less exposed
  means the candidate moves away.
* A feature becoming larger because of projected scale does not imply viewpoint
  progress.
* A feature becoming less occluded can occur during movement toward either the
  correct or incorrect side.
* A visually informative candidate can still be directionally wrong.
* A less visually clear candidate may still be correct if its global viewpoint
  moves toward the target.

Rules:

* Candidate order is randomized and meaningless.
* Evaluate every candidate exactly once.
* Compare every candidate directly against the CURRENT render.
* Compare every candidate endpoint directly against the target.
* First judge global viewpoint trajectory; only then discuss local clarity.
* Reference-view images are visual anchors only. They show what views look like,
  not which action should be selected.
* Describe visible differences before judging usefulness.
* Use image evidence, not semantic guesses.
* Do not assume screen-left, screen-right, screen-up, or screen-down corresponds
  to a particular camera action.
* Do not infer camera direction from textual labels.
* Do not invent an exact MEDIUM or FINE image.
* If a candidate moves globally away from the target side, reject it even if
  individual structures become clearer.
* Never recommend a smaller magnitude of a globally wrong-direction candidate.
  A smaller wrong-direction movement is still wrong.
* A candidate cannot receive `recommended` unless its global viewpoint
  trajectory is classified as `toward`.
* A candidate classified as `away` must receive `moves_away`.
* A candidate classified as `ambiguous` cannot be selected solely because it
  has the clearest features.
* Framing and projected scale must not override viewpoint alignment.
* Use local feature clarity only as secondary supporting evidence.

For EVERY candidate, use exactly this format:

candidate_id: <opaque candidate ID>

global_viewpoint_change:
<Describe the coherent whole-object viewpoint change from the current render to
this candidate. State which broad surfaces or sides become more exposed or more
hidden, how the silhouette changes, and how major landmarks move together.>

local_feature_change:
<Describe any individual structures that become clearer, less clear, larger,
smaller, more occluded, or less occluded. Do not yet interpret this as target
progress.>

clarity_relevance:
<Explain whether the local feature changes genuinely indicate movement toward
the target side or are merely improvements in visibility, framing, scale, or
recognizability.>

evidence_source:
direct_full_magnitude_render

scaled_expected_result:
<Describe what a reduced or minimal version of the same globally observed
trajectory would likely produce. Explicitly state that this is an interpolation,
not a directly observed result.>

target_side_comparison:
<Compare the candidate's broad visible side and viewing orientation with the
target. Explain whether the candidate exposes more of the target-facing side,
less of it, or changes the view along a different axis.>

target_endpoint_comparison:
<State whether the target appears closer to the current render, between the
current render and candidate endpoint, near the candidate endpoint, or in a
different visual direction.>

target_progress:
<toward | away | ambiguous | no_change>

priority_check:
<Confirm that the target_progress judgment is based primarily on global
viewpoint alignment rather than isolated feature clarity. If a feature becomes
clearer, explain why that does or does not indicate movement toward the target
side.>

history_check:
<new_progress | repeated | reversal | oscillation_risk>

why_this_candidate: <Give image-grounded evidence supporting this candidate. The primary reason must
be global movement toward the target side or orientation. Local clarity may be
mentioned only as secondary supporting evidence.>

why_not_this_candidate:
<Give image-grounded evidence against this candidate, including any possibility
that apparent improvement comes only from clearer local features, framing,
scale, or reduced occlusion rather than target-side progress.>

verdict:
<recommended | could_help | neutral | moves_away>

Verdict constraints:

* `recommended`:
  The candidate clearly moves the global viewpoint toward the target side and is
  better supported than the alternatives.

* `could_help`:
  The candidate probably moves toward the target, but the evidence is incomplete,
  indirect, or weaker than another candidate.

* `neutral`:
  The candidate mainly changes framing, scale, roll, visibility, or an unrelated
  viewing dimension without clear target-side progress.

* `moves_away`:
  The candidate exposes less of the target-facing side, exposes more of the
  opposite side, or otherwise moves in a globally different direction, even if
  some features become clearer.

After evaluating every candidate, choose a magnitude using this interpretation:

full — the candidate endpoint closely matches the target, the complete observed
global viewpoint change is useful, and there is no clear overshoot.

Current ---------------- Candidate ~= Target

reduced — the global viewpoint direction is correct, but the full movement is
too large and the target appears between the current render and candidate
endpoint.

Current -------- Target -------- Candidate

minimal — the current image is already globally close to the target, and only a
small interpolation along the correct candidate trajectory is needed.

Current -- Target ---------------- Candidate

Do not use magnitude to rescue a candidate whose global direction is wrong.

After evaluating every candidate individually, synthesize across all candidates
before deciding.

Use the following comparison order during synthesis:

1. Eliminate candidates that globally move away from the target side.
2. Separate true viewpoint progress from mere improvements in feature clarity.
3. Compare how strongly each remaining candidate exposes the target-facing side.
4. Compare whether each endpoint undershoots, matches, or overshoots the target.
5. Use local landmark clarity only to break ties between candidates with similar
   global viewpoint progress.
6. Select the candidate with the strongest evidence of global target-side
   progress, not the candidate with the clearest individual feature.

Then return exactly this final format:

Synthesis and conclusion:
<Compare every candidate directly against the others. Begin by stating which
candidates move toward, away from, or ambiguously relative to the target side.
Explicitly distinguish global viewpoint progress from isolated feature-clarity
improvements. Explain why the selected candidate is better aligned with the
target side than every alternative.>

Selected candidate:
<CANDIDATE_ID | STOP | UNDO_LAST>

Magnitude recommendation:
<full | reduced | minimal | not_applicable>

Expected visual result:
<Give one concrete and falsifiable prediction about the global viewpoint change,
such as which broad side should become more exposed or hidden. Do not predict
only that a feature will become clearer.>

You must never output a real camera-action name. Output only an opaque candidate
ID, STOP, or UNDO_LAST.

`not_applicable` is valid only for STOP or UNDO_LAST. Every opaque-candidate
selection must use full, reduced, or minimal.

Only select STOP when the current render already sufficiently matches the
target's global side and orientation.

Only select UNDO_LAST when the previous movement caused a clear global reversal
or moved the camera away from the target side.
"""


def _format_target_image_line(target_image_path: Optional[str]) -> str:
    if target_image_path and Path(target_image_path).exists():
        return "  A target reference image is also attached, immediately after the current image.\n"
    return ""


def _format_reference_items_line(reference_items: Optional[List[Tuple[str, str, str]]]) -> str:
    if reference_items:
        return (
            "  Labeled reference-view images are also attached above the candidates below — "
            "these are visual anchors describing what certain views look like, and are NOT "
            "camera-action instructions.\n"
        )
    return ""


# ------------------------------------------------------------------
# Response parsing
# ------------------------------------------------------------------

def _extract_blind_selection(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (selected_candidate, magnitude_recommendation) from a free-form
    blind-evaluation response. Accepts the value on the same line as the label or
    on the line below (mirrors action_parser.extract_action()'s tolerance).
    """
    candidate_match = re.search(
        r"Selected candidate:\**\s*\n?\s*\**\s*([A-Za-z][A-Za-z0-9_]*)", response, re.IGNORECASE
    )
    magnitude_match = re.search(
        r"Magnitude recommendation:\**\s*\n?\s*\**\s*([A-Za-z_]+)", response, re.IGNORECASE
    )
    selected_candidate = candidate_match.group(1).strip().upper() if candidate_match else None
    magnitude_recommendation = magnitude_match.group(1).strip().lower() if magnitude_match else None
    return selected_candidate, magnitude_recommendation


def _construct_final_action(
    selected_candidate: str,
    magnitude_recommendation: Optional[str],
    candidate_mapping: Dict[str, str],
) -> str:
    """Privately resolve the LLM's blind selection into a real, validated action
    name. Raises BlindSelectionError (never silently falls back to STOP, the
    first candidate, or an arbitrary/unvalidated action) if anything is invalid.
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
        # No magnitude variants exist for this one — whatever the LLM said is
        # accepted but irrelevant, same tolerance as STOP/UNDO_LAST.
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
# Step 2 (called every iteration): ask the LLM to select from the blind batch
# ------------------------------------------------------------------

def select_action_from_blind_candidates(
    current_image_path: str,
    batch: BlindCandidateBatch,
    target_description: str,
    target_image_path: Optional[str] = None,
    reference_items: Optional[List[Tuple[str, str, str]]] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Ask the LLM to select one blind candidate + a qualitative magnitude, then
    privately resolve and validate the real action. Raises BlindSelectionError on
    any invalid/unparseable selection — never applies an arbitrary action.

    Returns {"selected_candidate", "magnitude_recommendation", "final_action",
    "raw_response"}. The private candidate_mapping is used here only, never
    included in the returned dict's LLM-facing fields.
    """
    prompt = BLIND_SELECTION_PROMPT_TEMPLATE.format(
        target_description=target_description,
        target_image_line=_format_target_image_line(target_image_path),
        reference_items_line=_format_reference_items_line(reference_items),
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

    selected_candidate, magnitude_recommendation = _extract_blind_selection(response_text)
    if selected_candidate is None:
        print(
            "[blind_visual_rollout_agent] ERROR: could not find a 'Selected candidate:' line in "
            "the response.",
        )
        print(f"[blind_visual_rollout_agent] Currently valid candidate IDs: {sorted(batch.candidate_mapping)}")
        raise BlindSelectionError("response did not contain a parseable 'Selected candidate:' line")

    try:
        final_action = _construct_final_action(selected_candidate, magnitude_recommendation, batch.candidate_mapping)
    except BlindSelectionError:
        print(f"[blind_visual_rollout_agent] ERROR: invalid selected_candidate={selected_candidate!r} "
              f"magnitude_recommendation={magnitude_recommendation!r}")
        print(f"[blind_visual_rollout_agent] Currently valid candidate IDs: {sorted(batch.candidate_mapping)}")
        raise

    return {
        "selected_candidate": selected_candidate,
        "magnitude_recommendation": magnitude_recommendation,
        "final_action": final_action,
        "raw_response": response_text,
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
        "selection": selection,
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
) -> dict:
    """Run the blind visual-rollout camera-action-selection loop against an
    already-`.initialize()`d CameraReasoningSession.

    Every iteration: render the 4 scalable-movement COARSE candidates
    (render_candidate_rollouts, reused unmodified), assign them random opaque IDs
    and shuffle (prepare_blind_candidates), ask the LLM to pick one + a
    qualitative magnitude using only opaque IDs and images
    (select_action_from_blind_candidates), privately resolve + validate the real
    action, then apply it via the existing session.process_chatgpt_response()
    path. Stops early on STOP, after max_iterations, or if the LLM's selection
    can't be validated (raises BlindSelectionError rather than silently applying
    an arbitrary action).
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

        selection = select_action_from_blind_candidates(
            current_image_path,
            batch,
            session.target_description,
            target_image_path=session.target_image_path,
            reference_items=reference_items,
            action_history=session._action_history,
            model=model,
        )
        print("[blind_visual_rollout_agent] LLM reasoning (blind — no real action names visible to it):", flush=True)
        print(selection["raw_response"], flush=True)
        print(f"[blind_visual_rollout_agent] Selected candidate: {selection['selected_candidate']}", flush=True)
        print(f"[blind_visual_rollout_agent] Magnitude recommendation: {selection['magnitude_recommendation']}", flush=True)
        print(f"[blind_visual_rollout_agent] Resolved final action: {selection['final_action']}", flush=True)

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
