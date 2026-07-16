"""
Visual-rollout camera-action selection: instead of picking a camera action from
action *names* + a single current screenshot (the existing text/single-image
reasoning in prompt_writer.py/session.py), this module actually RENDERS what every
candidate action would produce, then asks the LLM to pick based on comparing those
real rendered outcomes against the target — not keywords in the action names.

Why: action names like "AZIMUTH_LEFT_COARSE" don't reliably tell a vision model what
the resulting frame will look like, and mapping target-description words ("top",
"left") straight onto action names is unreliable because target directions can be
anatomical/object-relative while camera actions are always camera-relative.
Rendering the actual candidate frames removes that guesswork.

This module is purely additive, mirroring frozen_reference_alignment.py's approach:
  - camera_actions.VALID_ACTIONS/apply_action/ACTION_DESCRIPTIONS (existing schema, unmodified)
  - camera_state.get_camera_state/set_camera_state/camera_distance (existing, unmodified)
  - volume_scene.save_screenshot (existing, unmodified)
  - chatgpt_client.ask_chatgpt, using its `reference_items` param to send every
    candidate image immediately followed by its own action description
  - CameraReasoningSession.process_chatgpt_response (existing action-application
    path — fed a synthetic "Next action:\\n<ACTION>" response, so screenshot/
    camera-state/history bookkeeping all go through the exact same code as the
    normal loop)

session.py, prompt_writer.py, camera_actions.py, camera_state.py are all untouched.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .camera_actions import ACTION_DESCRIPTIONS, VALID_ACTIONS, apply_action
from .camera_state import camera_distance, get_camera_state, set_camera_state
from .chatgpt_client import ask_chatgpt
from .view_description_generator import _extract_json_object  # reuse existing JSON-extraction logic
from .volume_scene import save_screenshot

MIN_CAMERA_DISTANCE = 1e-3  # mirrors session.py's guard against dollying past the focal point

ROLLOUT_SELECTION_PROMPT_TEMPLATE = """You are selecting the next camera action for a 3D rendered scene by comparing
ACTUAL RENDERED OUTCOMES, not by reasoning about action names.

Use comparative hypothesis reasoning rather than committing to the first plausible
candidate: evaluate every candidate action's actual rendered outcome, weigh
evidence for and against each one, then select the single best next action. Do NOT
shortlist — every candidate listed below must be evaluated exactly once.

You are given, in order:
  1. The CURRENT rendered image (before any action).
  2. The target view: {target_description}
{target_image_line}
  3. One rendered candidate image for every available camera action. Each is
     preceded by "[ACTION_NAME]", followed by the image, followed by a short
     mechanical description of what that action does to the camera. Every
     candidate image shows what the scene would ACTUALLY look like if that
     action were applied right now, from the CURRENT camera state — none of
     them are cumulative with each other.

Candidate actions available this iteration: {candidate_action_list}

{action_history_block}

Rules:
- Do not map target-description words (e.g. "top", "left") directly onto action
  names; camera actions are camera-relative, target directions may be
  anatomical/object-relative. Judge purely from how close each rendered candidate
  looks to the target.
- Every candidate action listed above MUST be evaluated exactly once, in the same
  order shown. Do not shortlist, group, skip, or replace candidates.
- The STOP candidate is identical to the current image (no camera change). Select
  it whenever the current image is already the closest match to the target — do
  not move the camera further just to make progress.
- Base every judgment on the rendered pixels of each candidate image, not on the
  action's name or description text.
- Do not use confidence as a substitute for evidence.

Return only valid JSON with this schema:
{{
  "current_visual_observation": "<describe only what is directly visible in the CURRENT image>",
  "target_difference": "<state the visual/spatial change still required to reach the target>",
  "candidate_evaluation": [
    {{
      "action": "...",
      "observed_visual_change": "<what actually changed in this candidate's rendered image, compared to the current image>",
      "relation_to_target": "<how this candidate's outcome changes the current-target difference>",
      "history_check": "new_progress | repeated | reversal | oscillation_risk | not_applicable",
      "verdict": "recommended | could_help | neutral_or_irrelevant | redundant_or_reversal | moves_away",
      "justification": "<specific evidence-based reason for the verdict, grounded in the rendered pixels>"
    }}
  ],
  "selected_action": "...",
  "reasoning": "why the selected candidate's rendered outcome is closer to the target than every other candidate, referencing the verdicts above"
}}

Rules:
- selected_action MUST be exactly one of the candidate action names listed above.
- candidate_evaluation MUST include every candidate action listed above, in the same order, not a subset.
- Return JSON only, no extra text.
"""


def _safe_stop_selection(reason: str, raw_response: Optional[str] = None) -> dict:
    result = {
        "current_visual_observation": "",
        "target_difference": "",
        "candidate_evaluation": [],
        "selected_action": "STOP",
        "reasoning": f"Falling back to STOP: {reason}.",
        "error": reason,
    }
    if raw_response is not None:
        result["raw_response"] = raw_response
    return result


# ------------------------------------------------------------------
# Step 1 (called every iteration): render one candidate image per action
# ------------------------------------------------------------------

def render_candidate_rollouts(
    session,
    current_image_path: str,
    action_subset: Optional[Set[str]] = None,
) -> List[dict]:
    """Render a one-step visual rollout of every candidate camera action.

    For every action (all of VALID_ACTIONS by default, or `action_subset` if
    given): restore the camera to its EXACT current (pre-rollout) state, apply
    exactly that one action, render, save the image, then restore the camera back
    to the pre-rollout state before moving to the next candidate — so rollouts
    never accumulate on top of each other and never leave the active camera moved.

    STOP's candidate is the current image (no camera change, no re-render).
    UNDO_LAST's candidate is the state one undo would restore to (peeked from
    session's internal undo stack, never popped) — omitted entirely if there is
    no history to undo. Candidates that would violate the MIN_CAMERA_DISTANCE
    guard (the same one session._apply_and_advance() enforces on real application)
    are silently skipped, since selecting them would be reverted anyway.

    Returns a list of {"action": str, "image_path": str}, always including STOP.
    """
    renderer = session._renderer
    render_window = session._render_window
    camera = renderer.GetActiveCamera()
    original_state = get_camera_state(camera)

    undo_target_state = session._camera_state_stack[-1] if session._camera_state_stack else None

    candidates_dir = Path(session.output_dir) / "screenshots" / "rollouts" / f"step_{session._step:03d}"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    actions = sorted(action_subset) if action_subset is not None else sorted(VALID_ACTIONS)

    rollouts = []
    for action in actions:
        if action == "STOP":
            rollouts.append({"action": "STOP", "image_path": current_image_path})
            continue

        if action == "UNDO_LAST":
            if undo_target_state is None:
                continue
            set_camera_state(camera, undo_target_state)
            renderer.ResetCameraClippingRange()
        else:
            set_camera_state(camera, original_state)
            apply_action(action, camera, renderer)

        if camera_distance(get_camera_state(camera)) < MIN_CAMERA_DISTANCE:
            set_camera_state(camera, original_state)
            renderer.ResetCameraClippingRange()
            continue

        render_window.Render()
        image_path = str(candidates_dir / f"{action}.png")
        save_screenshot(render_window, image_path)
        rollouts.append({"action": action, "image_path": image_path})

        # Always restore before the next candidate — rollouts must never leave
        # the active camera state changed.
        set_camera_state(camera, original_state)
        renderer.ResetCameraClippingRange()

    return rollouts


# ------------------------------------------------------------------
# Step 2 (called every iteration): ask the LLM to select from the rendered rollouts
# ------------------------------------------------------------------

def _format_action_history_block(action_history: Optional[List[Dict]]) -> str:
    if action_history:
        lines = "\n".join(f"  {i}. {entry['action']}" for i, entry in enumerate(action_history, 1))
        return f"Action history (use this to detect repetition, reversal, and oscillation):\n{lines}"
    return "Action history:\n  (none yet)"


def select_action_from_rollouts(
    current_image_path: str,
    rollouts: List[dict],
    target_description: str,
    target_image_path: Optional[str] = None,
    action_history: Optional[List[Dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """Ask the LLM to select one action by comparing the actual rendered
    candidates (rollouts) against the target — never falls back to reasoning
    about action names alone. Every candidate's image is sent immediately
    followed by its own mechanical description via ask_chatgpt's reference_items.

    The response schema mirrors prompt_writer.py's exhaustive per-action textual
    evaluation (effect/relation_to_target/history_check/verdict/justification),
    but grounded in the actual rendered candidate images rather than imagined ones.
    """
    candidate_actions = [r["action"] for r in rollouts]
    reference_items = [
        (r["action"], r["image_path"], ACTION_DESCRIPTIONS.get(r["action"], "")) for r in rollouts
    ]

    target_image_line = (
        "  A target reference image is also attached, immediately after the current image.\n"
        if target_image_path and Path(target_image_path).exists()
        else ""
    )
    prompt = ROLLOUT_SELECTION_PROMPT_TEMPLATE.format(
        target_description=target_description,
        target_image_line=target_image_line,
        candidate_action_list=", ".join(candidate_actions),
        action_history_block=_format_action_history_block(action_history),
    )

    response_text = ask_chatgpt(
        prompt=prompt,
        screenshot_path=current_image_path,
        target_image_path=target_image_path,
        reference_items=reference_items,
        model=model,
    )

    parsed = _extract_json_object(response_text)
    if parsed is None:
        print("[visual_rollout_agent] WARNING: selection response was not valid JSON; falling back to STOP.")
        return _safe_stop_selection("model output was not valid JSON", raw_response=response_text)

    selected = parsed.get("selected_action")
    if selected not in candidate_actions:
        print(
            f"[visual_rollout_agent] WARNING: model selected {selected!r}, which is not "
            f"among this iteration's candidates {candidate_actions}; falling back to STOP."
        )
        parsed["error"] = f"selected_action {selected!r} not among candidates"
        parsed["selected_action"] = "STOP"

    covered = {c.get("action") for c in parsed.get("candidate_evaluation", [])}
    missing = set(candidate_actions) - covered
    if missing:
        print(
            f"[visual_rollout_agent] WARNING: candidate_evaluation did not cover every "
            f"candidate action (missing: {sorted(missing)}) — selection still applied, this is diagnostic only."
        )

    return parsed


# ------------------------------------------------------------------
# Trace saving (debugging)
# ------------------------------------------------------------------

def save_rollout_trace(
    trace_dir: str,
    iteration: int,
    current_image_path: str,
    rollouts: List[dict],
    selection: dict,
    applied_action: Optional[str],
) -> str:
    selected_candidate_image = next(
        (r["image_path"] for r in rollouts if r["action"] == selection.get("selected_action")), None
    )
    trace = {
        "iteration": iteration,
        "current_image_path": current_image_path,
        "candidates": rollouts,
        "selection": selection,
        "selected_candidate_image": selected_candidate_image,
        "applied_action": applied_action,
    }
    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"rollout_{iteration:03d}.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)
    return str(trace_path)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def run_visual_rollout_alignment_loop(
    session,
    max_iterations: int = 10,
    trace_dir: Optional[str] = None,
    model: Optional[str] = None,
    dry_run: bool = False,
    action_subset: Optional[Set[str]] = None,
    on_iteration_end: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run the visual-rollout camera-action-selection loop against an already-
    `.initialize()`d CameraReasoningSession.

    Every iteration: render one candidate image per available action from the
    CURRENT camera state (render_candidate_rollouts — never accumulates, never
    leaves the active camera moved), ask the LLM to select one by comparing the
    actual rendered outcomes against the target (select_action_from_rollouts),
    then apply ONLY the selected action via the existing
    session.process_chatgpt_response() path (so screenshot/camera-state/history
    bookkeeping reuse the exact same code as the normal loop). Stops early on STOP
    or after max_iterations.

    `action_subset` restricts which actions get a rendered candidate each
    iteration (default: every action in VALID_ACTIONS, i.e. every candidate is
    always evaluated, never a preselected subset).

    `on_iteration_end`, if given, is called once per iteration with that
    iteration's trace_path, right after it's saved — e.g. pass a function that
    displays the candidate images so they show up live as each iteration runs,
    instead of only after the whole loop finishes. This module has no display/
    plotting dependency itself; the caller (e.g. a notebook cell) supplies it.
    """
    if trace_dir is None:
        trace_dir = str(Path(session.output_dir) / "rollout_traces")

    applied_actions = []
    for i in range(max_iterations):
        current_image_path = session.render_and_save()
        print(f"\n[visual_rollout_agent] Iteration {i}", flush=True)

        rollouts = render_candidate_rollouts(session, current_image_path, action_subset=action_subset)
        print(
            f"[visual_rollout_agent] Rendered {len(rollouts)} candidate rollouts: "
            f"{[r['action'] for r in rollouts]}",
            flush=True,
        )

        selection = select_action_from_rollouts(
            current_image_path,
            rollouts,
            session.target_description,
            target_image_path=session.target_image_path,
            action_history=session._action_history,
            model=model,
        )
        print(f"[visual_rollout_agent] Current visual observation: {selection.get('current_visual_observation')}", flush=True)
        print(f"[visual_rollout_agent] Target difference: {selection.get('target_difference')}", flush=True)
        print("[visual_rollout_agent] Candidate-by-candidate evaluation:", flush=True)
        for c in selection.get("candidate_evaluation", []):
            print(
                f"    {c.get('action'):<22s} verdict={c.get('verdict')!s:<24s} "
                f"history_check={c.get('history_check')}",
                flush=True,
            )
            print(f"        observed_visual_change: {c.get('observed_visual_change')}", flush=True)
            print(f"        relation_to_target:     {c.get('relation_to_target')}", flush=True)
            print(f"        justification:          {c.get('justification')}", flush=True)

        selected_action = selection.get("selected_action", "STOP")
        print(f"[visual_rollout_agent] Selected action: {selected_action}", flush=True)
        print(f"[visual_rollout_agent] Reasoning: {selection.get('reasoning')}", flush=True)

        applied_action = None
        if dry_run:
            print("[visual_rollout_agent] Dry run — action NOT applied.", flush=True)
        else:
            applied_action = session.process_chatgpt_response(f"Next action:\n{selected_action}")
            applied_actions.append(applied_action)

        if trace_dir:
            trace_path = save_rollout_trace(
                trace_dir, i, current_image_path, rollouts, selection, applied_action
            )
            print(f"[visual_rollout_agent] Saved trace: {trace_path}", flush=True)
            if on_iteration_end:
                on_iteration_end(trace_path)

        if not dry_run and applied_action == "STOP":
            break

    return {"mode": "visual_rollout_alignment", "applied_actions": applied_actions, "dry_run": dry_run}
