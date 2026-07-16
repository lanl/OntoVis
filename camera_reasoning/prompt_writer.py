from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .camera_actions import ACTION_DESCRIPTIONS, VALID_ACTIONS


ImageItemMetadata = Tuple[str, str]
CandidateImageMetadata = Tuple[str, str, str]  # (candidate_id, action_name, description)
REFERENCE_VIEW_MARKER = "Reference views (stable visual anchors):"
CANDIDATE_RENDER_MARKER = "Candidate action renders (image-grounded counterfactuals):"

ACTION_SCALE_SUFFIXES = ("COARSE", "MEDIUM", "FINE")
ACTION_SCALE_PRIORITY = {"COARSE": 0, "MEDIUM": 1, "FINE": 2}


def _candidate_image_id(index: int) -> str:
    """Return a neutral visual identifier that contains no action-direction words."""
    return f"CANDIDATE_{index:02d}"


def _normalize_candidate_items(
    candidate_action_items: Sequence[Tuple[str, ...]],
) -> List[CandidateImageMetadata]:
    """Normalize old and new candidate metadata formats.

    Backward-compatible two-tuples are interpreted as ``(action_name, description)``
    and assigned neutral IDs by attachment order. Three-tuples are interpreted as
    ``(candidate_id, action_name, description)``.
    """
    normalized: List[CandidateImageMetadata] = []
    for index, item in enumerate(candidate_action_items, 1):
        if len(item) == 3:
            candidate_id, action_name, description = item
        elif len(item) == 2:
            action_name, description = item
            candidate_id = _candidate_image_id(index)
        else:
            raise ValueError(
                "candidate_action_items entries must contain 2 or 3 strings"
            )
        normalized.append(
            (str(candidate_id), str(action_name), str(description))
        )
    return normalized


def _split_action_scale(action_name: str) -> Tuple[str, Optional[str]]:
    """Return ``(movement_family, scale)`` for names such as AZIMUTH_LEFT_FINE."""
    for scale in ACTION_SCALE_SUFFIXES:
        suffix = f"_{scale}"
        if action_name.endswith(suffix):
            return action_name[: -len(suffix)], scale
    return action_name, None


def _action_sort_key(action_name: str) -> Tuple[str, int, str]:
    """Group equal camera movements and put COARSE before MEDIUM before FINE."""
    family, scale = _split_action_scale(action_name)
    return family, ACTION_SCALE_PRIORITY.get(scale, 3), action_name


def _candidate_evidence_for_action(
    action_name: str,
    candidate_action_items: Sequence[CandidateImageMetadata],
) -> Tuple[Optional[str], Optional[str], str]:
    """Find the neutral candidate image that should ground an action evaluation.

    Returns ``(candidate_id, rendered_action, evidence_mode)``. Exact renders are
    preferred. MEDIUM/FINE actions may inherit only a same-family COARSE image as
    qualitative direction evidence.
    """
    by_action = {
        str(rendered_action): str(candidate_id)
        for candidate_id, rendered_action, _ in candidate_action_items
    }
    if action_name in by_action:
        return by_action[action_name], action_name, "direct_exact_scale"

    family, scale = _split_action_scale(action_name)
    coarse_action = f"{family}_COARSE"
    if scale in {"MEDIUM", "FINE"} and coarse_action in by_action:
        return (
            by_action[coarse_action],
            coarse_action,
            "inherited_same_movement_coarse",
        )

    return None, None, "no_visual_candidate"


def _reference_view_instruction_lines(
    reference_view_items: Sequence[ImageItemMetadata],
) -> List[str]:
    """Describe the semantic role of stable reference-view attachments."""
    lines = [
        REFERENCE_VIEW_MARKER,
        "The first attached image-item group contains stable reference views. These images",
        "are visual anchors for recognizing canonical or known viewpoints. They are NOT",
        "results of actions applied from the current camera state.",
        "",
        "Use reference views to:",
        "- identify which known view best resembles the current render;",
        "- identify which known view best represents or approximates the target;",
        "- compare diagnostic structures, silhouette, occlusion, depth, and foreshortening; and",
        "- ground anatomical/spatial view reasoning when textual labels are ambiguous.",
        "",
        "Do NOT describe a reference view as though it were a current-to-candidate camera",
        "transition. Do NOT select an action merely because a reference label sounds similar",
        "to the target. Reference views explain WHAT a view looks like; candidate renders",
        "show WHAT EACH ACTION ACTUALLY DOES from the current state.",
        "",
        "Reference-view inventory, in attachment order within the reference-view group:",
    ]
    for index, (label, description) in enumerate(reference_view_items, 1):
        lines.append(f"  {index}. label: {label}")
        lines.append(f"     metadata_description: {description}")
    lines.append("")
    return lines


def _candidate_render_instruction_lines(
    candidate_action_items: Sequence[CandidateImageMetadata],
) -> List[str]:
    """Explain image-first candidate reasoning inside action evaluation."""
    lines = [
        CANDIDATE_RENDER_MARKER,
        "The second attached image-item group contains one-step counterfactual renders",
        "generated from the exact current camera state. Candidate renders are evidence",
        "inside the matching action-by-action evaluation; do NOT create a separate",
        "candidate-render observation or evaluation section.",
        "",
        "IMAGE-FIRST REQUIREMENT:",
        "- Candidate attachments use neutral IDs such as CANDIDATE_01. The neutral ID is",
        "  only an image locator and contains no camera-direction information.",
        "- First compare the candidate pixels with the current render and write an",
        "  image_only_observation. During this observation, do not use or paraphrase words",
        "  from the mapped action name or metadata description.",
        "- Do not infer that an image moved left, right, up, down, clockwise, or",
        "  counterclockwise merely because those words occur in an action name.",
        "- Only after the image_only_observation is complete may you consult the mapping",
        "  from neutral candidate ID to action name.",
        "- Treat the mapped action name as an identifier, not as visual evidence.",
        "- If the pixels appear inconsistent with the mapped action label, explicitly mark",
        "  label_image_consistency as CONFLICT and trust the rendered pixels.",
        "- A valid justification must cite concrete visible differences. Restating the",
        "  action label or action description is not a justification.",
        "",
        "Candidate renders normally represent COARSE actions. Action names are grouped into",
        "movement families by removing the final _COARSE, _MEDIUM, or _FINE suffix.",
        "",
        "Evidence rules:",
        "- Evaluate a rendered action directly from its candidate image.",
        "- If a MEDIUM or FINE action has no exact candidate image, reuse only the COARSE",
        "  render from the SAME movement family as evidence for qualitative visual change.",
        "- For MEDIUM/FINE, preserve the image-observed direction and predict a smaller",
        "  magnitude. Do not claim that the COARSE image is the exact MEDIUM/FINE result.",
        "- Never transfer evidence between opposite or different movement families.",
        "- If no exact or same-family COARSE render exists, explicitly mark that no visual",
        "  candidate was available and use prediction only.",
        "",
        "Candidate-render inventory, in attachment order within the candidate group:",
    ]
    for candidate_id, _action_name, _description in candidate_action_items:
        lines.append(f"  candidate_image_id: {candidate_id}")
        lines.append("     inspect_by_attachment_order: yes")
        lines.append("     visual_reasoning_policy: pixels_only_before_action_mapping")
        lines.append("     metadata_policy: action words are not visual evidence")
    lines.append("")
    return lines


def _reference_view_output_lines(
    reference_view_items: Sequence[ImageItemMetadata],
) -> List[str]:
    lines = [
        "Reference-view grounding:",
        "<Inspect each stable reference view exactly once. Use it as a visual anchor, not as",
        "an action outcome.>",
        "",
    ]
    for label, _ in reference_view_items:
        lines += [
            f"reference_label: {label}",
            "  diagnostic_visual_features: <what is visibly characteristic of this reference>",
            "  similarity_to_current: <high | medium | low, with direct image evidence>",
            "  similarity_to_target: <high | medium | low, with direct image evidence>",
            "  role_in_reasoning: <how this reference helps identify the current view or target>",
            "",
        ]
    return lines


def append_attached_image_context(
    prompt: str,
    reference_view_items: Optional[Sequence[ImageItemMetadata]] = None,
    candidate_action_items: Optional[Sequence[Tuple[str, ...]]] = None,
) -> str:
    """Append distinct reference-view and candidate-render semantics to a custom prompt."""
    reference_view_items = list(reference_view_items or [])
    candidate_action_items = _normalize_candidate_items(candidate_action_items or [])
    extra: List[str] = []

    if reference_view_items and REFERENCE_VIEW_MARKER not in prompt:
        extra += _reference_view_instruction_lines(reference_view_items)
        extra += [
            "Required reference-grounding response section:",
            "Place this before current-view selection or action evaluation.",
            "",
        ]
        extra += _reference_view_output_lines(reference_view_items)

    if candidate_action_items and CANDIDATE_RENDER_MARKER not in prompt:
        extra += _candidate_render_instruction_lines(candidate_action_items)
        extra += [
            "Required integration rule for the response:",
            "Analyze each candidate only inside the matching action-by-action evaluation.",
            "Use same-family COARSE evidence to evaluate unrendered MEDIUM/FINE actions",
            "with an explicitly reduced movement magnitude.",
            "",
        ]

    if not extra:
        return prompt
    return prompt.rstrip() + "\n\n" + "\n".join(extra).rstrip() + "\n"


# Backward-compatible helper for any existing imports outside CameraReasoningSession.
def append_candidate_render_context(
    prompt: str,
    candidate_action_items: Optional[Sequence[ImageItemMetadata]],
) -> str:
    return append_attached_image_context(
        prompt,
        candidate_action_items=candidate_action_items,
    )


def write_llm_prompt(
    output_dir: str,
    camera_state: dict,
    action_history: List[Dict],
    target_description: str,
    screenshot_path: str,
    target_image_path: Optional[str] = None,
    spatial_context: Optional[str] = None,
    camera_view_hint: Optional[str] = None,
    reference_view_items: Optional[Sequence[ImageItemMetadata]] = None,
    candidate_action_items: Optional[Sequence[Tuple[str, ...]]] = None,
) -> str:
    lines: List[str] = []
    reference_view_items = list(reference_view_items or [])
    candidate_action_items = _normalize_candidate_items(candidate_action_items or [])
    has_reference_views = bool(reference_view_items)
    has_candidate_renders = bool(candidate_action_items)

    lines += [
        "You are ChatGPT, acting as a camera-reasoning assistant for a 3D VTK scene.",
        "",
        "Your job is to compare the current render with the target view and choose exactly",
        "one next camera action.",
        "",
        "Important image-group distinction:",
        "- reference_items are stable reference views used as visual anchors;",
        "- candidate_items are one-step action renders generated from the current camera.",
        "Never interchange these roles.",
        "",
        "Use comparative hypothesis reasoning rather than committing to the first plausible",
        "view interpretation. Evaluate every allowed action before selecting one.",
        "Candidate-action reasoning must be image-first: visual differences are evidence;",
        "action names and descriptions are identifiers only.",
        "",
        "Rules:",
        "- You are NOT allowed to invent raw camera coordinates.",
        "- You are NOT allowed to output arbitrary angles.",
        "- You are NOT allowed to generate VTK code.",
        "- You MUST choose exactly one final action from the allowed action list below.",
        "- View candidates MUST use exact names from canonical_views.",
        "- Every allowed action MUST be evaluated exactly once using its exact action name.",
        "- Do not use confidence as a substitute for evidence.",
        "- Do not use words in an action label as evidence that the rendered image changed",
        "  in that direction. State the pixel-visible change first, then check the label.",
        "- Prefer the smallest action that makes diagnostic progress and does not merely",
        "  repeat or undo recent actions.",
        "",
    ]

    if has_reference_views:
        lines += _reference_view_instruction_lines(reference_view_items)
    if has_candidate_renders:
        lines += _candidate_render_instruction_lines(candidate_action_items)

    if spatial_context:
        lines += [
            "Spatial knowledge (full JSON spec, verbatim — source of truth for view reasoning):",
            "```json",
            spatial_context,
            "```",
            "",
            "Anatomical/spatial view and image orientation are NOT the same thing. Screen",
            "top/bottom/left/right does not by itself prove which side of the object is viewed.",
            "",
        ]
        if camera_view_hint:
            lines += [
                "Camera-vector hint (independent deterministic evidence):",
                camera_view_hint,
                "If this conflicts with the images, explicitly state the conflict.",
                "",
            ]

    stage_number = 1
    lines += ["Before answering, work through these stages:"]
    lines += [
        f"  {stage_number}. VISUAL OBSERVATION: Describe only directly visible evidence in the current render."
    ]
    stage_number += 1
    if has_reference_views:
        lines += [
            f"  {stage_number}. REFERENCE VIEW GROUNDING: Inspect each stable reference view and",
            "     compare it separately with the current render and the target.",
        ]
        stage_number += 1
    lines += [
        f"  {stage_number}. VIEW HYPOTHESES: Generate up to three plausible current-view candidates.",
    ]
    stage_number += 1
    lines += [
        f"  {stage_number}. VIEW SELECTION: Select the best-supported current view and downgrade alternatives.",
    ]
    stage_number += 1
    lines += [
        f"  {stage_number}. TARGET DIFFERENCE: State the true viewpoint difference separately from roll, framing, or zoom.",
    ]
    stage_number += 1
    lines += [
        f"  {stage_number}. INTEGRATED IMAGE-FIRST ACTION EVALUATION: Evaluate every allowed",
        "     action exactly once. For each supplied candidate, write the pixel-based",
        "     observation before consulting its mapped action name. Then use same-family",
        "     COARSE image evidence at reduced magnitude for MEDIUM/FINE.",
    ]
    stage_number += 1
    lines += [
        f"  {stage_number}. ACTION SELECTION: Choose exactly one action only after all image",
        "     groups and actions have been evaluated.",
        "",
    ]

    lines += [
        "Target view:",
        target_description,
        "",
        "Current render screenshot:",
        str(screenshot_path),
    ]
    if target_image_path and Path(target_image_path).exists():
        lines += [f"Target image: {target_image_path}"]
    lines.append("")

    pos = camera_state["position"]
    fp = camera_state["focal_point"]
    up = camera_state["view_up"]
    va = camera_state["view_angle"]
    cr = camera_state["clipping_range"]
    import numpy as np

    dist = float(np.linalg.norm(np.array(pos) - np.array(fp)))
    lines += [
        "Current camera:",
        f"  position:       [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
        f"  focal_point:    [{fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f}]",
        f"  view_up:        [{up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f}]",
        f"  view_angle:     {va:.2f}",
        f"  clipping_range: [{cr[0]:.3f}, {cr[1]:.3f}]",
        f"  distance:       {dist:.3f}",
        "",
    ]

    if action_history:
        lines.append("Action history (detect repetition, reversal, and oscillation):")
        for i, entry in enumerate(action_history, 1):
            lines.append(f"  {i}. {entry['action']}")
        lines.append("")
    else:
        lines += ["Action history:", "  (none yet)", ""]

    lines.append("Allowed actions:")
    for name in sorted(VALID_ACTIONS, key=_action_sort_key):
        lines.append(f"  - {name}: {ACTION_DESCRIPTIONS[name]}")
    lines.append("")

    lines += ["Return exactly this format:", ""]
    lines += [
        "Visual observation:",
        "<Direct evidence from the current screenshot only; do not name a canonical view here.>",
        "",
    ]

    if has_reference_views:
        lines += _reference_view_output_lines(reference_view_items)

    if spatial_context:
        lines += [
            "View candidates:",
            "1. candidate: <exact canonical_views name>",
            "   evidence_for: <specific image, reference-view, spatial, or camera-vector evidence>",
            "   evidence_against: <contradictions or missing expected features>",
            "   assessment: <plausible | less_likely | contradicted>",
            "   confidence: <0.00-1.00>",
            "2. candidate: <exact canonical_views name>",
            "   evidence_for: ...",
            "   evidence_against: ...",
            "   assessment: ...",
            "   confidence: ...",
            "3. candidate: <exact canonical_views name, or NONE>",
            "   evidence_for: ...",
            "   evidence_against: ...",
            "   assessment: ...",
            "   confidence: ...",
            "",
            "Selected current view:",
            "<exact canonical_views name>",
            "",
            "View selection explanation:",
            "<Explain why it best fits the current image and relevant reference anchors.>",
            "",
        ]

    lines += [
        "Current-to-target difference:",
        "<State the viewpoint difference; separate it from image roll, framing, and zoom.>",
        "",
    ]

    lines += [
        "Integrated action-by-action evaluation:",
        "<Evaluate every allowed action exactly once in the order below. There must be no",
        "separate candidate-render evaluation section.>",
        "",
    ]

    for name in sorted(VALID_ACTIONS, key=_action_sort_key):
        family, scale = _split_action_scale(name)
        if has_candidate_renders:
            candidate_id, rendered_action, evidence_mode = _candidate_evidence_for_action(
                name, candidate_action_items
            )
            render_reference = candidate_id or "NOT_PROVIDED"
            rendered_action_reference = rendered_action or "NOT_PROVIDED"

            if evidence_mode == "direct_exact_scale":
                observation_instruction = (
                    "<describe only concrete pixel-visible differences from the current image; "
                    "do not derive the observation from the action name>"
                )
                scale_instruction = (
                    "<exact rendered scale; use the observed result without magnitude extrapolation>"
                )
                result_instruction = (
                    "<the resulting view directly observed in this action's candidate render>"
                )
            elif evidence_mode == "inherited_same_movement_coarse":
                observation_instruction = (
                    "<reuse only the pixel-visible qualitative change observed in the "
                    "same-family COARSE image; do not derive direction from its label>"
                )
                scale_instruction = (
                    f"<reduce the COARSE movement magnitude to {scale}; preserve direction and type>"
                )
                result_instruction = (
                    f"<predict the same qualitative change as {render_reference}, but with "
                    f"{scale} magnitude; do not claim the COARSE image is exact>"
                )
            else:
                observation_instruction = "<NOT_AVAILABLE; no exact or same-family COARSE render>"
                scale_instruction = "<predict from the action definition only>"
                result_instruction = "<specific predicted visible result for this action>"

            lines += [
                f"action: {name}",
                f"  movement_family: {family}",
                f"  action_scale: {scale or 'NOT_SCALED'}",
                f"  candidate_image_id: {render_reference}",
                f"  render_evidence_mode: {evidence_mode}",
                f"  image_only_observation: {observation_instruction}",
                "  forbidden_in_image_only_observation: <do not justify with action-label words>",
                f"  rendered_action_mapping_after_observation: {rendered_action_reference}",
                "  label_image_consistency: <CONSISTENT | CONFLICT | NOT_CHECKABLE, only after image observation>",
                f"  action_definition_after_observation: {ACTION_DESCRIPTIONS[name]}",
                f"  observed_family_change: <infer from image_only_observation, not from the label>",
                f"  scale_adjustment: {scale_instruction}",
                f"  predicted_or_observed_result: {result_instruction}",
                "  closest_reference_result: <reference label or NOT_AVAILABLE>",
                "  relation_to_target: <how this exact action scale changes the target difference>",
                "  target_progress: <toward | away | mixed_or_ambiguous | no_meaningful_change>",
                "  history_check: <new progress | repeated | reversal | oscillation risk | not applicable>",
                "  verdict: <recommended | could_help | neutral_or_irrelevant | redundant_or_reversal | moves_away>",
                "  justification: <cite specific visible differences; action-label wording alone is invalid evidence>",
                "",
            ]
        else:
            lines += [
                f"action: {name}",
                f"  movement_family: {family}",
                f"  action_scale: {scale or 'NOT_SCALED'}",
                f"  action_definition: {ACTION_DESCRIPTIONS[name]}",
                "  effect: <camera transformation>",
                "  relation_to_target: <how it changes the target difference>",
                "  predicted_visual_change: <specific visible result>",
                "  history_check: <new progress | repeated | reversal | oscillation risk | not applicable>",
                "  verdict: <recommended | could_help | neutral_or_irrelevant | redundant_or_reversal | moves_away>",
                "  justification: <specific evidence-based reason>",
                "",
            ]

    lines += [
        "Action selection explanation:",
        "<Explain why the selected movement family and scale are best. Begin from the",
        "candidate pixels and their target similarity, not from directional words in labels.",
        "Use the COARSE render",
        "to establish direction, but prefer MEDIUM/FINE when the target requires the same",
        "movement with a smaller correction. Compare against rejected families and scales.>",
        "",
        "Next action:",
        "ACTION_NAME",
        "",
        "Expected visual change:",
        "<One concrete, falsifiable result at the selected scale. For MEDIUM/FINE, state a",
        "smaller version of the same-family COARSE change rather than copying the full COARSE result.>",
    ]

    prompt = "\n".join(lines)
    out_path = Path(output_dir) / "llm_prompt.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt)
    return prompt