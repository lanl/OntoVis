"""Tests for the fixed-window isovalue specialist's two-stage, goal-blind selection
pipeline (see isovalue_adapter.py's module docstring). Covers:

- Pure-arithmetic window computation and the neutral evaluation color ramp.
- The 6 required behaviors from the refactor spec: blind prompt isolation, opaque
  candidate-id mapping, no-unsupported-evidence prompt constraints, abstention, generic
  (non-medical) dataset behavior, and color neutrality.
- Multi-view rendering: each candidate's 6 views are rendered as SEPARATE image files (never
  tiled) and sent together in ONE Stage-1 call, labeled with opaque "view_N" ids.
- End-to-end runs against the real VTK pipeline and a real dataset, with only the LLM call
  (ask_chatgpt) mocked.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from camera_reasoning.session import CameraReasoningSession
from visualization_orchestrator.specialists.isovalue_adapter import (
    BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE,
    CANDIDATE_ID_PREFIX,
    GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE,
    WINDOW_LABEL_PREFIX,
    IsovalueSpecialist,
    _observe_window_blind,
    build_evaluation_ramp_for_window,
    build_opacity_ramp_for_band,
    compute_fixed_windows,
    render_window_previews,
    run_isovalue_band_selection,
    view_action_map,
)
from visualization_orchestrator.state import VisualizationState

DATASET_PATH = str(Path(__file__).resolve().parent.parent / "data" / "vis_male_128x256x256_uint8.raw")
DIMENSIONS = (128, 256, 256)

# Terms that must never appear in the pipeline's own prompt templates -- the pipeline must
# work for arbitrary future volumetric datasets/goals, not just medical scans (spec Test 5).
# ("anatomy" is intentionally excluded: the blind prompt legitimately says "do not infer
# ...anatomy...from color" as an instruction to AVOID anatomical inference, which is the
# opposite of hardcoding a medical assumption.)
DATASET_SPECIFIC_TERMS = ["skull", "bone", "medical", "cranium", "tissue"]


@pytest.fixture
def session(tmp_path):
    s = CameraReasoningSession(
        raw_path=DATASET_PATH,
        dimensions=DIMENSIONS,
        isovalue=40,
        output_dir=str(tmp_path / "session_output"),
    )
    s.initialize()
    return s


def _blind_observation(clear=None, uncertain=None):
    return {
        "per_view_observations": {"view_0": [], "view_1": []},
        "cross_view_summary": {
            "clear_observations": clear or [],
            "uncertain_observations": uncertain or [],
        },
        "noise_and_artifacts": {"isolated_fragments": "low", "surface_noise": "low", "occlusion": "low"},
    }


def _stage2_response(decision, selected_candidate=None, explanation="", candidate_verdicts=None):
    return json.dumps({
        "candidate_verdicts": candidate_verdicts or {},
        "decision": decision,
        "selected_candidate": selected_candidate,
        "supporting_observations": [],
        "contradictory_observations": [],
        "explanation": explanation,
    })


def _passing_verdict(criteria):
    """A step-1 verdict where every listed criterion is independently satisfied -- clears
    the absolute floor in run_isovalue_band_selection for a "selected" decision."""
    return {"verdict": "passes", "criteria_met": list(criteria), "criteria_not_met": []}


def _is_stage1_call(extra_images):
    """Stage 1 calls always carry extra_images (the candidate's views); Stage 2 is
    text-only and never does."""
    return bool(extra_images)


# --- Pure arithmetic / rendering helpers --------------------------------------------------

def test_compute_fixed_windows_covers_full_range_with_no_gaps_or_overlap():
    windows = compute_fixed_windows(num_windows=8, value_range=(0.0, 256.0))

    assert len(windows) == 8
    assert windows[0]["low"] == 0
    assert windows[-1]["high"] == 256
    for prev, nxt in zip(windows[:-1], windows[1:]):
        assert prev["high"] == nxt["low"]
    for w in windows:
        assert w["low"] <= w["peak"] <= w["high"]


def test_compute_fixed_windows_does_not_depend_on_any_dataset():
    assert compute_fixed_windows(num_windows=4) == compute_fixed_windows(num_windows=4)


# --- Test 6: Color Neutrality --------------------------------------------------------------

def test_evaluation_ramp_is_grayscale_with_consistent_shades_across_windows():
    windows = compute_fixed_windows(num_windows=8)
    color_ramps = [build_evaluation_ramp_for_window(w)[1] for w in windows]

    for ramp in color_ramps:
        for _, r, g, b in ramp:
            assert r == g == b  # true gray, no hue -- color can't carry material identity

    # Anchored to each window's own [low, peak] range (like the warm palette is), not a
    # fixed absolute position -- so every window reaches the SAME brightness values, just
    # at different x-positions. A window's material is never rendered near-black just
    # because its own [low, high] happens to sit low on the absolute intensity scale.
    brightest_shades = {ramp[-1][1] for ramp in color_ramps}  # each ramp's final (brightest) gray value
    assert len(brightest_shades) == 1  # every window's brightest shade is identical

    # And it must differ from the final applied result's warm palette (hue-wise), for at
    # least one window (otherwise "neutral" would be a no-op).
    final_color_points = build_opacity_ramp_for_band(windows[0])[1]
    assert color_ramps[0] != final_color_points


def test_evaluation_ramp_does_not_render_low_windows_near_black():
    # Regression test: an earlier version anchored evaluation color to the ABSOLUTE
    # [0, 256) scale instead of each window's own range, so a low-intensity window's
    # material rendered at ~(low/256) brightness -- nearly invisible. The window's own
    # peak must now reach the same bright shade as any other window's peak.
    low_window = compute_fixed_windows(num_windows=8)[0]  # e.g. [0, 32)
    high_window = compute_fixed_windows(num_windows=8)[-1]  # e.g. [224, 256)

    low_peak_shade = next(rgb for x, *rgb in build_evaluation_ramp_for_window(low_window)[1] if x == low_window["peak"])
    high_peak_shade = next(rgb for x, *rgb in build_evaluation_ramp_for_window(high_window)[1] if x == high_window["peak"])

    assert low_peak_shade == high_peak_shade
    assert low_peak_shade[0] > 0.5  # actually bright, not near-black


# --- Multi-view rendering: separate images, opaque ids, sent together in one call ---------

def test_render_window_previews_saves_six_separate_files_not_a_tile(session, tmp_path):
    windows = compute_fixed_windows(num_windows=2)
    rendered = render_window_previews(session, windows, str(tmp_path / "previews"), multi_angle=True)

    assert len(rendered) == 2
    for window, view_images in rendered:
        assert set(view_images.keys()) == {f"view_{i}" for i in range(6)}
        paths = list(view_images.values())
        assert len(paths) == len(set(paths))  # six genuinely distinct files
        for path in paths:
            assert Path(path).is_file()
            assert "grid" not in Path(path).name  # never tiled


def test_render_window_previews_multi_angle_false_renders_one_view(session, tmp_path):
    windows = compute_fixed_windows(num_windows=1)
    rendered = render_window_previews(session, windows, str(tmp_path / "previews"), multi_angle=False)

    _, view_images = rendered[0]
    assert list(view_images.keys()) == ["view_0"]


def test_view_action_map_has_six_entries_and_is_never_exposed_to_the_model():
    mapping = view_action_map(multi_angle=True)
    assert set(mapping.keys()) == {f"view_{i}" for i in range(6)}
    # view_0 is the current/starting orientation (no action applied)
    assert mapping["view_0"] is None


# --- Test 1: Blind Prompt Isolation ---------------------------------------------------------

def test_blind_observation_prompt_contains_no_goal_or_window_identity(tmp_path):
    goal = "show the skull"
    label = f"{WINDOW_LABEL_PREFIX}32_64"
    view_images = {}
    for i in range(6):
        path = tmp_path / f"fake_view_{i}.png"
        path.write_bytes(b"not a real png, just needs to exist for the path check")
        view_images[f"view_{i}"] = str(path)

    seen_calls = []

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        seen_calls.append((prompt, extra_images))
        return json.dumps(_blind_observation())

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        _observe_window_blind(view_images, model=None)

    assert len(seen_calls) == 1
    prompt, extra_images = seen_calls[0]
    assert prompt == BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE  # fully static -- nothing interpolated
    assert goal not in prompt
    assert "skull" not in prompt.lower()
    assert label not in prompt
    assert "32" not in prompt and "64" not in prompt

    # All 6 views went in as extra_images (not screenshot_path), labeled only "view_N".
    assert extra_images == [(f"view_{i}", view_images[f"view_{i}"]) for i in range(6)]


def test_blind_prompt_template_itself_has_no_dataset_specific_vocabulary():
    lowered = BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE.lower()
    for term in DATASET_SPECIFIC_TERMS:
        assert term not in lowered


# --- Test 2: Opaque IDs ----------------------------------------------------------------------

def test_stage1_receives_six_separate_image_paths_per_candidate(session):
    windows = compute_fixed_windows(num_windows=3)
    stage1_extra_images = []

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            stage1_extra_images.append(extra_images)
            return json.dumps(_blind_observation())
        return _stage2_response("selected", f"{CANDIDATE_ID_PREFIX}0", "ok")

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        run_isovalue_band_selection(session, goal="show something", success_criteria=[], num_windows=3)

    assert len(stage1_extra_images) == len(windows)  # one call per candidate
    for extra_images in stage1_extra_images:
        assert len(extra_images) == 6  # six separate images, in ONE call
        labels = [label for label, _ in extra_images]
        assert labels == [f"view_{i}" for i in range(6)]
        paths = [path for _, path in extra_images]
        assert len(paths) == len(set(paths))  # six genuinely distinct files


def test_stage2_sees_opaque_candidate_ids_and_result_maps_back_to_real_window(session):
    windows = compute_fixed_windows()
    target_index = 5
    target_label = f"{WINDOW_LABEL_PREFIX}{windows[target_index]['low']}_{windows[target_index]['high']}"
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}{target_index}"

    seen_stage2_prompt = {}

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        seen_stage2_prompt["prompt"] = prompt
        return _stage2_response("selected", target_candidate_id, "matches")

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show something", success_criteria=[])

    prompt = seen_stage2_prompt["prompt"]
    for i in range(len(windows)):
        assert f"{CANDIDATE_ID_PREFIX}{i}" in prompt
    for w in windows:
        assert f"{WINDOW_LABEL_PREFIX}{w['low']}_{w['high']}" not in prompt  # real labels never sent to Stage 2

    assert result["selected_band_label"] == target_label  # correctly mapped back from the opaque id


# --- Test 3: No Unsupported Evidence (prompt constraint) -------------------------------------

def test_stage2_prompt_forbids_inventing_unsupported_structure():
    lowered = GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE.lower()
    assert "do not introduce structures or properties absent from the observations" in lowered


# --- Test 4: Abstention ------------------------------------------------------------------------

def test_no_match_is_propagated_without_falling_back_to_a_guess(session):
    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response("no_match", None, "nothing matches")

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show something absent", success_criteria=[])

    assert result["converged"] is False
    assert result["selected_band_label"] is None
    assert "nothing matches" in result["reasoning"]
    assert Path(result["final_image_path"]).is_file()  # still renders SOMETHING (current state), never guesses


def test_relative_best_of_a_bad_lot_is_rejected_even_if_model_says_selected(session):
    # Regression for the "WINDOW_0_32" case: Stage 2 can be tempted to pick whichever
    # candidate is LEAST bad relative to the others ("candidate_0 has less fragmentation
    # than 2-7") without that candidate actually satisfying the success criteria on its own
    # merits. The model's top-level "decision": "selected" must not be trusted blindly --
    # if the chosen candidate's own step-1 verdict is "fails", the whole selection is
    # rejected (treated the same as an explicit "no_match"), not silently accepted.
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}0"
    success_criteria = ["bone structure is clearly visible", "minimal noise or fragmentation"]

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response(
            "selected", target_candidate_id,
            "candidate_0 is relatively the best of the available candidates",
            candidate_verdicts={
                target_candidate_id: {
                    "verdict": "fails",
                    "criteria_met": ["bone structure is clearly visible"],
                    "criteria_not_met": ["minimal noise or fragmentation"],
                },
            },
        )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show the bone", success_criteria=success_criteria)

    assert result["converged"] is False
    assert result["selected_band_label"] is None
    assert "did not independently satisfy" in result["reasoning"]


def test_missing_verdict_entry_is_treated_as_failed_floor_not_a_free_pass(session):
    # If the model's response omits candidate_verdicts entirely (or omits the selected
    # candidate's entry) while still claiming "selected", that must not bypass the floor.
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}0"

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response("selected", target_candidate_id, "best match")  # no verdicts

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show bone", success_criteria=["bone is visible"])

    assert result["converged"] is False
    assert result["selected_band_label"] is None


def test_absolute_floor_is_skipped_when_no_success_criteria_given(session):
    # With no success criteria to independently check, there's nothing for the floor to
    # enforce -- a "selected" decision is accepted the same as before this feature existed.
    windows = compute_fixed_windows()
    target_index = 4
    target_label = f"{WINDOW_LABEL_PREFIX}{windows[target_index]['low']}_{windows[target_index]['high']}"
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}{target_index}"

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response("selected", target_candidate_id, "best match")  # no verdicts

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show something", success_criteria=[])

    assert result["converged"] is True
    assert result["selected_band_label"] == target_label


def test_invalid_stage2_response_also_reports_failure_not_a_guess(session):
    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return "not json"

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show something", success_criteria=[])

    assert result["converged"] is False
    assert result["selected_band_label"] is None


# --- Test 5: Generic Dataset Behavior ----------------------------------------------------------

def test_stage2_prompt_template_has_no_dataset_specific_vocabulary():
    lowered = GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE.lower()
    for term in DATASET_SPECIFIC_TERMS:
        assert term not in lowered


def test_pipeline_works_with_a_non_medical_goal_and_observations(session):
    # Synthetic, non-anatomical goal/observations -- branching/geological/particle-style
    # language -- to confirm nothing in the selection path special-cases medical vocabulary.
    windows = compute_fixed_windows()
    target_index = 2
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}{target_index}"

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation(
                clear=[{"description": "a branching, tubular network with several disconnected fragments",
                        "supporting_views": ["view_0"]}],
            ))
        return _stage2_response("selected", target_candidate_id, "shows a continuous branching network")

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(
            session, goal="show the branching flow structure, not isolated particles", success_criteria=[],
        )

    assert result["converged"] is True
    assert result["selected_band_label"] == f"{WINDOW_LABEL_PREFIX}{windows[target_index]['low']}_{windows[target_index]['high']}"


# --- End-to-end (real VTK pipeline, real dataset, mocked LLM) --------------------------------

def test_run_isovalue_band_selection_end_to_end(session):
    windows = compute_fixed_windows()
    target_index = 6
    target_label = f"{WINDOW_LABEL_PREFIX}{windows[target_index]['low']}_{windows[target_index]['high']}"
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}{target_index}"

    blind_call_count = {"n": 0}

    success_criteria = ["bone is visible"]

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            blind_call_count["n"] += 1
            return json.dumps(_blind_observation())
        return _stage2_response(
            "selected", target_candidate_id, "best match",
            candidate_verdicts={target_candidate_id: _passing_verdict(success_criteria)},
        )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(session, goal="show bone", success_criteria=success_criteria)

    assert blind_call_count["n"] == len(windows)  # Stage 1: one call per window
    assert result["converged"] is True
    assert result["selected_band_label"] == target_label
    assert result["final_opacity_points"] == build_opacity_ramp_for_band(windows[target_index])[0]
    assert Path(result["final_image_path"]).is_file()

    # Logging: every candidate's six view paths are recorded, keyed by opaque view id.
    assert set(result["view_images"][target_label].keys()) == {f"view_{i}" for i in range(6)}
    assert result["view_action_map"]["view_0"] is None

    # Session should now be in volume-rendering mode under the selected window's FINAL
    # (warm) ramp -- not the neutral evaluation ramp.
    assert session.use_volume_rendering is True
    assert session.opacity_points == result["final_opacity_points"]
    assert session.color_points == result["final_color_points"]
    assert session.color_points != list(build_evaluation_ramp_for_window(windows[target_index])[1])


def test_run_isovalue_band_selection_respects_num_windows(session):
    blind_call_count = {"n": 0}
    success_criteria = ["bone is visible"]

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            blind_call_count["n"] += 1
            return json.dumps(_blind_observation())
        return _stage2_response(
            "selected", f"{CANDIDATE_ID_PREFIX}0", "ok",
            candidate_verdicts={f"{CANDIDATE_ID_PREFIX}0": _passing_verdict(success_criteria)},
        )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(
            session, goal="show bone", success_criteria=success_criteria, num_windows=4,
        )

    assert len(result["bands"]) == 4
    assert blind_call_count["n"] == 4


# --- IsovalueSpecialist wiring -----------------------------------------------------------------

def test_isovalue_specialist_run_until_complete_success(session):
    windows = compute_fixed_windows()
    target_index = 3
    target_label = f"{WINDOW_LABEL_PREFIX}{windows[target_index]['low']}_{windows[target_index]['high']}"
    target_candidate_id = f"{CANDIDATE_ID_PREFIX}{target_index}"
    success_criteria = ["bone is visible"]

    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response(
            "selected", target_candidate_id, "best match",
            candidate_verdicts={target_candidate_id: _passing_verdict(success_criteria)},
        )

    iterations_seen = []
    specialist = IsovalueSpecialist(session, on_iteration=iterations_seen.append)
    state = VisualizationState(
        dataset_path=DATASET_PATH,
        camera_position=(0.0, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        isovalue=40.0,
        rendered_image_path=session.render_and_save(),
    )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        exec_result = specialist.run_until_complete(
            goal="show bone", state=state, constraints={}, success_criteria=success_criteria,
        )

    assert exec_result.status == "success"
    assert exec_result.goal_satisfied is True
    assert exec_result.iterations_used == len(windows) + 1  # N blind calls + 1 selection call
    assert exec_result.state_patch["transfer_function"] == (session.opacity_points, session.color_points)
    assert exec_result.state_patch["rendered_image_path"] is not None

    assert len(iterations_seen) == 1
    info = iterations_seen[0]
    assert info["agent_id"] == "isovalue_controller"
    assert info["selected_label"] == target_label
    assert len(info["candidates"]) == len(windows)
    assert any(c["selected"] for c in info["candidates"])
    # Each candidate's on_iteration entry records its own six view paths for debugging.
    for candidate in info["candidates"]:
        assert set(candidate["view_images"].keys()) == {f"view_{i}" for i in range(6)}


def test_isovalue_specialist_reports_failure_on_abstention(session):
    def fake_ask(prompt, extra_images=None, model=None, **kwargs):
        if _is_stage1_call(extra_images):
            return json.dumps(_blind_observation())
        return _stage2_response("no_match", None, "no candidate qualifies")

    specialist = IsovalueSpecialist(session)
    state = VisualizationState(
        dataset_path=DATASET_PATH,
        camera_position=(0.0, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        isovalue=40.0,
        rendered_image_path=session.render_and_save(),
    )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        exec_result = specialist.run_until_complete(
            goal="show something absent", state=state, constraints={}, success_criteria=[],
        )

    assert exec_result.status in ("partial", "failed")
    assert exec_result.goal_satisfied is False
