"""End-to-end tests for the histogram-band isovalue specialist against a real VTK session
and the real vis_male dataset -- only the LLM call (ask_chatgpt) is mocked. Verifies the
whole pipeline (histogram segmentation -> per-band preview renders -> band selection ->
deterministic opacity ramp -> applied transfer function) actually works end to end, not
just its individual pieces in isolation.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from camera_reasoning.session import CameraReasoningSession
from visualization_orchestrator.specialists.isovalue_adapter import (
    BAND_LABEL_PREFIX,
    IsovalueSpecialist,
    build_opacity_ramp_for_band,
    compute_histogram_bands,
    run_isovalue_band_selection,
)
from visualization_orchestrator.state import VisualizationState

DATASET_PATH = str(Path(__file__).resolve().parent.parent / "data" / "vis_male_128x256x256_uint8.raw")
DIMENSIONS = (128, 256, 256)

# foot's histogram is smoothly unimodal past its background spike -- no interior valley
# separates bone from soft tissue, so it only ever produces 2 top-level bands (unlike
# vis_male's clean 3). Used specifically to exercise the refinement round, since a
# refinement is meaningless on a dataset that already separates cleanly.
FOOT_DATASET_PATH = str(Path(__file__).resolve().parent.parent / "data" / "foot_256x256x256_uint8.raw")
FOOT_DIMENSIONS = (256, 256, 256)


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


@pytest.fixture
def foot_session(tmp_path):
    s = CameraReasoningSession(
        raw_path=FOOT_DATASET_PATH,
        dimensions=FOOT_DIMENSIONS,
        isovalue=40,
        output_dir=str(tmp_path / "foot_session_output"),
    )
    s.initialize()
    return s


def test_compute_histogram_bands_on_vis_male_dataset():
    bands = compute_histogram_bands(DATASET_PATH)

    assert len(bands) == 3
    total_fraction = sum(b["voxel_fraction"] for b in bands)
    assert total_fraction == pytest.approx(1.0, abs=1e-9)
    for low, high in zip(bands[:-1], bands[1:]):
        assert low["high"] == high["low"]
    for band in bands:
        assert band["low"] <= band["peak"] < band["high"]


def test_run_isovalue_band_selection_end_to_end(session):
    bands = compute_histogram_bands(DATASET_PATH)
    target_band = bands[-1]  # highest-intensity band -- bone, in this dataset
    target_label = f"{BAND_LABEL_PREFIX}{target_band['low']}_{target_band['high']}"

    fake_response = json.dumps({
        "band_evaluation": [
            {"band": f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}", "observation": "x", "satisfies_goal": b is target_band}
            for b in bands
        ],
        "selected_band": target_label,
        "goal_satisfied": True,
        "reasoning": "bone matches the goal",
    })

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", return_value=fake_response) as mock_ask:
        result = run_isovalue_band_selection(
            session, goal="show bone", success_criteria=["bone is visible"],
        )

    assert mock_ask.call_count == 1
    assert result["converged"] is True
    assert result["selected_band_label"] == target_label
    assert result["final_opacity_points"] == build_opacity_ramp_for_band(target_band)[0]
    assert Path(result["final_image_path"]).is_file()

    # Session should now be in volume-rendering mode under the selected band's ramp.
    assert session.use_volume_rendering is True
    assert session.opacity_points == result["final_opacity_points"]
    assert session.color_points == result["final_color_points"]


def test_run_isovalue_band_selection_invalid_llm_response_reports_failure(session):
    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", return_value="not json"):
        result = run_isovalue_band_selection(
            session, goal="show bone", success_criteria=["bone is visible"],
        )

    assert result["converged"] is False
    assert result["selected_band_label"] is None
    assert Path(result["final_image_path"]).is_file()


def test_isovalue_specialist_run_until_complete_success(session):
    bands = compute_histogram_bands(DATASET_PATH)
    target_band = bands[-1]
    target_label = f"{BAND_LABEL_PREFIX}{target_band['low']}_{target_band['high']}"

    fake_response = json.dumps({
        "band_evaluation": [
            {"band": f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}", "observation": "x", "satisfies_goal": b is target_band}
            for b in bands
        ],
        "selected_band": target_label,
        "goal_satisfied": True,
        "reasoning": "bone matches the goal",
    })

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

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", return_value=fake_response):
        exec_result = specialist.run_until_complete(
            goal="show bone", state=state, constraints={}, success_criteria=["bone is visible"],
        )

    assert exec_result.status == "success"
    assert exec_result.goal_satisfied is True
    assert exec_result.state_patch["transfer_function"] == (session.opacity_points, session.color_points)
    assert exec_result.state_patch["rendered_image_path"] is not None

    assert len(iterations_seen) == 1
    info = iterations_seen[0]
    assert info["agent_id"] == "isovalue_controller"
    assert info["selected_label"] == target_label
    assert len(info["candidates"]) == len(bands)
    assert any(c["selected"] for c in info["candidates"])


def _fake_band_selection_response(band_labels, selected_label, refine_further):
    return json.dumps({
        "band_evaluation": [
            {"band": label, "observation": "x", "satisfies_goal": label == selected_label}
            for label in band_labels
        ],
        "selected_band": selected_label,
        "goal_satisfied": True,
        "refine_further": refine_further,
        "reasoning": f"picked {selected_label}",
    })


def test_run_isovalue_band_selection_refines_when_llm_requests_it(foot_session):
    bands = compute_histogram_bands(FOOT_DATASET_PATH)
    assert len(bands) == 2  # background + one broad, undifferentiated band -- see module docstring
    top_labels = [f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}" for b in bands]
    broad_label = top_labels[-1]

    call_count = {"n": 0}

    def fake_ask(prompt, reference_items=None, model=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _fake_band_selection_response(top_labels, broad_label, refine_further=True)
        # Round 2: whatever sub-band labels appear in this call's prompt -- pick the last one.
        import re
        sub_labels = list(dict.fromkeys(re.findall(r"BAND_\d+_\d+", prompt)))
        return _fake_band_selection_response(sub_labels, sub_labels[-1], refine_further=False)

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        result = run_isovalue_band_selection(
            foot_session, goal="show the bone", success_criteria=["bone visible"],
        )

    assert call_count["n"] == 2
    assert result["refined"] is True
    assert result["initial_selected_band_label"] == broad_label
    assert result["selected_band_label"] != broad_label  # a narrower sub-range won instead
    assert result["refinement_bands"] is not None
    assert len(result["refinement_bands"]) > 1
    assert "Refined further" in result["reasoning"]
    assert Path(result["final_image_path"]).is_file()

    # The applied ramp must come from the SUB-band, not the original broad band.
    label_to_subband = {
        f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}": b for b in result["refinement_bands"]
    }
    winning_subband = label_to_subband[result["selected_band_label"]]
    assert result["final_opacity_points"] == build_opacity_ramp_for_band(winning_subband)[0]


def test_run_isovalue_band_selection_refinement_disabled(foot_session):
    bands = compute_histogram_bands(FOOT_DATASET_PATH)
    top_labels = [f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}" for b in bands]
    broad_label = top_labels[-1]

    fake_response = _fake_band_selection_response(top_labels, broad_label, refine_further=True)

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", return_value=fake_response) as mock_ask:
        result = run_isovalue_band_selection(
            foot_session, goal="show the bone", success_criteria=["bone visible"],
            allow_refinement=False,
        )

    assert mock_ask.call_count == 1
    assert result["refined"] is False
    assert result["selected_band_label"] == broad_label


def test_isovalue_specialist_reports_two_iterations_when_refined(foot_session):
    bands = compute_histogram_bands(FOOT_DATASET_PATH)
    top_labels = [f"{BAND_LABEL_PREFIX}{b['low']}_{b['high']}" for b in bands]
    broad_label = top_labels[-1]

    def fake_ask(prompt, reference_items=None, model=None, **kwargs):
        if "Bands available this call: " + ", ".join(top_labels) in prompt:
            return _fake_band_selection_response(top_labels, broad_label, refine_further=True)
        import re
        sub_labels = list(dict.fromkeys(re.findall(r"BAND_\d+_\d+", prompt)))
        return _fake_band_selection_response(sub_labels, sub_labels[0], refine_further=False)

    iterations_seen = []
    specialist = IsovalueSpecialist(foot_session, on_iteration=iterations_seen.append)
    state = VisualizationState(
        dataset_path=FOOT_DATASET_PATH,
        camera_position=(0.0, 0.0, 1.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
        isovalue=40.0,
        rendered_image_path=foot_session.render_and_save(),
    )

    with patch("visualization_orchestrator.specialists.isovalue_adapter.ask_chatgpt", side_effect=fake_ask):
        exec_result = specialist.run_until_complete(
            goal="show the bone", state=state, constraints={}, success_criteria=["bone visible"],
        )

    assert exec_result.status == "success"
    assert exec_result.iterations_used == 2
    assert "refined" in exec_result.reason

    assert len(iterations_seen) == 2
    assert iterations_seen[0]["iteration"] == 0
    assert iterations_seen[0]["selected_label"] == broad_label
    assert iterations_seen[1]["iteration"] == 1
    assert iterations_seen[1]["selected_label"] is not None
    assert iterations_seen[1]["selected_label"] != broad_label
