"""Tests for visualization_orchestrator.registry.AgentRegistry.

No pytest required -- plain asserts, run directly:
  .venv/bin/python tests/test_orchestrator_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import FakeSpecialist, make_camera_spec, make_isovalue_spec, make_state
from visualization_orchestrator.registry import AgentRegistry


def test_exact_capability_match():
    registry = AgentRegistry()
    spec = make_camera_spec()
    registry.register(FakeSpecialist(spec.agent_id, []), spec)

    state = make_state()
    selected = registry.select_agent("show_object_from_direction", state)
    assert selected is not None and selected.agent_id == "camera_controller"

    assert registry.select_agent("reveal_internal_structure", state) is None
    print("test_exact_capability_match: OK")


def test_multiple_candidate_agents_selected_by_priority():
    registry = AgentRegistry()
    low = make_camera_spec(agent_id="camera_low", priority=0)
    high = make_camera_spec(agent_id="camera_high", priority=5)
    registry.register(FakeSpecialist(low.agent_id, []), low)
    registry.register(FakeSpecialist(high.agent_id, []), high)

    state = make_state()
    candidates = registry.find_candidates("adjust_viewpoint", state)
    assert {c.agent_id for c in candidates} == {"camera_low", "camera_high"}

    selected = registry.select_agent("adjust_viewpoint", state)
    assert selected.agent_id == "camera_high"
    print("test_multiple_candidate_agents_selected_by_priority: OK")


def test_missing_required_state_excludes_agent():
    registry = AgentRegistry()
    spec = make_camera_spec()
    registry.register(FakeSpecialist(spec.agent_id, []), spec)

    state = make_state(rendered_image_path=None)  # required_state_fields includes rendered_image_path
    assert registry.select_agent("show_object_from_direction", state) is None
    assert registry.find_candidates("show_object_from_direction", state) == []
    print("test_missing_required_state_excludes_agent: OK")


def test_failed_agent_excluded_via_excluded_agent_ids():
    registry = AgentRegistry()
    low = make_camera_spec(agent_id="camera_a", priority=5)
    other = make_camera_spec(agent_id="camera_b", priority=0)
    registry.register(FakeSpecialist(low.agent_id, []), low)
    registry.register(FakeSpecialist(other.agent_id, []), other)

    state = make_state()
    selected = registry.select_agent("adjust_viewpoint", state, excluded_agent_ids={"camera_a"})
    assert selected.agent_id == "camera_b"
    print("test_failed_agent_excluded_via_excluded_agent_ids: OK")


def test_preferred_agent_id_overrides_priority():
    registry = AgentRegistry()
    low = make_camera_spec(agent_id="camera_low", priority=0)
    high = make_camera_spec(agent_id="camera_high", priority=5)
    registry.register(FakeSpecialist(low.agent_id, []), low)
    registry.register(FakeSpecialist(high.agent_id, []), high)

    state = make_state()
    selected = registry.select_agent("adjust_viewpoint", state, preferred_agent_id="camera_low")
    assert selected.agent_id == "camera_low"
    print("test_preferred_agent_id_overrides_priority: OK")


def test_capability_catalog_covers_all_registered_agents():
    registry = AgentRegistry()
    camera_spec = make_camera_spec()
    isovalue_spec = make_isovalue_spec()
    registry.register(FakeSpecialist(camera_spec.agent_id, []), camera_spec)
    registry.register(FakeSpecialist(isovalue_spec.agent_id, []), isovalue_spec)

    catalog = registry.capability_catalog()
    assert "reveal_internal_structure" in catalog
    assert "show_object_from_direction" in catalog
    assert registry.known_capabilities() == {
        "show_object_from_direction", "adjust_viewpoint",
        "reveal_internal_structure", "reduce_surface_noise",
    }
    print("test_capability_catalog_covers_all_registered_agents: OK")


if __name__ == "__main__":
    test_exact_capability_match()
    test_multiple_candidate_agents_selected_by_priority()
    test_missing_required_state_excludes_agent()
    test_failed_agent_excluded_via_excluded_agent_ids()
    test_preferred_agent_id_overrides_priority()
    test_capability_catalog_covers_all_registered_agents()
    print("\nAll tests passed.")
