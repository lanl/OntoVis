"""Tests for visualization_orchestrator.state (patch validation, ownership, cloning).

No pytest required -- plain asserts, run directly:
  .venv/bin/python tests/test_orchestrator_state.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import make_camera_spec, make_isovalue_spec, make_state
from visualization_orchestrator.state import PatchValidationError, apply_patch, clone_state, validate_patch


def test_camera_agent_cannot_modify_isovalue():
    camera_spec = make_camera_spec()
    try:
        validate_patch({"isovalue": 99.0}, camera_spec.owned_state_fields)
        assert False, "expected PatchValidationError"
    except PatchValidationError as exc:
        assert "isovalue" in str(exc)
    print("test_camera_agent_cannot_modify_isovalue: OK")


def test_isovalue_agent_cannot_modify_camera_position():
    isovalue_spec = make_isovalue_spec()
    try:
        validate_patch({"camera_position": (1.0, 2.0, 3.0)}, isovalue_spec.owned_state_fields)
        assert False, "expected PatchValidationError"
    except PatchValidationError as exc:
        assert "camera_position" in str(exc)
    print("test_isovalue_agent_cannot_modify_camera_position: OK")


def test_invalid_patch_field_rejected():
    isovalue_spec = make_isovalue_spec()
    try:
        validate_patch({"not_a_real_field": 1}, isovalue_spec.owned_state_fields)
        assert False, "expected PatchValidationError"
    except PatchValidationError as exc:
        assert "do not exist" in str(exc)
    print("test_invalid_patch_field_rejected: OK")


def test_valid_patch_applies_without_mutating_original():
    state = make_state(isovalue=40.0)
    snapshot = clone_state(state)

    isovalue_spec = make_isovalue_spec()
    patch = {"isovalue": 55.0, "rendered_image_path": "output/new.png"}
    validate_patch(patch, isovalue_spec.owned_state_fields)  # must not raise
    new_state = apply_patch(state, patch)

    assert new_state.isovalue == 55.0
    assert new_state.rendered_image_path == "output/new.png"
    assert state.isovalue == 40.0, "apply_patch must not mutate the original state"
    assert snapshot.isovalue == 40.0
    print("test_valid_patch_applies_without_mutating_original: OK")


def test_failed_execution_rolls_state_back_via_clone():
    state = make_state(isovalue=40.0)
    snapshot = clone_state(state)

    # Simulate a specialist mutating its local working copy and then failing --
    # the executor's rollback path is just "discard new_state, keep snapshot".
    attempted = apply_patch(state, {"isovalue": 200.0})
    assert attempted.isovalue == 200.0

    rolled_back = snapshot
    assert rolled_back.isovalue == 40.0
    assert state.isovalue == 40.0
    print("test_failed_execution_rolls_state_back_via_clone: OK")


if __name__ == "__main__":
    test_camera_agent_cannot_modify_isovalue()
    test_isovalue_agent_cannot_modify_camera_position()
    test_invalid_patch_field_rejected()
    test_valid_patch_applies_without_mutating_original()
    test_failed_execution_rolls_state_back_via_clone()
    print("\nAll tests passed.")
