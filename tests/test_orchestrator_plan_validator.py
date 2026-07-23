"""Tests for visualization_orchestrator.plan_validator.

No pytest required -- plain asserts, run directly:
  .venv/bin/python tests/test_orchestrator_plan_validator.py
(Also pytest-collectible, since every test is a module-level `test_*` function.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization_orchestrator.models import PlannedTask, VisualizationPlan
from visualization_orchestrator.plan_validator import PlanValidationError, validate_plan

KNOWN_CAPABILITIES = {"reveal_internal_structure", "show_object_from_direction", "reduce_surface_noise"}


def _task(**overrides) -> PlannedTask:
    defaults = dict(
        task_id="t1", goal="do something", required_capability="reveal_internal_structure",
        dependencies=[], success_criteria=["it looks right"],
    )
    defaults.update(overrides)
    return PlannedTask(**defaults)


def test_valid_linear_plan_passes():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[
            _task(task_id="a", required_capability="reveal_internal_structure"),
            _task(task_id="b", required_capability="show_object_from_direction", dependencies=["a"]),
        ],
        final_success_criteria=["overall looks right"],
    )
    validate_plan(plan, KNOWN_CAPABILITIES)  # must not raise
    print("test_valid_linear_plan_passes: OK")


def test_unknown_capability_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[_task(required_capability="teleport_camera")],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("unknown capability" in e for e in exc.errors)
    print("test_unknown_capability_rejected: OK")


def test_missing_dependency_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[_task(task_id="a", dependencies=["does_not_exist"])],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("unknown task" in e for e in exc.errors)
    print("test_missing_dependency_rejected: OK")


def test_cyclic_dependency_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[
            _task(task_id="a", dependencies=["b"]),
            _task(task_id="b", dependencies=["a"]),
        ],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("cycle" in e.lower() for e in exc.errors)
    print("test_cyclic_dependency_rejected: OK")


def test_duplicate_task_id_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[_task(task_id="a"), _task(task_id="a", required_capability="show_object_from_direction")],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("Duplicate task IDs" in e for e in exc.errors)
    print("test_duplicate_task_id_rejected: OK")


def test_empty_success_criteria_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal",
        tasks=[_task(success_criteria=[])],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("empty success_criteria" in e for e in exc.errors)
    print("test_empty_success_criteria_rejected: OK")


def test_empty_final_success_criteria_rejected():
    plan = VisualizationPlan(interpreted_goal="goal", tasks=[_task()], final_success_criteria=[])
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("final_success_criteria" in e for e in exc.errors)
    print("test_empty_final_success_criteria_rejected: OK")


def test_self_dependency_rejected():
    plan = VisualizationPlan(
        interpreted_goal="goal", tasks=[_task(task_id="a", dependencies=["a"])],
        final_success_criteria=["overall looks right"],
    )
    try:
        validate_plan(plan, KNOWN_CAPABILITIES)
        assert False, "expected PlanValidationError"
    except PlanValidationError as exc:
        assert any("depends on itself" in e for e in exc.errors)
    print("test_self_dependency_rejected: OK")


if __name__ == "__main__":
    test_valid_linear_plan_passes()
    test_unknown_capability_rejected()
    test_missing_dependency_rejected()
    test_cyclic_dependency_rejected()
    test_duplicate_task_id_rejected()
    test_empty_success_criteria_rejected()
    test_empty_final_success_criteria_rejected()
    test_self_dependency_rejected()
    print("\nAll tests passed.")
