"""Tests for visualization_orchestrator.executor.VisualizationExecutor, using fake
specialists/planner/verifier so no real LLM/VTK calls happen.

No pytest required -- plain asserts, run directly:
  .venv/bin/python tests/test_orchestrator_executor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import (
    FakePlanner,
    FakeSpecialist,
    FakeVerifier,
    failure_result,
    make_camera_spec,
    make_isovalue_spec,
    make_state,
    success_result,
)
from visualization_orchestrator.executor import VisualizationExecutor
from visualization_orchestrator.models import FinalVerificationResult, PlannedTask, VisualizationPlan
from visualization_orchestrator.registry import AgentRegistry


def _registry(camera_specialist=None, isovalue_specialist=None):
    registry = AgentRegistry()
    camera_spec = make_camera_spec()
    isovalue_spec = make_isovalue_spec()
    registry.register(camera_specialist or FakeSpecialist("camera_controller", []), camera_spec)
    registry.register(isovalue_specialist or FakeSpecialist("isovalue_controller", []), isovalue_spec)
    return registry


def _task(task_id, capability, dependencies=None, max_attempts=1):
    return PlannedTask(
        task_id=task_id, goal=f"do {task_id}", required_capability=capability,
        dependencies=dependencies or [], success_criteria=[f"{task_id} looks right"], max_attempts=max_attempts,
    )


def _ok_verifier():
    return FakeVerifier([FinalVerificationResult(
        success=True, confidence=0.9, satisfied_criteria=["ok"], unsatisfied_criteria=[], diagnosis="ok",
    )])


def test_isovalue_only_request_runs_only_isovalue_specialist():
    camera = FakeSpecialist("camera_controller", [])
    isovalue = FakeSpecialist("isovalue_controller", [success_result("isovalue_controller", {"isovalue": 60.0})])
    registry = _registry(camera, isovalue)

    plan = VisualizationPlan(
        interpreted_goal="reduce noise",
        tasks=[_task("t1", "reduce_surface_noise")],
        final_success_criteria=["noise reduced"],
    )
    executor = VisualizationExecutor(registry=registry, planner=FakePlanner([plan]), verifier=_ok_verifier())
    result = executor.execute(plan, make_state(), "Reduce the surface noise.")

    assert result.success
    assert len(isovalue.calls) == 1
    assert len(camera.calls) == 0
    assert result.final_state.isovalue == 60.0
    print("test_isovalue_only_request_runs_only_isovalue_specialist: OK")


def test_camera_only_request_runs_only_camera_specialist():
    camera = FakeSpecialist("camera_controller", [success_result("camera_controller", {"camera_position": (1.0, 0.0, 0.0)})])
    isovalue = FakeSpecialist("isovalue_controller", [])
    registry = _registry(camera, isovalue)

    plan = VisualizationPlan(
        interpreted_goal="rotate",
        tasks=[_task("t1", "adjust_viewpoint")],
        final_success_criteria=["rotated"],
    )
    executor = VisualizationExecutor(registry=registry, planner=FakePlanner([plan]), verifier=_ok_verifier())
    result = executor.execute(plan, make_state(), "Rotate to the right.")

    assert result.success
    assert len(camera.calls) == 1
    assert len(isovalue.calls) == 0
    print("test_camera_only_request_runs_only_camera_specialist: OK")


def test_isovalue_followed_by_camera_order_preserved():
    camera = FakeSpecialist("camera_controller", [success_result("camera_controller", {"camera_position": (1.0, 0.0, 0.0)})])
    isovalue = FakeSpecialist("isovalue_controller", [success_result("isovalue_controller", {"isovalue": 60.0})])
    registry = _registry(camera, isovalue)

    plan = VisualizationPlan(
        interpreted_goal="reveal then view",
        tasks=[
            _task("reveal", "reveal_internal_structure"),
            _task("view", "show_object_from_direction", dependencies=["reveal"]),
        ],
        final_success_criteria=["done"],
    )
    executor = VisualizationExecutor(registry=registry, planner=FakePlanner([plan]), verifier=_ok_verifier())
    result = executor.execute(plan, make_state(), "Show the inside of the head from behind.")

    assert result.success
    assert [r.task_id for r in result.task_records] == ["reveal", "view"]
    print("test_isovalue_followed_by_camera_order_preserved: OK")


def test_camera_followed_by_isovalue_order_preserved():
    camera = FakeSpecialist("camera_controller", [success_result("camera_controller", {"camera_position": (1.0, 0.0, 0.0)})])
    isovalue = FakeSpecialist("isovalue_controller", [success_result("isovalue_controller", {"isovalue": 60.0})])
    registry = _registry(camera, isovalue)

    plan = VisualizationPlan(
        interpreted_goal="view then optimize",
        tasks=[
            _task("view", "show_object_from_direction"),
            _task("optimize", "reveal_internal_structure", dependencies=["view"]),
        ],
        final_success_criteria=["done"],
    )
    executor = VisualizationExecutor(registry=registry, planner=FakePlanner([plan]), verifier=_ok_verifier())
    result = executor.execute(
        plan, make_state(), "Move to the posterior view first, then find the best isovalue."
    )

    assert result.success
    assert [r.task_id for r in result.task_records] == ["view", "optimize"]
    print("test_camera_followed_by_isovalue_order_preserved: OK")


def test_already_satisfied_task_skipped_when_supported():
    """When the planner determines nothing needs to run (current state already satisfies
    the request), it returns a plan with zero tasks -- the executor must go straight to
    verification instead of erroring on an empty task list."""
    registry = _registry()
    plan = VisualizationPlan(
        interpreted_goal="already showing the skull",
        tasks=[],
        final_success_criteria=["skull is visible"],
    )
    executor = VisualizationExecutor(registry=registry, planner=FakePlanner([plan]), verifier=_ok_verifier())
    result = executor.execute(plan, make_state(), "Show the skull.")

    assert result.success
    assert result.task_records == []
    print("test_already_satisfied_task_skipped_when_supported: OK")


def test_specialist_failure_triggers_replanning():
    isovalue = FakeSpecialist("isovalue_controller", [failure_result("isovalue_controller", suggested=["adjust_opacity_transfer_function"])])
    camera = FakeSpecialist("camera_controller", [success_result("camera_controller", {"camera_position": (1.0, 0.0, 0.0)})])
    registry = _registry(camera, isovalue)

    failing_plan = VisualizationPlan(
        interpreted_goal="reveal", tasks=[_task("reveal", "reveal_internal_structure")],
        final_success_criteria=["done"],
    )
    replacement_plan = VisualizationPlan(
        interpreted_goal="reveal via camera", tasks=[_task("view", "show_object_from_direction")],
        final_success_criteria=["done"],
    )
    # `failing_plan` is passed directly to execute() below and never goes through
    # planner.plan() itself -- only REPLAN outputs come from the fake planner's queue.
    planner = FakePlanner([replacement_plan])
    executor = VisualizationExecutor(registry=registry, planner=planner, verifier=_ok_verifier(), max_replans=2)
    result = executor.execute(failing_plan, make_state(), "Show the inside of the head.")

    assert result.success
    assert result.replans_used == 1
    assert planner.calls[0]["failure_context"]["task_id"] == "reveal"
    assert len(camera.calls) == 1
    print("test_specialist_failure_triggers_replanning: OK")


def test_replanning_limit_respected():
    isovalue = FakeSpecialist("isovalue_controller", [failure_result("isovalue_controller")])
    registry = _registry(isovalue_specialist=isovalue)

    failing_plan = VisualizationPlan(
        interpreted_goal="reveal", tasks=[_task("reveal", "reveal_internal_structure")],
        final_success_criteria=["done"],
    )
    planner = FakePlanner([failing_plan])  # always returns the same failing plan
    executor = VisualizationExecutor(registry=registry, planner=planner, verifier=_ok_verifier(), max_replans=2)
    result = executor.execute(failing_plan, make_state(), "Show the inside of the head.")

    assert not result.success
    assert result.replans_used == 2
    print("test_replanning_limit_respected: OK")


if __name__ == "__main__":
    test_isovalue_only_request_runs_only_isovalue_specialist()
    test_camera_only_request_runs_only_camera_specialist()
    test_isovalue_followed_by_camera_order_preserved()
    test_camera_followed_by_isovalue_order_preserved()
    test_already_satisfied_task_skipped_when_supported()
    test_specialist_failure_triggers_replanning()
    test_replanning_limit_respected()
    print("\nAll tests passed.")
