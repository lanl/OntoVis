"""Tests for visualization_orchestrator.verifier.FinalVerifier (with ask_chatgpt monkey-
patched -- no real LLM call) and for the executor's targeted-replan wiring when final
verification fails.

No pytest required -- plain asserts, run directly:
  .venv/bin/python tests/test_orchestrator_verifier.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator_fakes import FakePlanner, FakeSpecialist, make_camera_spec, make_isovalue_spec, make_state, success_result
from visualization_orchestrator import verifier as verifier_module
from visualization_orchestrator.executor import VisualizationExecutor
from visualization_orchestrator.models import PlannedTask, VisualizationPlan
from visualization_orchestrator.registry import AgentRegistry
from visualization_orchestrator.verifier import FinalVerifier


class _PatchedAskChatgpt:
    """Context manager that temporarily replaces verifier_module.ask_chatgpt."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self._original = None

    def __enter__(self):
        self._original = verifier_module.ask_chatgpt
        verifier_module.ask_chatgpt = lambda **kwargs: self.response_text
        return self

    def __exit__(self, *exc_info):
        verifier_module.ask_chatgpt = self._original


def test_all_criteria_satisfied():
    response = json.dumps({
        "success": True, "confidence": 0.95,
        "satisfied_criteria": ["internal structures visible", "posterior view"],
        "unsatisfied_criteria": [], "diagnosis": "Looks correct.", "suggested_capabilities": [],
    })
    with _PatchedAskChatgpt(response):
        result = FinalVerifier().verify(
            "Show the inside of the head from behind.", make_state(), [],
            ["internal structures visible", "posterior view"],
        )
    assert result.success is True
    assert result.unsatisfied_criteria == []
    print("test_all_criteria_satisfied: OK")


def test_camera_criterion_fails():
    response = json.dumps({
        "success": False, "confidence": 0.7,
        "satisfied_criteria": ["internal structures visible"],
        "unsatisfied_criteria": ["camera is not showing a posterior view"],
        "diagnosis": "Internal structure is revealed but the camera is still frontal.",
        "suggested_capabilities": ["show_object_from_direction"],
    })
    with _PatchedAskChatgpt(response):
        result = FinalVerifier().verify(
            "Show the inside of the head from behind.", make_state(), [],
            ["internal structures visible", "posterior view"],
        )
    assert result.success is False
    assert "camera is not showing a posterior view" in result.unsatisfied_criteria
    assert "show_object_from_direction" in result.suggested_capabilities
    print("test_camera_criterion_fails: OK")


def test_visibility_criterion_fails():
    response = json.dumps({
        "success": False, "confidence": 0.6,
        "satisfied_criteria": ["posterior view"],
        "unsatisfied_criteria": ["internal structures are not visibly exposed"],
        "diagnosis": "Camera is correctly posterior but the outer surface still occludes everything.",
        "suggested_capabilities": ["reveal_internal_structure", "apply_clipping"],
    })
    with _PatchedAskChatgpt(response):
        result = FinalVerifier().verify(
            "Show the inside of the head from behind.", make_state(), [],
            ["internal structures visible", "posterior view"],
        )
    assert result.success is False
    assert "internal structures are not visibly exposed" in result.unsatisfied_criteria
    print("test_visibility_criterion_fails: OK")


def test_targeted_repair_plan_is_generated_on_verification_failure():
    """When final verification fails after every task individually succeeded, the executor
    must hand the unsatisfied criteria back to the planner (targeted repair), not restart
    the whole plan from scratch."""
    from visualization_orchestrator.models import FinalVerificationResult

    class TwoStageFakeVerifier:
        def __init__(self):
            self.calls = []
            self._results = [
                FinalVerificationResult(
                    success=False, confidence=0.5, satisfied_criteria=["internal structures visible"],
                    unsatisfied_criteria=["posterior view"], diagnosis="Camera still frontal.",
                    suggested_capabilities=["show_object_from_direction"],
                ),
                FinalVerificationResult(
                    success=True, confidence=0.9, satisfied_criteria=["internal structures visible", "posterior view"],
                    unsatisfied_criteria=[], diagnosis="ok",
                ),
            ]

        def verify(self, user_instruction, state, task_records, final_success_criteria):
            index = min(len(self.calls), len(self._results) - 1)
            self.calls.append(True)
            return self._results[index]

    registry = AgentRegistry()
    camera = FakeSpecialist("camera_controller", [success_result("camera_controller", {"camera_position": (1.0, 0.0, 0.0)})])
    isovalue = FakeSpecialist("isovalue_controller", [success_result("isovalue_controller", {"isovalue": 60.0})])
    registry.register(camera, make_camera_spec())
    registry.register(isovalue, make_isovalue_spec())

    first_plan = VisualizationPlan(
        interpreted_goal="reveal internal structure",
        tasks=[PlannedTask(task_id="reveal", goal="reveal", required_capability="reveal_internal_structure", success_criteria=["revealed"])],
        final_success_criteria=["internal structures visible", "posterior view"],
    )
    repair_plan = VisualizationPlan(
        interpreted_goal="add posterior view",
        tasks=[PlannedTask(task_id="view", goal="view", required_capability="show_object_from_direction", success_criteria=["posterior"])],
        final_success_criteria=["internal structures visible", "posterior view"],
    )
    # `first_plan` is passed directly to execute() below and never goes through
    # planner.plan() itself -- only the REPLAN output comes from the fake planner's queue.
    planner = FakePlanner([repair_plan])
    verifier = TwoStageFakeVerifier()

    executor = VisualizationExecutor(registry=registry, planner=planner, verifier=verifier, max_replans=2)
    result = executor.execute(first_plan, make_state(), "Show the inside of the head from behind.")

    assert result.success
    assert result.replans_used == 1
    assert planner.calls[0]["failure_context"]["unsatisfied_criteria"] == ["posterior view"]
    assert len(camera.calls) == 1  # only ran once, via the repair plan
    print("test_targeted_repair_plan_is_generated_on_verification_failure: OK")


if __name__ == "__main__":
    test_all_criteria_satisfied()
    test_camera_criterion_fails()
    test_visibility_criterion_fails()
    test_targeted_repair_plan_is_generated_on_verification_failure()
    print("\nAll tests passed.")
