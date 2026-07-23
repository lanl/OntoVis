"""High-level facade bundling a CameraReasoningSession + the full registry/planner/
executor/verifier stack into one object, so a caller (a notebook, a script, a future
GUI) can interact with the whole system without wiring the pieces together by hand --
see notebooks/visualization_orchestrator_demo.ipynb and demo.py for callers.
"""
from typing import List, Optional

from camera_reasoning.session import CameraReasoningSession

from .executor import VisualizationExecutor
from .models import ExecutionResult
from .planner import Planner
from .registry import AgentRegistry
from .specialists import CAMERA_AGENT_SPEC, CameraSpecialist, ISOVALUE_AGENT_SPEC, IsovalueSpecialist
from .state import VisualizationState
from .verifier import FinalVerifier


class VisualizationOrchestrator:
    """One object wrapping a live VTK session and the planner/registry/executor/verifier
    stack. State carries forward between calls to `.run()`, so a sequence of instructions
    behaves like a multi-turn conversation against the same visualization.
    """

    def __init__(
        self,
        dataset_path: str,
        dimensions: tuple,
        scalar_type: str = "uint8",
        isovalue: float = 40,
        output_dir: str = "output/orchestrator_session",
        model: Optional[str] = None,
        max_replans: int = 2,
        max_total_tasks: int = 10,
        max_total_agent_calls: int = 10,
    ):
        self.session = CameraReasoningSession(
            raw_path=dataset_path,
            dimensions=dimensions,
            scalar_type=scalar_type,
            isovalue=isovalue,
            output_dir=output_dir,
            target_description="No target description provided.",
        )
        self.session.initialize()

        self.registry = AgentRegistry()
        self.registry.register(CameraSpecialist(self.session, model=model), CAMERA_AGENT_SPEC)
        self.registry.register(IsovalueSpecialist(self.session, model=model), ISOVALUE_AGENT_SPEC)

        self.planner = Planner(model=model)
        self.executor = VisualizationExecutor(
            registry=self.registry,
            planner=self.planner,
            verifier=FinalVerifier(model=model),
            max_replans=max_replans,
            max_total_tasks=max_total_tasks,
            max_total_agent_calls=max_total_agent_calls,
        )

        self.state: VisualizationState = self._build_initial_state()
        self.history: List[ExecutionResult] = []

    def run(self, instruction: str) -> ExecutionResult:
        """Plan and execute one instruction against the current state, carrying the
        resulting state forward for the next call."""
        plan = self.planner.plan(instruction, self.state, self.registry)
        print(f"[Orchestrator] Interpreted goal: {plan.interpreted_goal}")
        print(f"[Orchestrator] Task graph: {[(t.task_id, t.required_capability, t.dependencies) for t in plan.tasks]}")

        result = self.executor.execute(plan, self.state, instruction)
        self.state = result.final_state
        self.history.append(result)

        print(f"[Orchestrator] Success: {result.success} (replans={result.replans_used}, "
              f"agent_calls={result.total_agent_calls})")
        return result

    @property
    def image_path(self) -> Optional[str]:
        """Path to the most recently rendered image, for inline display."""
        return self.state.rendered_image_path

    def _build_initial_state(self) -> VisualizationState:
        image_path = self.session.render_and_save()
        camera = self.session._renderer.GetActiveCamera()
        return VisualizationState(
            dataset_path=self.session.raw_path,
            camera_position=tuple(camera.GetPosition()),
            focal_point=tuple(camera.GetFocalPoint()),
            view_up=tuple(camera.GetViewUp()),
            isovalue=self.session.isovalue,
            rendered_image_path=image_path,
        )
