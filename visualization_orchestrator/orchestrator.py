"""High-level facade bundling a CameraReasoningSession + the full registry/planner/
executor/verifier stack into one object, so a caller (a notebook, a script, a future
GUI) can interact with the whole system without wiring the pieces together by hand --
see notebooks/visualization_orchestrator_demo.ipynb and demo.py for callers.
"""
from typing import Callable, Dict, List, Optional

from camera_reasoning.session import CameraReasoningSession

from .executor import VisualizationExecutor
from .models import ExecutionResult
from .planner import Planner
from .registry import AgentRegistry
from .specialists import (
    CAMERA_AGENT_SPEC,
    CameraSpecialist,
    ISOVALUE_AGENT_SPEC,
    IsovalueSpecialist,
    ORIENTATION_AGENT_SPEC,
    OrientationSpecialist,
)
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
        camera_reference_image_paths: Optional[Dict[str, str]] = None,
        camera_node_descriptions: Optional[Dict[str, str]] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
        camera_sequential_diagnosis: bool = True,
    ):
        """`camera_reference_image_paths`/`camera_node_descriptions` are passed straight
        through to CameraSpecialist's reference-view bank (see
        specialists.camera_adapter.load_reference_bank) -- not loaded automatically here.
        If you have a bank generated for this exact dataset/isovalue (e.g. via
        examples/generate_camera_relative_views.py), load it yourself and pass it in;
        otherwise the camera agent still works, just without reference-grounded selection.

        `on_iteration`, if given, is called once per INTERNAL specialist iteration (not
        just once per `.run()` call) with a normalized dict: {"agent_id", "iteration",
        "current_image_path", "candidates", "selected_label", "reasoning", "extra"}.
        `candidates` is a list of per-candidate dicts, shape depends on the agent:
          - camera: {"label", "image_path", "selected", "real_action", "similarity_score",
            "confidence", "reference_match_quality", "comparison_to_target"}, one call per
            iteration of the internal loop.
          - isovalue: {"label", "image_path", "selected", "real_action" (always None),
            "observation", "satisfies_goal"}, one call total (a single batched candidate
            sweep, reported as iteration 0).
          - orientation: `candidates` is always empty (single current image per iteration);
            `extra` carries {"is_correctly_oriented", "roll_degrees", "observation"}.
        Pass e.g. a notebook display function to see progress live instead of only the
        final result of each `.run()` call.

        `camera_sequential_diagnosis` (default True): diagnose blind candidates one LLM
        call at a time instead of all together in one call -- more LLM calls per
        iteration, but avoids the model mislabeling which candidate a score/description
        belongs to when several similar-looking renders are shown at once. Set False to
        go back to the single-batched-call behavior.
        """
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
        self.registry.register(
            CameraSpecialist(
                self.session,
                model=model,
                reference_image_paths=camera_reference_image_paths,
                node_descriptions=camera_node_descriptions,
                on_iteration=on_iteration,
                sequential_diagnosis=camera_sequential_diagnosis,
            ),
            CAMERA_AGENT_SPEC,
        )
        self.registry.register(
            IsovalueSpecialist(self.session, model=model, on_iteration=on_iteration),
            ISOVALUE_AGENT_SPEC,
        )
        self.registry.register(
            OrientationSpecialist(self.session, model=model, on_iteration=on_iteration),
            ORIENTATION_AGENT_SPEC,
        )

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
