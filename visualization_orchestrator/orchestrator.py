"""High-level facade bundling a CameraReasoningSession + the full registry/planner/
executor/verifier stack into one object, so a caller (a notebook, a script, a future
GUI) can interact with the whole system without wiring the pieces together by hand --
see notebooks/visualization_orchestrator_demo.ipynb and demo.py for callers.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from camera_reasoning.session import CameraReasoningSession

from .executor import VisualizationExecutor
from .models import ExecutionResult
from .planner import Planner
from .registry import AgentRegistry
from .render_log import RenderLog
from .specialists import (
    APPLY_EXACT_PARAMETERS_SPEC,
    CAMERA_AGENT_SPEC,
    CameraSpecialist,
    ISOVALUE_AGENT_SPEC,
    IsovalueSpecialist,
    ORIENTATION_AGENT_SPEC,
    OrientationSpecialist,
    ApplyExactParametersSpecialist,
)
from .state import VisualizationState, apply_patch
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
        spacing: tuple = (1.0, 1.0, 1.0),
        isovalue: float = 40,
        output_dir: str = "output/orchestrator_session",
        model: Optional[str] = None,
        max_replans: int = 2,
        max_total_tasks: int = 10,
        max_total_agent_calls: int = 10,
        camera_reference_image_paths: Optional[Dict[str, str]] = None,
        camera_node_descriptions: Optional[Dict[str, str]] = None,
        on_iteration: Optional[Callable[[dict], None]] = None,
        camera_sequential_diagnosis: bool = False,
    ):
        """`spacing` defaults to isotropic (1.0, 1.0, 1.0) -- pass the dataset's real
        per-axis voxel spacing for an anisotropic volume (e.g. vis_male_128x256x256's
        (1.57774, 0.995861, 1.00797), see camera_reasoning/volume_scene.py's
        load_raw_volume docstring). Getting this wrong silently distorts geometry (e.g.
        squeezes a shorter axis) rather than raising an error.

        `camera_reference_image_paths`/`camera_node_descriptions` are passed straight
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

        `camera_sequential_diagnosis` (default False): diagnose all blind candidates
        together in one LLM call per iteration -- fast, but the model can occasionally
        mislabel which candidate a score/description belongs to when several
        similar-looking renders are shown at once. Set True to diagnose candidates one
        LLM call at a time instead (N calls per iteration, N = candidate count), trading
        speed for that per-candidate isolation.
        """
        self.session = CameraReasoningSession(
            raw_path=dataset_path,
            dimensions=dimensions,
            scalar_type=scalar_type,
            spacing=spacing,
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
        self.registry.register(ApplyExactParametersSpecialist(self.session), APPLY_EXACT_PARAMETERS_SPEC)

        self.render_log = RenderLog(image_dir=str(Path(output_dir) / "render_log"))
        self.planner = Planner(model=model)
        self.executor = VisualizationExecutor(
            registry=self.registry,
            planner=self.planner,
            verifier=FinalVerifier(model=model),
            render_log=self.render_log,
            max_replans=max_replans,
            max_total_tasks=max_total_tasks,
            max_total_agent_calls=max_total_agent_calls,
        )

        self.state: VisualizationState = self._build_initial_state()
        self.render_log.append(agent_id=None, task_id=None, goal="initial state", state=self.state)
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

    def render_from_log(self, index: int) -> str:
        """Reproduce a past render_log entry's exact image (e.g. orchestrator.render_from_log(-3)),
        and make it the current state going forward -- see render_from_params()."""
        return self.render_from_params(self.render_log[index].params)

    def render_from_params(self, params: Dict) -> str:
        """Deterministically re-render from a full parameter snapshot (see
        render_log.snapshot_params -- the same shape every render_log entry stores), and
        adopt it as the current state so subsequent instructions (including a direct,
        relative one like "rotate left 60 degrees") build on top of THIS render rather
        than wherever the live camera happened to be before this call.
        """
        self.session._require_initialized()
        camera = self.session._renderer.GetActiveCamera()
        camera.SetPosition(*params["camera_position"])
        camera.SetFocalPoint(*params["focal_point"])
        camera.SetViewUp(*params["view_up"])
        camera.OrthogonalizeViewUp()
        self.session._renderer.ResetCameraClippingRange()

        if params.get("transfer_function") is not None:
            opacity_points, color_points = params["transfer_function"]
            self.session.set_transfer_function(opacity_points, color_points)
        elif params.get("isovalue") is not None:
            self.session.set_isovalue(params["isovalue"])

        image_path = self.session.render_and_save()
        self.state = apply_patch(self.state, {
            "camera_position": tuple(params["camera_position"]),
            "focal_point": tuple(params["focal_point"]),
            "view_up": tuple(params["view_up"]),
            "isovalue": params.get("isovalue"),
            "transfer_function": params.get("transfer_function"),
            "rendered_image_path": image_path,
        })
        self.render_log.append(agent_id="replay", task_id=None, goal="replay stored parameters", state=self.state)
        return image_path

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
