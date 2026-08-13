"""Shared visualization state passed between the planner, executor, and every specialist.

Every specialist may READ the full state but must only WRITE the fields it owns
(AgentSpec.owned_state_fields, see capabilities.py). Ownership is enforced here via
validate_patch(), not scattered across the executor or individual specialists.
"""
import copy
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, Optional


@dataclass
class VisualizationState:
    dataset_path: str

    # Camera state -- owned by camera specialists.
    camera_position: tuple
    focal_point: tuple
    view_up: tuple
    zoom: float = 1.0

    # Goal-neutral description of what the dataset IS (e.g. "CT scan of a mouse hindlimb"),
    # extracted by the planner from the user's own instruction -- see
    # planner_prompt.py's PLANNER_SYSTEM_INSTRUCTIONS. Deliberately separate from any task's
    # "goal" field: specialists that must stay goal-blind (see
    # specialists/isovalue_adapter.py's Stage 1) may still read this, since it says what kind
    # of data is being looked at without saying what to find in it. Not owned by any
    # specialist -- set once by the orchestrator after planning (VisualizationOrchestrator.run),
    # never via a specialist's state_patch.
    dataset_description: Optional[str] = None

    # Rendering and visibility state -- owned by isovalue/transfer-function/clipping specialists.
    isovalue: Optional[float] = None
    transfer_function: Optional[Any] = None
    clipping_planes: list = field(default_factory=list)

    # Semantic state
    selected_target: Optional[str] = None
    current_view_description: Optional[str] = None

    # Output state
    rendered_image_path: Optional[str] = None
    saved_outputs: list = field(default_factory=list)

    # Execution history (informational only -- the authoritative record lives in
    # history.ExecutionHistory / ExecutionResult.task_records; this field exists so a
    # state summary handed to the planner/verifier can carry a short trailing digest).
    history: list = field(default_factory=list)


_STATE_FIELD_NAMES = {f.name for f in fields(VisualizationState)}


class PatchValidationError(ValueError):
    """Raised when a specialist's state_patch touches fields it doesn't own, or a field
    that doesn't exist on VisualizationState at all."""


def clone_state(state: VisualizationState) -> VisualizationState:
    """Deep-copy a state so a specialist's in-progress work can be rolled back on failure
    without mutating the caller's live state."""
    return copy.deepcopy(state)


def validate_patch(patch: Dict[str, Any], owned_fields: set) -> None:
    """Raise PatchValidationError unless every key in `patch` is both a real
    VisualizationState field and one this agent is declared to own."""
    unknown_fields = set(patch) - _STATE_FIELD_NAMES
    if unknown_fields:
        raise PatchValidationError(
            f"state_patch references fields that do not exist on VisualizationState: {sorted(unknown_fields)}"
        )
    unowned_fields = set(patch) - owned_fields
    if unowned_fields:
        raise PatchValidationError(
            f"state_patch touches fields this agent does not own: {sorted(unowned_fields)} "
            f"(owned fields: {sorted(owned_fields)})"
        )


def apply_patch(state: VisualizationState, patch: Dict[str, Any]) -> VisualizationState:
    """Return a NEW state with `patch` applied on top of `state`. Never mutates `state` --
    callers that want a rollback point should keep the pre-patch state (see clone_state)."""
    return replace(state, **patch)


def state_summary(state: VisualizationState) -> dict:
    """Compact, JSON-friendly summary of the current state for planner/verifier prompts.
    Large/opaque fields (transfer_function) are summarized rather than dumped in full."""
    return {
        "dataset_path": state.dataset_path,
        "dataset_description": state.dataset_description,
        "camera_position": list(state.camera_position),
        "focal_point": list(state.focal_point),
        "view_up": list(state.view_up),
        "zoom": state.zoom,
        "isovalue": state.isovalue,
        "has_transfer_function": state.transfer_function is not None,
        "clipping_planes_count": len(state.clipping_planes),
        "selected_target": state.selected_target,
        "current_view_description": state.current_view_description,
        "rendered_image_path": state.rendered_image_path,
        "saved_outputs": list(state.saved_outputs),
    }
