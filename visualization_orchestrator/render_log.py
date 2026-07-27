"""Flat, explicit log of every render this orchestrator has committed -- one entry per
successfully-applied task, in order. Exists so a caller can:

  - see exactly which parameters produced any given image (rec.params <-> rec.image_path),
  - reproduce that exact render later (see VisualizationOrchestrator.render_from_params),
  - and ground a direct/relative instruction ("rotate left 60 degrees") against the last
    committed parameters explicitly and inspectably, rather than relying on the live VTK
    camera's hidden mutable state (which already applies deltas on top of the last
    finished query correctly, since it's the same persistent session across calls -- this
    log doesn't change that behavior, it just makes "what were the last params" a real,
    queryable thing instead of an implicit side effect).

Deliberately stores the FULL parameter snapshot, not state_summary()'s version -- that one
only reports `has_transfer_function: bool` for prompt brevity, which isn't enough to
reproduce a transfer-function-driven render later.
"""
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state import VisualizationState


def snapshot_params(state: VisualizationState) -> Dict[str, Any]:
    """Full, replay-capable parameter snapshot for one committed render."""
    return {
        "camera_position": list(state.camera_position),
        "focal_point": list(state.focal_point),
        "view_up": list(state.view_up),
        "isovalue": state.isovalue,
        "transfer_function": state.transfer_function,
        "clipping_planes": list(state.clipping_planes),
    }


@dataclass
class RenderLogEntry:
    index: int
    agent_id: Optional[str]
    task_id: Optional[str]
    goal: str
    params: Dict[str, Any]
    image_path: Optional[str]


@dataclass
class RenderLog:
    # If given, append() copies each render to its own file here instead of just storing
    # whatever path the state happened to report -- CameraReasoningSession.render_and_save()
    # always writes to the same "latest.png", so without a persisted copy, every earlier
    # entry's image_path would end up silently pointing at whatever got rendered LAST
    # (overwritten in place), making old entries unreadable even though their params are
    # still correct.
    image_dir: Optional[str] = None
    entries: List[RenderLogEntry] = field(default_factory=list)

    def append(
        self, *, agent_id: Optional[str], task_id: Optional[str], goal: str, state: VisualizationState,
    ) -> RenderLogEntry:
        index = len(self.entries)
        image_path = self._persist_image(index, state.rendered_image_path)
        entry = RenderLogEntry(
            index=index, agent_id=agent_id, task_id=task_id, goal=goal,
            params=snapshot_params(state), image_path=image_path,
        )
        self.entries.append(entry)
        return entry

    def _persist_image(self, index: int, source_path: Optional[str]) -> Optional[str]:
        if not self.image_dir or not source_path or not Path(source_path).exists():
            return source_path
        dest_dir = Path(self.image_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"render_{index:04d}{Path(source_path).suffix}"
        shutil.copy(source_path, dest)
        return str(dest)

    @property
    def last(self) -> Optional[RenderLogEntry]:
        return self.entries[-1] if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> RenderLogEntry:
        return self.entries[index]
