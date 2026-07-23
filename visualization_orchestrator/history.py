"""Execution history recorder -- a plain append-only log of per-task state transitions,
used both for debugging output and as the "recent execution history" context fed back
into replanning prompts.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExecutionHistory:
    entries: List[dict] = field(default_factory=list)

    def record(self, **kwargs) -> None:
        self.entries.append(kwargs)

    def summary(self, limit: int = 10) -> str:
        if not self.entries:
            return "(none yet)"
        lines = [
            f"- task={e.get('task_id')} agent={e.get('agent_id')} status={e.get('status')} "
            f"reason={e.get('reason')}"
            for e in self.entries[-limit:]
        ]
        return "\n".join(lines)
