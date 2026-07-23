"""Deterministic, non-LLM validation of a planner-produced VisualizationPlan.

Catches everything the planner prompt asks the LLM not to do, but can't actually
guarantee on its own: duplicate task IDs, unknown capabilities, missing/self
dependencies, cycles, and empty success criteria. Planner.plan() uses this to reject
and retry a bad plan before it ever reaches the executor.
"""
from typing import List, Set

from .models import VisualizationPlan


class PlanValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _find_cycle(graph: dict) -> List[str]:
    """Return one cycle (list of task_ids) if the dependency graph has one, else []."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    path: List[str] = []

    def visit(node: str) -> List[str]:
        color[node] = GRAY
        path.append(node)
        for dep in graph.get(node, []):
            if dep not in color:
                continue  # unknown dependency; reported separately
            if color[dep] == GRAY:
                cycle_start = path.index(dep)
                return path[cycle_start:] + [dep]
            if color[dep] == WHITE:
                found = visit(dep)
                if found:
                    return found
        path.pop()
        color[node] = BLACK
        return []

    for node in graph:
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return []


def validate_plan(plan: VisualizationPlan, known_capabilities: Set[str]) -> None:
    """Raise PlanValidationError (with a list of human-readable problems) if `plan` is
    invalid. Returns None (no exception) if the plan is valid."""
    errors: List[str] = []

    task_ids = [t.task_id for t in plan.tasks]
    duplicate_ids = {tid for tid in task_ids if task_ids.count(tid) > 1}
    if duplicate_ids:
        errors.append(f"Duplicate task IDs: {sorted(duplicate_ids)}")

    id_set = set(task_ids)

    for task in plan.tasks:
        if task.required_capability not in known_capabilities:
            errors.append(
                f"Task {task.task_id!r} uses unknown capability {task.required_capability!r}; "
                f"known capabilities: {sorted(known_capabilities)}"
            )
        if not task.success_criteria:
            errors.append(f"Task {task.task_id!r} has empty success_criteria")
        for dep in task.dependencies:
            if dep == task.task_id:
                errors.append(f"Task {task.task_id!r} depends on itself")
            elif dep not in id_set:
                errors.append(f"Task {task.task_id!r} depends on unknown task {dep!r}")

    if not plan.final_success_criteria:
        errors.append("final_success_criteria must not be empty")

    if not duplicate_ids:  # cycle detection assumes a well-formed id set
        graph = {t.task_id: t.dependencies for t in plan.tasks}
        cycle = _find_cycle(graph)
        if cycle:
            errors.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

    if errors:
        raise PlanValidationError(errors)
