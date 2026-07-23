"""Deterministic task-graph executor.

This is plain Python, not an LLM call: it selects ready tasks (dependencies satisfied),
resolves a concrete agent via the registry, calls that specialist's `run_until_complete`
as an opaque high-level unit (its internal render/evaluate/act loop is never touched
here), validates and transactionally applies the returned state patch, and records every
transition. Replanning is delegated to Replanner and is bounded by max_replans;
total work is additionally bounded by max_total_tasks / max_total_agent_calls so nothing
retries or replans indefinitely.
"""
from typing import Dict, List, Optional, Set

from .history import ExecutionHistory
from .models import (
    AgentExecutionResult,
    ExecutionResult,
    FinalVerificationResult,
    PlannedTask,
    TaskExecutionRecord,
    VisualizationPlan,
)
from .plan_validator import PlanValidationError, validate_plan
from .planner import Planner, PlannerError
from .registry import AgentRegistry
from .replanner import Replanner
from .state import (
    PatchValidationError,
    VisualizationState,
    apply_patch,
    clone_state,
    state_summary,
    validate_patch,
)
from .verifier import FinalVerifier


class VisualizationExecutor:
    def __init__(
        self,
        registry: AgentRegistry,
        planner: Optional[Planner] = None,
        verifier: Optional[FinalVerifier] = None,
        max_replans: int = 2,
        max_total_tasks: int = 10,
        max_total_agent_calls: int = 10,
    ):
        self.registry = registry
        self.planner = planner or Planner()
        self.replanner = Replanner(self.planner)
        self.verifier = verifier or FinalVerifier()
        self.max_replans = max_replans
        self.max_total_tasks = max_total_tasks
        self.max_total_agent_calls = max_total_agent_calls

    def execute(
        self, plan: VisualizationPlan, initial_state: VisualizationState, user_instruction: str
    ) -> ExecutionResult:
        validate_plan(plan, self.registry.known_capabilities())

        state = initial_state
        history = ExecutionHistory()
        records: List[TaskExecutionRecord] = []
        counters = {"tasks_run": 0, "agent_calls": 0, "replans": 0}

        remaining_tasks: List[PlannedTask] = list(plan.tasks)
        completed_ids: Set[str] = set()
        task_attempts: Dict[str, int] = {}
        current_final_criteria = plan.final_success_criteria
        current_goal = plan.interpreted_goal

        while True:
            state, failure_context = self._run_tasks_to_completion_or_stall(
                remaining_tasks, state, completed_ids, task_attempts, history, records, counters
            )

            if failure_context is not None:
                replanned = self._try_replan(
                    user_instruction, state, history, failure_context, counters
                )
                if replanned is None:
                    return self._finish(
                        success=False,
                        interpreted_goal=current_goal,
                        state=state,
                        records=records,
                        verification=None,
                        counters=counters,
                        diagnosis=failure_context.get("reason", "Execution failed."),
                    )
                remaining_tasks = replanned.tasks
                current_final_criteria = replanned.final_success_criteria
                current_goal = replanned.interpreted_goal
                continue

            # every task in remaining_tasks is now in completed_ids -- verify the whole goal.
            verification = self.verifier.verify(
                user_instruction, state, records, current_final_criteria
            )
            print(
                f"[Verifier] Final goal satisfied={verification.success} "
                f"confidence={verification.confidence}"
            )
            if verification.success:
                return self._finish(
                    success=True,
                    interpreted_goal=current_goal,
                    state=state,
                    records=records,
                    verification=verification,
                    counters=counters,
                )

            failure_context = {
                "task_id": None,
                "required_capability": None,
                "reason": verification.diagnosis,
                "unsatisfied_criteria": verification.unsatisfied_criteria,
                "suggested_capabilities": verification.suggested_capabilities,
            }
            replanned = self._try_replan(user_instruction, state, history, failure_context, counters)
            if replanned is None:
                return self._finish(
                    success=False,
                    interpreted_goal=current_goal,
                    state=state,
                    records=records,
                    verification=verification,
                    counters=counters,
                    diagnosis=verification.diagnosis,
                )
            remaining_tasks = replanned.tasks
            current_final_criteria = replanned.final_success_criteria
            current_goal = replanned.interpreted_goal
            # Tasks the replan reuses by task_id that are already completed are treated as
            # still satisfied and skipped; anything with a new task_id runs fresh.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tasks_to_completion_or_stall(
        self,
        tasks: List[PlannedTask],
        state: VisualizationState,
        completed_ids: Set[str],
        task_attempts: Dict[str, int],
        history: ExecutionHistory,
        records: List[TaskExecutionRecord],
        counters: dict,
    ):
        """Run ready tasks one at a time until every task in `tasks` is complete, or a
        task permanently fails, or a resource limit is hit. Returns (state, failure_context)
        where failure_context is None iff every task completed successfully."""
        task_excluded_agents: Dict[str, Set[str]] = {}

        while True:
            pending = [t for t in tasks if t.task_id not in completed_ids]
            if not pending:
                return state, None

            ready = [t for t in pending if all(dep in completed_ids for dep in t.dependencies)]
            if not ready:
                return state, {
                    "task_id": None,
                    "required_capability": None,
                    "reason": (
                        f"Execution stalled: no ready task among {[t.task_id for t in pending]} "
                        "-- remaining dependencies can never be satisfied."
                    ),
                    "suggested_capabilities": [],
                }

            if counters["tasks_run"] >= self.max_total_tasks:
                return state, {
                    "task_id": None, "required_capability": None,
                    "reason": "max_total_tasks limit reached before all tasks completed.",
                    "suggested_capabilities": [],
                }
            if counters["agent_calls"] >= self.max_total_agent_calls:
                return state, {
                    "task_id": None, "required_capability": None,
                    "reason": "max_total_agent_calls limit reached before all tasks completed.",
                    "suggested_capabilities": [],
                }

            task = ready[0]
            print(f"[Executor] Ready task: {task.task_id}")

            constraints = {c.key: c.value for c in task.constraints}
            excluded = task_excluded_agents.get(task.task_id, set())
            spec = self.registry.select_agent(
                task.required_capability, state, constraints,
                preferred_agent_id=task.preferred_agent_id, excluded_agent_ids=excluded,
            )

            if spec is None:
                reason = (
                    f"No registered agent can satisfy capability {task.required_capability!r} "
                    "given the current state/constraints."
                )
                print(f"[Registry] {reason}")
                records.append(TaskExecutionRecord(
                    task_id=task.task_id, agent_id=None, status="failed", goal=task.goal,
                    state_before=state_summary(state), state_patch={},
                    state_after=state_summary(state), reason=reason, iterations_used=0,
                ))
                return state, {
                    "task_id": task.task_id, "required_capability": task.required_capability,
                    "reason": reason, "unsatisfied_criteria": task.success_criteria,
                    "suggested_capabilities": [],
                }

            print(f"[Registry] Selected {spec.agent_id} for {task.required_capability}")
            specialist = self.registry.get_specialist(spec.agent_id)

            snapshot = clone_state(state)
            counters["agent_calls"] += 1
            counters["tasks_run"] += 1
            task_attempts[task.task_id] = task_attempts.get(task.task_id, 0) + 1

            result = specialist.run_until_complete(task.goal, state, constraints, task.success_criteria)
            print(
                f"[Agent:{spec.agent_id}] status={result.status} goal_satisfied={result.goal_satisfied} "
                f"after {result.iterations_used} iteration(s) -- {result.reason}"
            )

            attempts_exhausted = task_attempts[task.task_id] >= task.max_attempts
            accept = result.status in ("success", "partial") and (result.goal_satisfied or attempts_exhausted)

            if accept:
                state, error = self._apply_result(spec, snapshot, result, task, history, records)
                if error is not None:
                    return state, error
                completed_ids.add(task.task_id)
                continue

            # Not accepted -- roll back and either retry or give up on this task permanently.
            state = snapshot
            records.append(TaskExecutionRecord(
                task_id=task.task_id, agent_id=spec.agent_id, status="failed", goal=task.goal,
                state_before=state_summary(snapshot), state_patch={},
                state_after=state_summary(snapshot), reason=result.reason,
                iterations_used=result.iterations_used,
            ))
            history.record(task_id=task.task_id, agent_id=spec.agent_id, status="failed", reason=result.reason)

            if not attempts_exhausted:
                print(
                    f"[Executor] Task {task.task_id} failed (attempt "
                    f"{task_attempts[task.task_id]}/{task.max_attempts}); retrying."
                )
                task_excluded_agents.setdefault(task.task_id, set())
                continue

            return state, {
                "task_id": task.task_id, "required_capability": task.required_capability,
                "reason": result.reason, "unsatisfied_criteria": result.unsatisfied_criteria,
                "suggested_capabilities": result.suggested_capabilities,
            }

    def _apply_result(
        self,
        spec,
        snapshot: VisualizationState,
        result: AgentExecutionResult,
        task: PlannedTask,
        history: ExecutionHistory,
        records: List[TaskExecutionRecord],
    ):
        try:
            validate_patch(result.state_patch, spec.owned_state_fields)
        except PatchValidationError as exc:
            reason = f"Rejected state_patch from {spec.agent_id}: {exc}"
            print(f"[Executor] {reason}")
            records.append(TaskExecutionRecord(
                task_id=task.task_id, agent_id=spec.agent_id, status="failed", goal=task.goal,
                state_before=state_summary(snapshot), state_patch={},
                state_after=state_summary(snapshot), reason=reason,
                iterations_used=result.iterations_used,
            ))
            return snapshot, {
                "task_id": task.task_id, "required_capability": task.required_capability,
                "reason": reason, "unsatisfied_criteria": result.unsatisfied_criteria,
                "suggested_capabilities": result.suggested_capabilities,
            }

        new_state = apply_patch(snapshot, result.state_patch)
        print(f"[Executor] Applied state patch: {list(result.state_patch.keys())}")
        status = "success" if result.goal_satisfied else "partial"
        records.append(TaskExecutionRecord(
            task_id=task.task_id, agent_id=spec.agent_id, status=status, goal=task.goal,
            state_before=state_summary(snapshot), state_patch=dict(result.state_patch),
            state_after=state_summary(new_state), reason=result.reason,
            iterations_used=result.iterations_used,
        ))
        history.record(task_id=task.task_id, agent_id=spec.agent_id, status=status, reason=result.reason)
        return new_state, None

    def _try_replan(
        self,
        user_instruction: str,
        state: VisualizationState,
        history: ExecutionHistory,
        failure_context: dict,
        counters: dict,
    ) -> Optional[VisualizationPlan]:
        if counters["replans"] >= self.max_replans:
            print(f"[Executor] Replan limit ({self.max_replans}) reached; stopping.")
            return None
        counters["replans"] += 1
        try:
            new_plan = self.replanner.replan(
                user_instruction, state, self.registry, history.summary(), failure_context
            )
        except PlannerError as exc:
            print(f"[Executor] Replanning failed: {exc}")
            return None
        try:
            validate_plan(new_plan, self.registry.known_capabilities())
        except PlanValidationError as exc:
            print(f"[Executor] Replanned plan failed validation: {exc}")
            return None
        return new_plan

    @staticmethod
    def _finish(
        success: bool,
        interpreted_goal: str,
        state: VisualizationState,
        records: List[TaskExecutionRecord],
        verification: Optional[FinalVerificationResult],
        counters: dict,
        diagnosis: str = "",
    ) -> ExecutionResult:
        return ExecutionResult(
            success=success,
            interpreted_goal=interpreted_goal,
            final_state=state,
            task_records=records,
            final_verification=verification,
            replans_used=counters["replans"],
            total_agent_calls=counters["agent_calls"],
            diagnosis=diagnosis,
        )
