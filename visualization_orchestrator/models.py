"""Validated data models that cross the planner/executor/specialist boundaries.

The planner's raw LLM output is untrusted, so its schema (TaskConstraint/PlannedTask/
VisualizationPlan) is a pydantic model with strict field types -- malformed output raises
ValidationError instead of silently producing a broken plan. Everything downstream of
validated input (specialist results, execution records) is a plain dataclass.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .state import VisualizationState


class TaskConstraint(BaseModel):
    key: str
    value: Any


class PlannedTask(BaseModel):
    task_id: str
    goal: str
    required_capability: str
    dependencies: List[str] = Field(default_factory=list)
    success_criteria: List[str]
    constraints: List[TaskConstraint] = Field(default_factory=list)
    preferred_agent_id: Optional[str] = None
    max_attempts: int = 1


class VisualizationPlan(BaseModel):
    interpreted_goal: str
    tasks: List[PlannedTask]
    final_success_criteria: List[str]
    assumptions: List[str] = Field(default_factory=list)


@dataclass
class AgentExecutionResult:
    agent_id: str
    status: Literal["success", "failed", "partial"]
    goal_satisfied: bool
    state_patch: Dict[str, Any]

    confidence: float
    reason: str

    satisfied_criteria: List[str]
    unsatisfied_criteria: List[str]

    suggested_capabilities: List[str] = field(default_factory=list)
    failure_type: Optional[str] = None

    iterations_used: int = 0
    artifacts: List[str] = field(default_factory=list)


@dataclass
class FinalVerificationResult:
    success: bool
    confidence: float
    satisfied_criteria: List[str]
    unsatisfied_criteria: List[str]
    diagnosis: str
    suggested_capabilities: List[str] = field(default_factory=list)


@dataclass
class TaskExecutionRecord:
    task_id: str
    agent_id: Optional[str]
    status: str
    goal: str
    state_before: dict
    state_patch: dict
    state_after: dict
    reason: str
    iterations_used: int = 0


@dataclass
class ExecutionResult:
    success: bool
    interpreted_goal: str
    final_state: VisualizationState
    task_records: List[TaskExecutionRecord]
    final_verification: Optional[FinalVerificationResult]
    replans_used: int
    total_agent_calls: int
    diagnosis: str = ""
