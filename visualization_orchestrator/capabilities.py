"""Agent/capability specification schema.

Every specialist registers one AgentSpec describing what it can do (as a list of named
capabilities, e.g. "reveal_internal_structure"), what shared-state fields it owns, what
it needs to already be present in state to run at all, and selection metadata (cost,
priority). The planner only ever talks about capability names; AgentRegistry (registry.py)
is what maps a capability name to a concrete agent.
"""
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class AgentCapability:
    name: str
    description: str
    examples: List[str] = field(default_factory=list)


@dataclass
class AgentSpec:
    agent_id: str
    description: str
    capabilities: List[AgentCapability]
    owned_state_fields: Set[str]
    required_state_fields: Set[str]
    side_effects: List[str] = field(default_factory=list)
    cost: float = 1.0
    priority: int = 0

    def capability_names(self) -> Set[str]:
        return {c.name for c in self.capabilities}
