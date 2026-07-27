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
    # True for specialists that apply an exact, already-fully-specified value with no
    # internal visual judgment of their own (see specialists/direct_adapter.py) -- the
    # executor trusts their own goal_satisfied report and skips the vision-LLM
    # FinalVerifier for a plan made up entirely of these. Re-running one of these (e.g.
    # after a false-negative verifier replan) applies its stated delta AGAIN rather than
    # converging toward a goal, so letting an unreliable vision check second-guess and
    # replan a deterministic action risks silently compounding it instead of fixing
    # anything -- see the roll-doubling bug this was introduced to prevent.
    deterministic: bool = False

    def capability_names(self) -> Set[str]:
        return {c.name for c in self.capabilities}
