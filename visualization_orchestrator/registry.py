"""Capability -> agent resolution. The planner only ever names a capability; this is the
single place that decides which concrete specialist actually handles it, so selection
logic (required-state checks, priority, cost, exclusions) never leaks into the executor
or the planner prompt.
"""
from typing import Dict, List, Optional, Set

from .capabilities import AgentSpec
from .state import VisualizationState


class AgentRegistry:
    def __init__(self):
        self._specialists: Dict[str, object] = {}
        self._specs: Dict[str, AgentSpec] = {}

    def register(self, specialist, spec: AgentSpec) -> None:
        if spec.agent_id in self._specs:
            raise ValueError(f"agent_id {spec.agent_id!r} is already registered")
        self._specialists[spec.agent_id] = specialist
        self._specs[spec.agent_id] = spec

    def get_spec(self, agent_id: str) -> AgentSpec:
        return self._specs[agent_id]

    def get_specialist(self, agent_id: str):
        return self._specialists[agent_id]

    def known_capabilities(self) -> Set[str]:
        return {cap.name for spec in self._specs.values() for cap in spec.capabilities}

    def capability_catalog(self) -> Dict[str, dict]:
        """capability_name -> {"description", "examples", "agents"} -- used to build the
        planner prompt dynamically so it never hard-codes which capabilities exist."""
        catalog: Dict[str, dict] = {}
        for spec in self._specs.values():
            for cap in spec.capabilities:
                entry = catalog.setdefault(
                    cap.name, {"description": cap.description, "examples": [], "agents": []}
                )
                for example in cap.examples:
                    if example not in entry["examples"]:
                        entry["examples"].append(example)
                entry["agents"].append(spec.agent_id)
        return catalog

    def find_candidates(
        self,
        capability_name: str,
        state: VisualizationState,
        constraints: Optional[dict] = None,
        excluded_agent_ids: Optional[Set[str]] = None,
    ) -> List[AgentSpec]:
        excluded = excluded_agent_ids or set()
        candidates = []
        for spec in self._specs.values():
            if spec.agent_id in excluded:
                continue
            if capability_name not in spec.capability_names():
                continue
            missing = [
                field_name
                for field_name in spec.required_state_fields
                if getattr(state, field_name, None) in (None, "")
            ]
            if missing:
                continue
            candidates.append(spec)
        return candidates

    def select_agent(
        self,
        capability_name: str,
        state: VisualizationState,
        constraints: Optional[dict] = None,
        preferred_agent_id: Optional[str] = None,
        excluded_agent_ids: Optional[Set[str]] = None,
    ) -> Optional[AgentSpec]:
        candidates = self.find_candidates(capability_name, state, constraints, excluded_agent_ids)
        if not candidates:
            return None
        if preferred_agent_id:
            preferred = [c for c in candidates if c.agent_id == preferred_agent_id]
            if preferred:
                return preferred[0]
        # Higher priority wins; ties broken by lower cost, then registration order (stable sort).
        return sorted(candidates, key=lambda spec: (-spec.priority, spec.cost))[0]
