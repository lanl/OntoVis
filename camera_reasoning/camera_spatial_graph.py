"""
CameraSpatialGraph: a camera-relative spatial graph of reference-view nodes.

Nodes are neutrally-named image/camera-state snapshots (view_000, view_001, ...
— no anatomical or directional labels). Edges each correspond to exactly one
elementary VTK camera action (Azimuth or Elevation by some number of degrees)
that was actually applied when the node was generated (see
camera_relative_views.py). Every forward edge has a matching inverse edge
(negated degrees), so any node-to-node route can be replayed in either
direction.

Backed by a networkx.DiGraph internally. Typical usage:

    graph = CameraSpatialGraph.from_json("reference_views_relative/camera_graph.json")
    graph.validate()
    route = graph.route("view_003", "view_014")
    # route["actions"] -> [{"type": "elevation", "degrees": 45.0}]  (single hop)
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

VALID_ACTION_TYPES = {"azimuth", "elevation"}
REQUIRED_CAMERA_FIELDS = ["position", "focal_point", "view_up", "direction_of_projection", "clipping_range"]
EXPECTED_NODE_COUNT = 18


def _invert_action(action: dict) -> dict:
    return {"type": action["type"], "degrees": -float(action["degrees"])}


def _action_key(action: dict, digits: int = 6):
    return (action.get("type"), round(float(action.get("degrees", 0.0)), digits))


class CameraSpatialGraph:
    """A directed graph of camera-relative reference-view nodes.

    Construction is normally via the classmethods `from_json` (load a
    pre-built camera_graph.json) or `from_camera_nodes` (build one, with
    auto-derived inverse edges, from camera_relative_views.py's flat
    camera_nodes.json records).
    """

    def __init__(
        self,
        nodes_list: List[dict],
        edges: List[dict],
        root_node_id: str,
        version: int = 1,
        description: str = "",
    ):
        self.version = version
        self.description = description
        self.root_node_id = root_node_id
        self.edges: List[dict] = [dict(e) for e in edges]

        self.duplicate_node_ids: List[str] = []
        self.nodes: Dict[str, dict] = {}
        seen = set()
        for n in nodes_list:
            node_id = n["node_id"]
            if node_id in seen:
                self.duplicate_node_ids.append(node_id)
                continue
            seen.add(node_id)
            self.nodes[node_id] = {k: v for k, v in n.items() if k != "node_id"}

        self._graph = self._build_networkx()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for node_id, data in self.nodes.items():
            g.add_node(node_id, **data)
        for edge in self.edges:
            if edge["source"] not in self.nodes or edge["target"] not in self.nodes:
                # Kept out of the traversable graph, but still reported by validate().
                continue
            g.add_edge(edge["source"], edge["target"], action=edge.get("action"), cost=edge.get("cost", 1.0))
        return g

    @classmethod
    def from_json(cls, path: str) -> "CameraSpatialGraph":
        """Load a full graph JSON: {version, description, root_node_id, nodes, edges}."""
        data = json.loads(Path(path).read_text())
        return cls(
            nodes_list=data["nodes"],
            edges=data["edges"],
            root_node_id=data["root_node_id"],
            version=data.get("version", 1),
            description=data.get("description", ""),
        )

    @classmethod
    def from_camera_nodes(cls, camera_nodes: List[dict], description: str = "") -> "CameraSpatialGraph":
        """Build a graph from camera_relative_views.py's flat per-node records
        ([{"node_id","parent_id","action","image_path","camera"}, ...]),
        auto-deriving both the forward edge (parent -> child, the action that
        was actually applied) and its inverse edge (child -> parent, the
        negated action) for every non-root node.
        """
        nodes_list = [
            {"node_id": n["node_id"], "image_path": n["image_path"], "camera": n["camera"]}
            for n in camera_nodes
        ]

        roots = [n["node_id"] for n in camera_nodes if n.get("parent_id") is None]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root node (parent_id=None), found {roots}")
        root_node_id = roots[0]

        edges: List[dict] = []
        for n in camera_nodes:
            if n.get("parent_id") is None:
                continue
            action = n["action"]
            edges.append({"source": n["parent_id"], "target": n["node_id"], "action": dict(action), "cost": 1.0})
            edges.append({"source": n["node_id"], "target": n["parent_id"], "action": _invert_action(action), "cost": 1.0})

        return cls(nodes_list=nodes_list, edges=edges, root_node_id=root_node_id, description=description)

    @classmethod
    def from_camera_nodes_json(cls, path: str, description: str = "") -> "CameraSpatialGraph":
        camera_nodes = json.loads(Path(path).read_text())
        return cls.from_camera_nodes(camera_nodes, description=description)

    def to_json(self, path: str) -> None:
        payload = {
            "version": self.version,
            "description": self.description,
            "root_node_id": self.root_node_id,
            "nodes": [{"node_id": node_id, **data} for node_id, data in self.nodes.items()],
            "edges": self.edges,
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))

    # kept as an alias — "save" reads naturally at call sites, "to_json" states the format.
    save = to_json

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError (listing every problem found) if the graph is not
        well-formed. Checks: duplicate node IDs, missing source/target nodes on
        edges, invalid action types, missing inverse edges, disconnected nodes,
        malformed camera metadata, and node count != 18.
        """
        problems: List[str] = []

        if self.duplicate_node_ids:
            problems.append(f"duplicate node IDs: {sorted(set(self.duplicate_node_ids))}")

        if len(self.nodes) != EXPECTED_NODE_COUNT:
            problems.append(f"graph must contain exactly {EXPECTED_NODE_COUNT} nodes, found {len(self.nodes)}")

        node_ids = set(self.nodes.keys())

        for edge in self.edges:
            source, target = edge.get("source"), edge.get("target")
            if source not in node_ids:
                problems.append(f"edge references missing source node {source!r}")
            if target not in node_ids:
                problems.append(f"edge references missing target node {target!r}")
            action = edge.get("action") or {}
            if action.get("type") not in VALID_ACTION_TYPES:
                problems.append(f"edge {source!r}->{target!r} has invalid action type {action.get('type')!r}")
            elif "degrees" not in action:
                problems.append(f"edge {source!r}->{target!r} action is missing 'degrees'")

        valid_edges = [
            e for e in self.edges
            if e.get("source") in node_ids and e.get("target") in node_ids and e.get("action")
            and e["action"].get("type") in VALID_ACTION_TYPES
        ]
        edge_key_set = {(e["source"], e["target"], _action_key(e["action"])) for e in valid_edges}
        for e in valid_edges:
            inverse_key = (e["target"], e["source"], _action_key(_invert_action(e["action"])))
            if inverse_key not in edge_key_set:
                problems.append(
                    f"edge {e['source']!r}->{e['target']!r} (action={e['action']}) has no matching inverse edge"
                )

        if self.root_node_id not in node_ids:
            problems.append(f"root_node_id {self.root_node_id!r} is not a node in the graph")
        else:
            reachable = set(nx.dfs_preorder_nodes(self._graph.to_undirected(), self.root_node_id))
            unreachable = node_ids - reachable
            if unreachable:
                problems.append(f"nodes not reachable from root {self.root_node_id!r}: {sorted(unreachable)}")

        for node_id, data in self.nodes.items():
            if "image_path" not in data:
                problems.append(f"node {node_id!r} is missing image_path")
            camera = data.get("camera")
            if not isinstance(camera, dict):
                problems.append(f"node {node_id!r} is missing camera metadata")
                continue
            for field in REQUIRED_CAMERA_FIELDS:
                if field not in camera:
                    problems.append(f"node {node_id!r} camera metadata missing {field!r}")

        if problems:
            raise ValueError("CameraSpatialGraph validation failed:\n" + "\n".join(f"  - {p}" for p in problems))

    def is_connected(self) -> bool:
        if not self.nodes:
            return True
        return nx.is_weakly_connected(self._graph)

    # ------------------------------------------------------------------
    # Queries / routing
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> List[str]:
        if node_id not in self.nodes:
            raise ValueError(f"unknown node_id {node_id!r}")
        return list(self._graph.successors(node_id))

    def shortest_path(self, source_node_id: str, target_node_id: str) -> List[str]:
        if source_node_id not in self.nodes:
            raise ValueError(f"unknown source_node_id {source_node_id!r}")
        if target_node_id not in self.nodes:
            raise ValueError(f"unknown target_node_id {target_node_id!r}")
        if source_node_id == target_node_id:
            return [source_node_id]
        try:
            return nx.shortest_path(self._graph, source_node_id, target_node_id, weight="cost")
        except nx.NetworkXNoPath:
            raise ValueError(f"no path from {source_node_id!r} to {target_node_id!r}")

    def actions_for_path(self, path: List[str]) -> List[dict]:
        actions = []
        for u, v in zip(path, path[1:]):
            edge_data = self._graph.get_edge_data(u, v)
            if edge_data is None:
                raise ValueError(f"no edge from {u!r} to {v!r}")
            actions.append(dict(edge_data["action"]))
        return actions

    def route(self, source_node_id: str, target_node_id: str) -> dict:
        """Deterministic node-to-node route: shortest node path + the exact
        sequence of single-op camera actions to replay it."""
        path = self.shortest_path(source_node_id, target_node_id)
        actions = self.actions_for_path(path)
        total_cost = sum(
            self._graph.get_edge_data(u, v)["cost"] for u, v in zip(path, path[1:])
        ) if len(path) > 1 else 0.0
        return {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "node_path": path,
            "actions": actions,
            "total_cost": total_cost,
        }
