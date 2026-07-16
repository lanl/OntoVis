"""
Tests for the camera-relative spatial graph (camera_spatial_graph.py,
camera_relative_views.py, image_view_matcher.GraphImageMatcher).

No pytest required — plain asserts, run directly (from the repo root, so the
reference_views_relative/ paths below resolve):
  .venv/bin/python tests/test_camera_spatial_graph.py

(Also pytest-collectible, since every test is a module-level `test_*` function.)

Prerequisite: reference_views_relative/camera_graph.json must already exist —
generate it with examples/generate_camera_relative_views.py.
"""
import re
import sys
from pathlib import Path

import numpy as np
import vtk

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.camera_spatial_graph import CameraSpatialGraph, VALID_ACTION_TYPES
from camera_reasoning.image_view_matcher import GraphImageMatcher

GRAPH_PATH = "reference_views_relative/camera_graph.json"
EMBEDDING_CACHE_PATH = "reference_views_relative/graph_embeddings.pkl"
NODE_ID_PATTERN = re.compile(r"^view_\d{3}$")
SEMANTIC_LABELS = ["dorsal", "plantar", "medial", "lateral", "distal", "proximal"]
REPLAY_TOLERANCE = 1e-2


def _load_graph() -> CameraSpatialGraph:
    graph = CameraSpatialGraph.from_json(GRAPH_PATH)
    graph.validate()
    return graph


def test_graph_has_exactly_18_nodes():
    graph = _load_graph()
    assert len(graph.nodes) == 18, f"expected 18 nodes, found {len(graph.nodes)}"
    print("test_graph_has_exactly_18_nodes: OK")


def test_graph_is_connected():
    graph = _load_graph()
    assert graph.is_connected(), "graph must be (weakly) connected"
    print("test_graph_is_connected: OK")


def test_every_forward_edge_has_a_valid_inverse_edge():
    graph = _load_graph()

    def action_key(action):
        return (action["type"], round(float(action["degrees"]), 6))

    edge_keys = {(e["source"], e["target"], action_key(e["action"])) for e in graph.edges}
    for e in graph.edges:
        inverse = (e["target"], e["source"], action_key({"type": e["action"]["type"], "degrees": -e["action"]["degrees"]}))
        assert inverse in edge_keys, f"edge {e['source']}->{e['target']} ({e['action']}) has no inverse edge"
    print("test_every_forward_edge_has_a_valid_inverse_edge: OK")


def test_every_edge_action_is_a_single_supported_camera_operation():
    graph = _load_graph()
    for e in graph.edges:
        action = e["action"]
        assert action["type"] in VALID_ACTION_TYPES, f"unsupported action type: {action['type']!r}"
        assert isinstance(action["degrees"], (int, float)), f"non-numeric degrees: {action['degrees']!r}"
    print("test_every_edge_action_is_a_single_supported_camera_operation: OK")


def test_every_node_has_image_and_camera_metadata():
    graph = _load_graph()
    required_camera_fields = ["position", "focal_point", "view_up", "direction_of_projection", "clipping_range"]
    for node_id, data in graph.nodes.items():
        assert "image_path" in data, f"node {node_id} missing image_path"
        assert Path(data["image_path"]).exists(), f"node {node_id} image_path does not exist: {data['image_path']}"
        camera = data.get("camera")
        assert isinstance(camera, dict), f"node {node_id} missing camera metadata"
        for field in required_camera_fields:
            assert field in camera, f"node {node_id} camera metadata missing {field!r}"
    print("test_every_node_has_image_and_camera_metadata: OK")


def test_shortest_path_from_a_node_to_itself_is_zero_actions():
    graph = _load_graph()
    node_id = graph.root_node_id
    route = graph.route(node_id, node_id)
    assert route["node_path"] == [node_id]
    assert route["actions"] == []
    assert route["total_cost"] == 0.0
    print("test_shortest_path_from_a_node_to_itself_is_zero_actions: OK")


def test_every_node_reachable_from_view_000():
    graph = _load_graph()
    assert graph.root_node_id == "view_000"
    for node_id in graph.nodes:
        path = graph.shortest_path("view_000", node_id)
        assert path[0] == "view_000" and path[-1] == node_id
    print("test_every_node_reachable_from_view_000: OK")


def test_replaying_a_direct_edge_reaches_target_camera_state():
    graph = _load_graph()
    checked = 0
    for e in graph.edges:
        source_camera = graph.nodes[e["source"]]["camera"]
        target_camera = graph.nodes[e["target"]]["camera"]

        cam = vtk.vtkCamera()
        cam.SetPosition(*source_camera["position"])
        cam.SetFocalPoint(*source_camera["focal_point"])
        cam.SetViewUp(*source_camera["view_up"])

        action = e["action"]
        if action["type"] == "azimuth":
            cam.Azimuth(action["degrees"])
        else:
            cam.Elevation(action["degrees"])
        cam.OrthogonalizeViewUp()

        pos_err = np.linalg.norm(np.array(cam.GetPosition()) - np.array(target_camera["position"]))
        fp_err = np.linalg.norm(np.array(cam.GetFocalPoint()) - np.array(target_camera["focal_point"]))
        up_err = np.linalg.norm(np.array(cam.GetViewUp()) - np.array(target_camera["view_up"]))
        assert max(pos_err, fp_err, up_err) <= REPLAY_TOLERANCE, (
            f"edge {e['source']}->{e['target']} ({action}) replay error exceeds tolerance: "
            f"pos={pos_err:.6f} fp={fp_err:.6f} up={up_err:.6f}"
        )
        checked += 1
    assert checked == len(graph.edges)
    print(f"test_replaying_a_direct_edge_reaches_target_camera_state: OK ({checked} edges checked)")


def test_known_reference_image_matches_its_own_node():
    graph = _load_graph()
    matcher = GraphImageMatcher(graph, cache_path=EMBEDDING_CACHE_PATH)
    matcher.build_index()

    probe_node_id = "view_009"
    probe_image_path = graph.nodes[probe_node_id]["image_path"]
    result = matcher.match(probe_image_path, top_k=3)
    assert result["best_node_id"] == probe_node_id, (
        f"expected {probe_node_id}'s own image to match itself, got {result['best_node_id']!r}"
    )
    assert result["top_matches"][0]["similarity"] >= 0.999
    print("test_known_reference_image_matches_its_own_node: OK")


def test_no_semantic_orientation_labels_anywhere():
    graph = _load_graph()
    for node_id in graph.nodes:
        assert NODE_ID_PATTERN.match(node_id), f"node_id {node_id!r} is not neutrally named (view_NNN)"
        lowered = node_id.lower()
        for label in SEMANTIC_LABELS:
            assert label not in lowered, f"node_id {node_id!r} contains semantic label {label!r}"

    description_text = (graph.description or "").lower()
    for label in SEMANTIC_LABELS:
        assert label not in description_text, f"graph description contains semantic label {label!r}"
    print("test_no_semantic_orientation_labels_anywhere: OK")


if __name__ == "__main__":
    test_graph_has_exactly_18_nodes()
    test_graph_is_connected()
    test_every_forward_edge_has_a_valid_inverse_edge()
    test_every_edge_action_is_a_single_supported_camera_operation()
    test_every_node_has_image_and_camera_metadata()
    test_shortest_path_from_a_node_to_itself_is_zero_actions()
    test_every_node_reachable_from_view_000()
    test_replaying_a_direct_edge_reaches_target_camera_state()
    test_known_reference_image_matches_its_own_node()
    test_no_semantic_orientation_labels_anywhere()
    print("\nAll tests passed.")
