"""
Generate the 18-node camera-relative view bank + graph JSON + per-node
descriptions for a dataset.

Run with: .venv/bin/python examples/generate_camera_relative_views.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.camera_relative_views import generate_camera_relative_views
from camera_reasoning.camera_spatial_graph import CameraSpatialGraph
from camera_reasoning.graph_view_description_generator import generate_graph_view_descriptions

OUTPUT_DIR = "reference_views_relative"

# Short, plain-language description of what's actually being rendered. This is the
# ONLY object-specific wording in this script — it's passed to the LLM as context
# when generating each node's description (see graph_view_description_generator.py)
# so descriptions are accurate no matter which dataset this is pointed at below.
OBJECT_DESCRIPTION = "a 3D object of human skull"

records = generate_camera_relative_views(
    raw_path="data/skull_256x256x256_uint8.raw",
    dimensions=(256, 256, 256),
    scalar_type="uint8",
    isovalue=40,
    output_dir=OUTPUT_DIR,
)

graph = CameraSpatialGraph.from_camera_nodes(
    records,
    description=f"Camera-relative spatial graph of {OBJECT_DESCRIPTION} (18 neutral nodes, single-op edges).",
)
graph.validate()
graph.to_json(f"{OUTPUT_DIR}/camera_graph.json")
print(f"[generate_camera_relative_views] Graph valid: {len(graph.nodes)} nodes, {len(graph.edges)} edges.")
print(f"[generate_camera_relative_views] Wrote {OUTPUT_DIR}/camera_graph.json")

generate_graph_view_descriptions(
    nodes_path=f"{OUTPUT_DIR}/camera_nodes.json",
    output_path=f"{OUTPUT_DIR}/view_descriptions.json",
    object_description=OBJECT_DESCRIPTION,
)
