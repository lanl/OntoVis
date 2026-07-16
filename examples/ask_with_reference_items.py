"""
Demo of ask_chatgpt's `reference_items` parameter: sends the graph-relative
reference bank (reference_views_relative/) to the LLM with each reference image
placed immediately followed by its own description — image, description, image,
description, ... — instead of all images first and all text after.

By default only the first REFERENCE_LIMIT nodes are sent (keeps the request small
for a quick smoke test); set REFERENCE_LIMIT = None to send the full 18-node bank.

Run with: .venv/bin/python examples/ask_with_reference_items.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning.chatgpt_client import ask_chatgpt

NODES_PATH = "reference_views_relative/camera_nodes.json"
DESCRIPTIONS_PATH = "reference_views_relative/view_descriptions.json"
QUERY_IMAGE_PATH = "reference_views_relative/view_005.png"  # any current/query screenshot

REFERENCE_LIMIT = 3  # set to None to send all 18 reference items

nodes = json.loads(Path(NODES_PATH).read_text())
descriptions = {d["node_id"]: d["description"] for d in json.loads(Path(DESCRIPTIONS_PATH).read_text())}

reference_items = [
    (n["node_id"], n["image_path"], descriptions[n["node_id"]])
    for n in nodes
    if n["node_id"] in descriptions
]
if REFERENCE_LIMIT is not None:
    reference_items = reference_items[:REFERENCE_LIMIT]

print(f"Sending {len(reference_items)} reference items (image + description pairs): "
      f"{[label for label, _, _ in reference_items]}")

prompt = (
    "Below is a query image, followed by several labeled reference images. Each "
    "reference image is immediately followed by its own description.\n\n"
    "Task: identify which single reference label is most visually similar to the "
    "query image, using both the images and their descriptions.\n\n"
    "Return only valid JSON: {\"most_similar_label\": \"...\", \"reasoning\": \"...\"}"
)

response = ask_chatgpt(prompt=prompt, screenshot_path=QUERY_IMAGE_PATH, reference_items=reference_items)
print("\nResponse:\n" + response)
