"""
Generates detailed textual descriptions for the camera-relative spatial graph's
reference images (reference_views_relative/view_000.png .. view_017.png), using
the same AI model interface the camera-reasoning agent already uses
(camera_reasoning.chatgpt_client.ask_chatgpt — the same model configured via the
AI_MODEL/AI_URL env vars used by manual_chatgpt_loop.ipynb).

This is the graph-native counterpart to view_description_generator.py. It is
intentionally a separate module rather than a mode flag on that one, because the
prompt itself must differ in a way that matters: the graph's nodes are neutral
(view_000, view_001, ...) with no anatomical/directional identity, so the prompt
must NOT ask the model to name or infer a viewing direction (dorsal/plantar/
medial/lateral/distal/proximal) — only to describe what is visibly in the frame.
Mentioning anatomical structures themselves (metatarsals, phalanges, ...) is fine;
labeling the *camera direction* in those terms is not, since node identity here is
purely geometric (node_id + camera state), not semantic.

Scope, deliberately narrow: reads camera_nodes.json and writes
view_descriptions.json alongside it. Does not touch camera_relative_views.py,
camera_spatial_graph.py, or the semantic reference_views/ pipeline.

Runs standalone:
  python camera_reasoning/graph_view_description_generator.py \\
      --input reference_views_relative/camera_nodes.json \\
      --output reference_views_relative/view_descriptions.json \\
      --object-description "a human skull"
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_reasoning.chatgpt_client import ask_chatgpt  # noqa: E402  (needs sys.path fix above)

DEFAULT_NODES_PATH = "reference_views_relative/camera_nodes.json"
DEFAULT_OUTPUT_PATH = "reference_views_relative/view_descriptions.json"

REQUIRED_FIELDS = ["node_id", "description"]

# Fallback used only if no object_description is given — deliberately generic
# (this module has no default object of its own; see camera_relative_views.py).
DEFAULT_OBJECT_CONTEXT = "a 3D object"

PROMPT_TEMPLATE = """
You are analyzing a rendered image of {object_context} captured from a specific
camera position. The image's internal identifier is: {node_id}

Your goal is to create a visually grounded, view-specific description that can
be used to distinguish this camera view from other views of the same object.

Describe only information that is visibly supported by the image. Do not rely
on the node identifier, expected camera pose, or assumptions about how the
object is normally oriented.

Return only one valid JSON object using exactly this structure:

{{
  "node_id": "{node_id}",
  "description": "A concise but detailed summary of the visible view.",
  "visible_landmarks": [
    {{
      "name": "landmark or visually distinctive component",
      "screen_position": "top, bottom, left, right, center, or a combination",
      "depth_position": "foreground, middle-ground, background, or unclear",
      "visibility": "dominant, clear, partial, heavily occluded, or uncertain",
      "visual_evidence": "brief description of the visible evidence"
    }}
  ],
  "image_orientation": {{
    "dominant_object_axis": "direction of the object's main visible axis in image coordinates, or unclear",
    "screen_top": "object regions or landmarks visible near the top",
    "screen_bottom": "object regions or landmarks visible near the bottom",
    "screen_left": "object regions or landmarks visible on the left",
    "screen_right": "object regions or landmarks visible on the right",
    "roll_cues": "visible evidence that the object appears upright, tilted, rotated in-plane, or unclear",
    "confidence": 0.0
  }},
  "spatial_relationships": [
    "A concise relationship between visible landmarks, such as one structure being above, beside, in front of, behind, or partially covering another."
  ],
  "silhouette_and_depth": {{
    "silhouette": "overall visible contour and major protrusions, recesses, arches, curves, or flat regions",
    "foreshortening": "which components appear compressed, elongated, end-on, or unaffected",
    "occlusion": "important components that are partially hidden or overlap one another",
    "depth_summary": "how the visible structures are arranged from foreground to background"
  }},
  "distinctive_view_features": [
    "The most discriminative visible features that separate this view from other possible views."
  ],
  "uncertainty": {{
    "ambiguous_landmarks": [
      "structures whose identity or orientation cannot be determined confidently"
    ],
    "overall_confidence": 0.0
  }}
}}

Rules:

1. Use image-relative coordinates only:
   - screen top
   - screen bottom
   - screen left
   - screen right
   - center
   - foreground
   - background

2. Do not infer a canonical, anatomical, geographic, or object-relative viewing
   direction unless it is unambiguously visible and explicitly supported by
   {object_context}.

3. Do not use labels such as front view, rear view, top view, bottom view,
   left-side view, or right-side view merely from assumptions about the camera.

4. Do not infer hidden structures or properties that are not visible.

5. Identify landmarks using the most specific visually supported name.
   If the identity is uncertain, use a neutral visual description such as:
   - large rounded region
   - elongated component
   - circular opening
   - thin protrusion
   - clustered small components
   - curved outer boundary

6. Do not force every structure to have a semantic name. A correct neutral
   description is preferable to an incorrect specific identification.

7. Prioritize stable visual landmarks and their spatial arrangement over:
   - surface texture
   - lighting differences
   - minor rendering noise
   - detached artifacts
   - speculative damage or degradation

8. Mention rendering artifacts only when they significantly obscure or alter
   the visible silhouette.

9. The description must emphasize what makes this view visually distinctive.
   Avoid generic statements that would apply equally to many views of the
   object.

10. Keep the main description focused. It should normally contain between
    three and six sentences rather than an unnecessarily long paragraph.

11. The image_orientation confidence and overall_confidence values must be
    numbers from 0.0 to 1.0.

12. Do not provide camera movement recommendations or instructions.

13. Do not include markdown, commentary, or any text outside the JSON object.
"""

def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a model response: a bare
    object, one wrapped in a ```json fenced code block, or one embedded in
    surrounding prose. Returns None if nothing parseable is found.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _validate_description(description: dict) -> List[str]:
    problems = []
    for field in REQUIRED_FIELDS:
        if field not in description:
            problems.append(f"missing field {field!r}")
    if "description" in description and not isinstance(description["description"], str):
        problems.append("description must be a string")
    return problems


def generate_description_for_node(
    node_id: str,
    image_path: str,
    object_description: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Call the existing AI model interface on a single graph-node image and
    return its parsed, validated description dict.

    `object_description` is a short, plain-language description of what the
    rendered object actually is (e.g. "a human skull", "a foot bone structure")
    — given to the model as context so it describes the image correctly
    regardless of which dataset this pipeline is pointed at. Falls back to a
    generic DEFAULT_OBJECT_CONTEXT if not given.

    Raises ValueError on any failure (missing image, unparseable JSON, missing/
    malformed fields) so the batch driver can warn and skip without crashing.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"image path does not exist: {image_path}")

    prompt = PROMPT_TEMPLATE.format(
        node_id=node_id,
        object_context=object_description or DEFAULT_OBJECT_CONTEXT,
    )
    response_text = ask_chatgpt(prompt=prompt, screenshot_path=str(path), model=model)

    parsed = _extract_json_object(response_text)
    if parsed is None:
        raise ValueError(f"model response was not valid JSON: {response_text!r}")

    problems = _validate_description(parsed)
    if problems:
        raise ValueError(f"model output failed validation: {'; '.join(problems)}")

    # Trust the input node_id over whatever the model echoed back.
    return {
        "node_id": node_id,
        "description": parsed["description"],
    }


def generate_graph_view_descriptions(
    nodes_path: str = DEFAULT_NODES_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    object_description: Optional[str] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """Read camera_nodes.json, generate a detailed description for each node via
    the AI model, and write the results to output_path. Never modifies nodes_path.

    `object_description` (e.g. "a human skull") is passed through to every call —
    see generate_description_for_node() for why this matters.

    A single node failing (missing image, bad/unparseable model output, missing
    fields, ...) is logged as a warning and skipped — it does not stop processing
    of the remaining nodes.
    """
    nodes_file = Path(nodes_path)
    if not nodes_file.exists():
        raise FileNotFoundError(
            f"Graph node records not found at {nodes_file}. "
            "Generate the camera-relative view bank first (see "
            "camera_reasoning/camera_relative_views.py)."
        )

    entries = json.loads(nodes_file.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of node entries in {nodes_file}, got {type(entries).__name__}")

    descriptions = []
    for entry in entries:
        node_id = entry.get("node_id")
        image_path = entry.get("image_path")

        if not node_id:
            print(f"[graph_view_description_generator] WARNING: skipping entry with no node_id: {entry}")
            continue
        if not image_path:
            print(f"[graph_view_description_generator] WARNING: skipping {node_id!r} — no image_path")
            continue

        print(f"[graph_view_description_generator] Describing {node_id!r} ({image_path})...")
        try:
            description = generate_description_for_node(
                node_id, image_path, object_description=object_description, model=model
            )
        except ValueError as e:
            print(f"[graph_view_description_generator] WARNING: skipping {node_id!r} — {e}")
            continue
        except Exception as e:
            print(f"[graph_view_description_generator] WARNING: skipping {node_id!r} — unexpected error: {e}")
            continue

        descriptions.append(description)
        print(f"[graph_view_description_generator]   -> {description['description'][:120]}...")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(descriptions, f, indent=2)

    print(f"[graph_view_description_generator] Wrote {len(descriptions)}/{len(entries)} descriptions to {out_path}")
    return descriptions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate detailed, neutral (non-anatomical-direction) descriptions for camera-relative graph node images."
    )
    parser.add_argument("--input", default=DEFAULT_NODES_PATH, help="Path to camera_nodes.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to write view_descriptions.json")
    parser.add_argument(
        "--object-description", default=None,
        help="Short description of what object is being rendered (e.g. 'a human skull'), "
             "given to the model as context so it describes the images correctly regardless "
             "of which dataset this is pointed at.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override the AI model (defaults to the AI_MODEL env var / chatgpt_client default).",
    )
    return parser


def main(argv=None) -> List[dict]:
    args = _build_arg_parser().parse_args(argv)
    return generate_graph_view_descriptions(
        nodes_path=args.input,
        output_path=args.output,
        object_description=args.object_description,
        model=args.model,
    )


if __name__ == "__main__":
    main()
