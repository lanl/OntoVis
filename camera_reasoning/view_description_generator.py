"""
Generates short structured textual descriptions for existing reference-view images,
using the same AI model interface the camera-reasoning agent already uses
(camera_reasoning.chatgpt_client.ask_chatgpt) — no separate API integration.

Scope, deliberately narrow: this only reads reference_views_metadata.json and writes
view_descriptions.json. It does NOT touch reference_views.py (camera/image generation),
session.py/prompt_writer.py (the agent loop), and it does NOT implement any camera-
movement decision logic — purely descriptive, for later use.

Runs standalone:
  python camera_reasoning/view_description_generator.py \\
      --input reference_views/reference_views_metadata.json \\
      --output reference_views/view_descriptions.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_reasoning.chatgpt_client import ask_chatgpt  # noqa: E402  (needs sys.path fix above)

DEFAULT_METADATA_PATH = "reference_views/reference_views_metadata.json"
DEFAULT_OUTPUT_PATH = "reference_views/view_descriptions.json"

REQUIRED_FIELDS = ["view_name", "short_description"]

PROMPT_TEMPLATE = """You are describing a rendered image of foot bones from a specific camera view.

The reference view name is: {view_name}

Generate as detailed a description of this image as possible.

Return only valid JSON with the following fields:
- view_name
- short_description

Definitions:
- dorsal = top side of the foot
- plantar = bottom or sole side of the foot
- medial = big-toe side of the foot
- lateral = little-toe side of the foot
- distal = toe-end direction
- proximal = ankle or heel-end direction

Rules:
- The short_description should be a thorough, multi-sentence paragraph (not just one
  or two sentences). Cover: the overall viewing direction/angle, which named bone
  groups are visible (phalanges, metatarsals, tarsals, calcaneus, etc.) and roughly
  where each sits in the frame (top/bottom/left/right/center/foreground/background),
  the visible silhouette/contour and any arches or curves, and any structures that
  are foreshortened, occluded, or only partially visible from this angle.
- Do not include camera movement advice.
- Do not include extra explanation outside the JSON.
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
    """Return a list of validation problems; an empty list means it's valid."""
    problems = []
    for field in REQUIRED_FIELDS:
        if field not in description:
            problems.append(f"missing field {field!r}")

    if "short_description" in description and not isinstance(description["short_description"], str):
        problems.append("short_description must be a string")

    return problems


def generate_description_for_view(view_name: str, image_path: str, model: Optional[str] = None) -> dict:
    """Call the existing AI model interface on a single reference-view image and
    return its parsed, validated description dict.

    Raises ValueError on any failure (missing image, unparseable JSON, missing/
    malformed fields) so the batch driver can warn and skip without crashing.
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"image path does not exist: {image_path}")

    prompt = PROMPT_TEMPLATE.format(view_name=view_name)
    response_text = ask_chatgpt(prompt=prompt, screenshot_path=str(path), model=model)

    parsed = _extract_json_object(response_text)
    if parsed is None:
        raise ValueError(f"model response was not valid JSON: {response_text!r}")

    problems = _validate_description(parsed)
    if problems:
        raise ValueError(f"model output failed validation: {'; '.join(problems)}")

    # Trust the metadata's view_name over whatever the model echoed back.
    return {
        "view_name": view_name,
        "short_description": parsed["short_description"],
    }


def generate_view_descriptions(
    metadata_path: str = DEFAULT_METADATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    model: Optional[str] = None,
) -> List[dict]:
    """Read reference_views_metadata.json, generate a description for each view via
    the AI model, and write the results to output_path. Never modifies metadata_path.

    A single view failing (missing image, bad/unparseable model output, missing
    fields, ...) is logged as a warning and skipped — it does not stop processing
    of the remaining views.
    """
    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Reference view metadata not found at {metadata_file}. "
            "Generate reference views first (see camera_reasoning/reference_views.py)."
        )

    entries = json.loads(metadata_file.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of view entries in {metadata_file}, got {type(entries).__name__}")

    descriptions = []
    for entry in entries:
        view_name = entry.get("view_name")
        image_path = entry.get("image_path")

        if not view_name:
            print(f"[view_description_generator] WARNING: skipping entry with no view_name: {entry}")
            continue
        if not image_path:
            print(f"[view_description_generator] WARNING: skipping {view_name!r} — no image_path in metadata")
            continue

        print(f"[view_description_generator] Describing {view_name!r} ({image_path})...")
        try:
            description = generate_description_for_view(view_name, image_path, model=model)
        except ValueError as e:
            print(f"[view_description_generator] WARNING: skipping {view_name!r} — {e}")
            continue
        except Exception as e:
            print(f"[view_description_generator] WARNING: skipping {view_name!r} — unexpected error: {e}")
            continue

        descriptions.append(description)
        print(f"[view_description_generator]   -> {description['short_description']}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(descriptions, f, indent=2)

    print(f"[view_description_generator] Wrote {len(descriptions)}/{len(entries)} descriptions to {out_path}")
    return descriptions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate short structured descriptions for existing reference-view images."
    )
    parser.add_argument("--input", default=DEFAULT_METADATA_PATH, help="Path to reference_views_metadata.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to write view_descriptions.json")
    parser.add_argument(
        "--model", default=None,
        help="Override the AI model (defaults to the AI_MODEL env var / chatgpt_client default).",
    )
    return parser


def main(argv=None) -> List[dict]:
    args = _build_arg_parser().parse_args(argv)
    return generate_view_descriptions(metadata_path=args.input, output_path=args.output, model=args.model)


if __name__ == "__main__":
    main()
