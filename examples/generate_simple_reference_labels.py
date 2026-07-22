"""Write a simple {node_id: label} reference-view mapping for an already-
generated medical reference-view dataset (reuses existing photos, no LLM
landmark diagnosis call).

Alternative to examples/generate_medical_reference_views.py, which is left
completely untouched -- run that one instead whenever you want the full
structured landmark/relations/description_sentences schema back.

Run with:
  .venv/bin/python examples/generate_simple_reference_labels.py
  .venv/bin/python examples/generate_simple_reference_labels.py \\
      --config reference_views_medical/skull/config.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_reasoning.medical_reference_views import load_dataset_config  # noqa: E402
from camera_reasoning.simple_reference_labels import write_simple_reference_labels  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "reference_views_medical" / "skull" / "config.json"

# Edit to match your dataset's seed view_label values (see config.json "seeds").
BASE_LABELS = {
    "front": "front",
    "right": "right",
    "left": "left",
}

# Per-degree override of the rotation direction word, in case a specific
# rotation was actually generated counterclockwise instead of the pipeline's
# default (see DEFAULT_ROTATION_DIRECTION in camera_reasoning/simple_reference_labels.py,
# which matches medical_reference_views.py's im.rotate(-degrees, ...) call).
# Leave empty to use the default ("clockwise") for every rotation.
ROTATION_DIRECTIONS = {}


def main():
    parser = argparse.ArgumentParser(
        description="Write a simple node_id -> label reference-view mapping."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to the dataset config.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults to <dataset_directory>/reference_views_simple.json)",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    output_path = Path(args.output) if args.output else None
    written = write_simple_reference_labels(config, BASE_LABELS, ROTATION_DIRECTIONS, output_path)
    print(f"Wrote simple reference labels to {written}")


if __name__ == "__main__":
    main()
