"""
Generic CLI entry point for the medical reference-view pipeline described in
README.md. Reads a dataset config.json, generates (or loads) one frozen
landmark profile, generates rotated reference images, diagnoses every image
against that same profile, and writes reference_views.json.

The engine itself lives in camera_reasoning/medical_reference_views.py and
must stay object-agnostic; all object-specific data comes from --config.

Run with:
  .venv/bin/python examples/generate_medical_reference_views.py \\
      --config reference_views_medical/skull/config.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_reasoning.medical_reference_views import (  # noqa: E402  (needs sys.path fix above)
    generate_reference_views,
    load_dataset_config,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rotated reference images and structured landmark descriptions for a medical dataset."
    )
    parser.add_argument("--config", required=True, help="Path to the dataset config.json")
    parser.add_argument("--model", default=None, help="Override the AI model (defaults to AI_MODEL / chatgpt_client default)")
    parser.add_argument("--force-profile", action="store_true", help="Regenerate the landmark profile even if one is already frozen")
    parser.add_argument("--force-descriptions", action="store_true", help="Regenerate descriptions even if a cached signature is still valid")
    parser.add_argument("--skip-rotations", action="store_true", help="Reuse existing rotated images instead of regenerating them")
    parser.add_argument("--describe-only", action="store_true", help="Diagnose existing images only; do not generate rotations")
    return parser


def main(argv=None) -> dict:
    args = _build_arg_parser().parse_args(argv)
    config = load_dataset_config(args.config)
    return generate_reference_views(
        config,
        model=args.model,
        force_profile=args.force_profile,
        force_descriptions=args.force_descriptions,
        skip_rotations=args.skip_rotations,
        describe_only=args.describe_only,
    )


if __name__ == "__main__":
    main()
