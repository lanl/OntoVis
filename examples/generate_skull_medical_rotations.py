"""
Compatibility wrapper for the skull dataset.

The rotation/diagnosis engine is now generic (see
camera_reasoning/medical_reference_views.py and README.md) and contains no
skull-specific assumptions. All skull-specific data -- object context, seed
images, view labels, rotation angles -- lives in
reference_views_medical/skull/config.json, not in this file.

This script is kept only so existing tooling/notebooks that call it directly
keep working; it just loads the skull config and invokes the generic
pipeline.

Run with: .venv/bin/python examples/generate_skull_medical_rotations.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_reasoning.medical_reference_views import (  # noqa: E402  (needs sys.path fix above)
    generate_reference_views,
    load_dataset_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "reference_views_medical/skull/config.json"


def generate_skull_rotations() -> dict:
    config = load_dataset_config(CONFIG_PATH)
    return generate_reference_views(config)


if __name__ == "__main__":
    generate_skull_rotations()
