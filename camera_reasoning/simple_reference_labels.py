"""Simple, non-LLM reference-view labeling: reference image -> one sentence.

Alternative to the full landmark-based pipeline in medical_reference_views.py
(kept completely untouched by this module -- use that one when you need
structured landmarks/relations/description_sentences). This one just maps
each already-generated reference image to a short human-written sentence
like "Left view of the human skull. Rotated 45 degrees clockwise.", for
cases where that's all you need and you don't want to pay for an LLM
landmark-diagnosis call.

Reuses the same dataset config.json (seeds + rotations) and the same
already-rendered images on disk -- it does not regenerate or move anything.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from camera_reasoning.medical_reference_views import DatasetConfig

# generate_rotated_images() in medical_reference_views.py does
# im.rotate(-degrees, ...) -- PIL rotates counter-clockwise for positive
# angles, so negating makes the saved image rotated CLOCKWISE by `degrees`.
# Keep this in sync with that function; it's the ground truth for direction.
DEFAULT_ROTATION_DIRECTION = "clockwise"


def build_simple_labels(
    config: DatasetConfig,
    base_labels: Dict[str, str],
    rotation_directions: Optional[Dict[int, str]] = None,
    default_rotation_direction: str = DEFAULT_ROTATION_DIRECTION,
) -> Dict[str, str]:
    """Map every already-generated node_id to a one-sentence description.

    base_labels: seed.view_label (from config.json's "seeds", e.g. "front",
    "right", "left") -> the term to use for that seed's un-rotated image
    (e.g. "front", "right", "left"). Produces sentences like:
        "Front view of the human skull."
        "Left view of the human skull. Rotated 45 degrees clockwise."

    rotation_directions: optional per-degree override of the direction word
    (e.g. {45: "counterclockwise"}) in case a specific rotation was generated
    differently from the rest. Defaults to default_rotation_direction for
    every entry in config.rotations.
    """
    rotation_directions = rotation_directions or {}
    labels: Dict[str, str] = {}
    for seed in config.seeds:
        base_term = base_labels.get(seed.view_label, seed.view_label)
        base_sentence = f"{base_term.capitalize()} view of the {config.object_context}."
        labels[seed.node_id] = base_sentence

        for degrees in config.rotations:
            node_id = f"{seed.node_id}_rot{degrees:03d}"
            direction = rotation_directions.get(degrees, default_rotation_direction)
            labels[node_id] = f"{base_sentence} Rotated {degrees} degrees {direction}."
    return labels


def write_simple_reference_labels(
    config: DatasetConfig,
    base_labels: Dict[str, str],
    rotation_directions: Optional[Dict[int, str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Build the simple label mapping and write it as JSON.

    Defaults to <dataset_directory>/reference_views_simple.json -- a
    different filename from reference_views.json on purpose, so this never
    overwrites output from the full landmark pipeline.
    """
    labels = build_simple_labels(config, base_labels, rotation_directions)
    output_path = output_path or (config.dataset_directory / "reference_views_simple.json")
    output_path = Path(output_path)
    output_path.write_text(json.dumps(labels, indent=2))
    return output_path
