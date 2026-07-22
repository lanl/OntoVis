"""
Generic pipeline for generating rotated reference images and structured,
landmark-based descriptions for medical or scientific objects (skulls, feet,
brains, bones, ...). See README.md at the repo root for the full spec.

This module must stay object-agnostic: it must not assume a chin, teeth, eye
sockets, an anterior/posterior side, or any other anatomy-specific concept.
Object-specific data lives in a dataset config.json and in the frozen
landmark_profile.json generated per dataset -- never in this file.

Pipeline:
    dataset config
        -> load or build one frozen landmark profile (Stage 1, model call)
        -> generate rotated reference images
        -> diagnose every image against the same profile (Stage 2, model call)
        -> validate the structured landmark signature
        -> render canonical, deterministic sentences
        -> write reference_views.json
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from camera_reasoning.chatgpt_client import ask_chatgpt

SCHEMA_VERSION = "medical_reference_views.v1"
PROFILE_SCHEMA_VERSION = "landmark_profile.v1"
PROMPT_VERSION = "v2"  # bumped: diagnosis prompt now includes the known view_label as context

MAX_VALIDATION_RETRIES = 2
MIN_LANDMARKS = 6
MAX_LANDMARKS = 15

VISIBILITY_VALUES = {"visible", "partially_visible", "not_visible", "uncertain"}
CLARITY_VALUES = {"clear", "unclear", "not_applicable"}
HORIZONTAL_VALUES = {"left", "right", "aligned", "unclear", "not_applicable"}
VERTICAL_VALUES = {"above", "below", "aligned", "unclear", "not_applicable"}
APPARENT_SIZE_VALUES = {"larger", "smaller", "similar", "unclear", "not_applicable"}
VISIBILITY_DOMINANCE_VALUES = {"a_more_visible", "b_more_visible", "similar", "unclear", "not_applicable"}
OVERLAP_VALUES = {"overlapping", "separate", "unclear", "not_applicable"}

DIMENSION_VALUES = {
    "horizontal": HORIZONTAL_VALUES,
    "vertical": VERTICAL_VALUES,
    "apparent_size": APPARENT_SIZE_VALUES,
    "visibility_dominance": VISIBILITY_DOMINANCE_VALUES,
    "overlap": OVERLAP_VALUES,
}
ALLOWED_DIMENSIONS = set(DIMENSION_VALUES)

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass
class SeedImage:
    node_id: str
    view_label: str


@dataclass
class DatasetConfig:
    object_id: str
    object_context: str
    dataset_directory: Path
    seeds: List[SeedImage]
    rotations: List[int]

    @property
    def landmark_profile_path(self) -> Path:
        return self.dataset_directory / "landmark_profile.json"

    @property
    def reference_views_path(self) -> Path:
        return self.dataset_directory / "reference_views.json"


def load_dataset_config(path) -> DatasetConfig:
    data = json.loads(Path(path).read_text())
    return DatasetConfig(
        object_id=data["object_id"],
        object_context=data["object_context"],
        dataset_directory=Path(data["dataset_directory"]),
        seeds=[SeedImage(node_id=s["node_id"], view_label=s["view_label"]) for s in data["seeds"]],
        rotations=list(data["rotations"]),
    )


# ---------------------------------------------------------------------------
# Landmark profile
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    id: str
    name: str
    definition: str


@dataclass
class RelationSpec:
    id: str
    landmark_a: str
    landmark_b: str
    dimensions: List[str]


@dataclass
class LandmarkProfile:
    schema_version: str
    object_id: str
    object_context: str
    landmarks: List[Landmark]
    relation_specs: List[RelationSpec]

    def landmark_ids(self) -> List[str]:
        return [lm.id for lm in self.landmarks]

    def landmark_by_id(self, landmark_id: str) -> Landmark:
        for lm in self.landmarks:
            if lm.id == landmark_id:
                return lm
        raise KeyError(landmark_id)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "object_id": self.object_id,
            "object_context": self.object_context,
            "landmarks": [lm.__dict__ for lm in self.landmarks],
            "relation_specs": [rs.__dict__ for rs in self.relation_specs],
        }

    @staticmethod
    def from_dict(data: dict) -> "LandmarkProfile":
        return LandmarkProfile(
            schema_version=data.get("schema_version", PROFILE_SCHEMA_VERSION),
            object_id=data.get("object_id", ""),
            object_context=data.get("object_context", ""),
            landmarks=[Landmark(**lm) for lm in data["landmarks"]],
            relation_specs=[RelationSpec(**rs) for rs in data["relation_specs"]],
        )

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


LANDMARK_PROFILE_PROMPT_TEMPLATE = """
You are defining a frozen landmark profile for {object_context}. This profile
will be reused, unchanged, for every reference image and every future
candidate image of this exact dataset. You are shown {num_seeds} image(s) of
the same object from different views -- inspect them together and produce one
shared profile, not a separate profile per image.

Select between {min_landmarks} and {max_landmarks} landmarks that are:
- visually recognizable and concrete
- stable and identifiable across multiple views
- useful for distinguishing camera orientation
- detectable directly from image pixels
- named consistently

Do not select:
- camera-view labels (e.g. "front view")
- anatomical orientation conclusions (e.g. "anterior side")
- vague regions (e.g. "interesting area")
- extremely small or unreliably detectable structures
- redundant variants of another landmark
- exhaustive anatomical inventories with little discriminative value

For each landmark, provide:
- "id": a unique, lowercase snake_case identifier that refers to one visual
  structure and will never be renamed
- "name": a short human-readable name
- "definition": a one-sentence visual definition

Then define "relation_specs": a small set of landmark pairs whose spatial
relationship helps distinguish orientation. Do not include every possible
pair -- only pairs whose relationship is discriminative. For each relation
spec provide:
- "id": a unique snake_case identifier
- "landmark_a" / "landmark_b": landmark ids defined above (the canonical
  order is "landmark_a relative to landmark_b")
- "dimensions": a non-empty subset of exactly these allowed values:
  {allowed_dimensions}

Return only one valid JSON object using exactly this structure, with no
markdown, commentary, or text outside the JSON:

{{
  "landmarks": [
    {{"id": "...", "name": "...", "definition": "..."}}
  ],
  "relation_specs": [
    {{"id": "...", "landmark_a": "...", "landmark_b": "...", "dimensions": ["..."]}}
  ]
}}
"""


def _extract_json_object(text: str) -> Optional[dict]:
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


def _normalize_profile_dict(parsed: dict, object_context: str) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    landmarks = parsed.get("landmarks")
    relation_specs = parsed.get("relation_specs")
    if not isinstance(landmarks, list) or not isinstance(relation_specs, list):
        raise ValueError("response must contain 'landmarks' and 'relation_specs' lists")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "object_context": object_context,
        "landmarks": landmarks,
        "relation_specs": relation_specs,
    }


def _validate_profile_dict(profile_dict: dict) -> List[str]:
    errors = []
    landmarks = profile_dict["landmarks"]
    relation_specs = profile_dict["relation_specs"]

    if not (MIN_LANDMARKS <= len(landmarks) <= MAX_LANDMARKS):
        errors.append(f"expected {MIN_LANDMARKS}-{MAX_LANDMARKS} landmarks, got {len(landmarks)}")

    seen_ids = set()
    for lm in landmarks:
        if not isinstance(lm, dict) or not all(k in lm for k in ("id", "name", "definition")):
            errors.append(f"malformed landmark entry: {lm!r}")
            continue
        lid = lm["id"]
        if not isinstance(lid, str) or not _SNAKE_CASE_RE.match(lid):
            errors.append(f"landmark id {lid!r} must be lowercase snake_case")
        if lid in seen_ids:
            errors.append(f"duplicate landmark id: {lid!r}")
        seen_ids.add(lid)

    seen_relation_ids = set()
    for rs in relation_specs:
        required = ("id", "landmark_a", "landmark_b", "dimensions")
        if not isinstance(rs, dict) or not all(k in rs for k in required):
            errors.append(f"malformed relation spec: {rs!r}")
            continue
        rid = rs["id"]
        if rid in seen_relation_ids:
            errors.append(f"duplicate relation id: {rid!r}")
        seen_relation_ids.add(rid)
        if rs["landmark_a"] not in seen_ids:
            errors.append(f"relation {rid!r} references unknown landmark_a {rs['landmark_a']!r}")
        if rs["landmark_b"] not in seen_ids:
            errors.append(f"relation {rid!r} references unknown landmark_b {rs['landmark_b']!r}")
        dims = rs["dimensions"]
        if not isinstance(dims, list) or not dims:
            errors.append(f"relation {rid!r} must have a non-empty 'dimensions' list")
        elif not set(dims) <= ALLOWED_DIMENSIONS:
            errors.append(f"relation {rid!r} has invalid dimensions: {sorted(set(dims) - ALLOWED_DIMENSIONS)}")

    return errors


def build_landmark_profile(
    seed_image_paths: List[Path],
    object_context: str,
    model: Optional[str] = None,
) -> LandmarkProfile:
    """Inspect all seed images together and generate one shared, frozen
    landmark profile. Never produces a separate profile per image."""
    if not seed_image_paths:
        raise ValueError("build_landmark_profile requires at least one seed image")

    prompt = LANDMARK_PROFILE_PROMPT_TEMPLATE.format(
        object_context=object_context,
        num_seeds=len(seed_image_paths),
        min_landmarks=MIN_LANDMARKS,
        max_landmarks=MAX_LANDMARKS,
        allowed_dimensions=", ".join(sorted(ALLOWED_DIMENSIONS)),
    )
    primary = str(seed_image_paths[0])
    extra = [(f"seed_{i + 2}", str(p)) for i, p in enumerate(seed_image_paths[1:])]

    last_error = None
    for _ in range(MAX_VALIDATION_RETRIES + 1):
        response_text = ask_chatgpt(prompt=prompt, screenshot_path=primary, extra_images=extra or None, model=model)
        parsed = _extract_json_object(response_text)
        if parsed is None:
            last_error = f"model response was not valid JSON: {response_text!r}"
            prompt = f"{prompt}\n\nYour previous response was invalid: {last_error}\nReturn a corrected JSON object only."
            continue

        try:
            profile_dict = _normalize_profile_dict(parsed, object_context)
            errors = _validate_profile_dict(profile_dict)
        except ValueError as e:
            errors = [str(e)]

        if not errors:
            return LandmarkProfile.from_dict(profile_dict)

        last_error = "; ".join(errors)
        prompt = f"{prompt}\n\nYour previous response had these validation errors: {last_error}\nReturn a corrected JSON object only."

    raise ValueError(f"Failed to generate a valid landmark profile after {MAX_VALIDATION_RETRIES + 1} attempts: {last_error}")


def load_or_build_landmark_profile(
    config: DatasetConfig,
    model: Optional[str] = None,
    force: bool = False,
) -> LandmarkProfile:
    profile_path = config.landmark_profile_path
    if profile_path.exists() and not force:
        return LandmarkProfile.from_dict(json.loads(profile_path.read_text()))

    seed_paths = [config.dataset_directory / f"{seed.node_id}.png" for seed in config.seeds]
    profile = build_landmark_profile(seed_paths, config.object_context, model=model)
    profile.object_id = config.object_id

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2))
    print(f"[medical_reference_views] Wrote landmark profile to {profile_path}")
    return profile


# ---------------------------------------------------------------------------
# Rotation generation
# ---------------------------------------------------------------------------

def _normalize_mode_for_processing(im: Image.Image) -> Image.Image:
    """Keep grayscale/RGB/RGBA as-is; convert anything else (palette, bilevel,
    unusual pixel formats) to a mode getpixel()/rotate() can handle safely."""
    if im.mode in ("L", "LA", "RGB", "RGBA"):
        return im
    if im.mode in ("1", "I", "I;16", "F"):
        return im.convert("L")
    return im.convert("RGBA")


def estimate_background_fill(im: Image.Image):
    """Estimate a background fill value from the image corners, matching the
    image's own mode: a scalar for "L", a same-length tuple for multi-channel
    modes ("LA", "RGB", "RGBA", ...). Safe for very small images."""
    w, h = im.size
    xs = sorted({min(2, w - 1), max(w - 3, 0)})
    ys = sorted({min(2, h - 1), max(h - 3, 0)})
    pts = [im.getpixel((x, y)) for x in xs for y in ys]

    if im.mode == "L":
        return int(round(sum(pts) / len(pts)))

    num_channels = len(pts[0])
    return tuple(int(round(sum(p[c] for p in pts) / len(pts))) for c in range(num_channels))


def generate_rotated_images(config: DatasetConfig) -> List[dict]:
    """Rotate each seed image clockwise in the image plane through
    config.rotations, keeping the full expanded image and filling exposed
    corners with an estimated background color/value. Returns raw image
    records (not yet diagnosed)."""
    records = []
    for seed in config.seeds:
        seed_path = config.dataset_directory / f"{seed.node_id}.png"
        im = _normalize_mode_for_processing(Image.open(seed_path))
        fill = estimate_background_fill(im)

        records.append({
            "node_id": seed.node_id,
            "source_node_id": seed.node_id,
            "view_label": seed.view_label,
            "in_plane_rotation_degrees": 0,
            "image_path": str(seed_path),
        })

        for degrees in config.rotations:
            # Clockwise in-plane rotation; PIL rotates counter-clockwise for
            # positive angles, so negate. expand=True keeps the full rotated
            # image instead of cropping the corners.
            rotated = im.rotate(-degrees, expand=True, fillcolor=fill)
            out_name = f"{seed.node_id}_rot{degrees:03d}"
            out_path = config.dataset_directory / f"{out_name}.png"
            rotated.save(out_path)

            records.append({
                "node_id": out_name,
                "source_node_id": seed.node_id,
                "view_label": seed.view_label,
                "in_plane_rotation_degrees": degrees,
                "image_path": str(out_path),
            })
            print(f"[medical_reference_views] Saved {out_path} ({rotated.size[0]}x{rotated.size[1]})")

    return records


def _existing_image_records(config: DatasetConfig) -> List[dict]:
    """Build image records from already-generated files on disk, without
    regenerating rotations (used by --skip-rotations / --describe-only)."""
    records = []
    for seed in config.seeds:
        seed_path = config.dataset_directory / f"{seed.node_id}.png"
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed image not found: {seed_path}")
        records.append({
            "node_id": seed.node_id,
            "source_node_id": seed.node_id,
            "view_label": seed.view_label,
            "in_plane_rotation_degrees": 0,
            "image_path": str(seed_path),
        })
        for degrees in config.rotations:
            out_name = f"{seed.node_id}_rot{degrees:03d}"
            out_path = config.dataset_directory / f"{out_name}.png"
            if not out_path.exists():
                raise FileNotFoundError(
                    f"Rotated image not found: {out_path}. Run without --skip-rotations first."
                )
            records.append({
                "node_id": out_name,
                "source_node_id": seed.node_id,
                "view_label": seed.view_label,
                "in_plane_rotation_degrees": degrees,
                "image_path": str(out_path),
            })
    return records


# ---------------------------------------------------------------------------
# Stage 2: image diagnosis
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT_TEMPLATE = """
You are diagnosing a single image of {object_context} using a frozen landmark
profile. Evaluate only what is visibly supported by the image pixels.
{view_label_context}
Landmarks to evaluate (evaluate every one; do not add, omit, or rename any):
{landmark_list}

For each landmark return:
- "visibility": one of {visibility_values}
- "clarity": one of {clarity_values}
- "confidence": a number from 0.0 to 1.0

Rules:
- If visibility is "not_visible", clarity must be "not_applicable".
- If visibility is "uncertain", clarity should generally be "unclear" or
  "not_applicable".
- Prefer "unclear"/"uncertain" over guessing.

Relations to evaluate (evaluate only the listed dimensions for each pair,
describing landmark_a relative to landmark_b):
{relation_list}

Use only these fixed allowed values for each dimension:
- horizontal: {horizontal_values}
- vertical: {vertical_values}
- apparent_size: {apparent_size_values}
- visibility_dominance: {visibility_dominance_values}
- overlap: {overlap_values}

Apply this logic for every relation:
- If either landmark is not_visible: set every requested dimension for that
  pair to "not_applicable".
- If either landmark is uncertain, or too unclear to locate: set every
  requested dimension for that pair to "unclear".
- Otherwise, evaluate only the requested dimensions normally.

All spatial relations use image coordinates only (displayed image
left/right/top/bottom). Do not interpret image left/right/top/bottom as
anatomical left/right or superior/inferior.

Return only one valid JSON object using exactly this structure, with no
markdown, commentary, or text outside the JSON:

{{
  "landmarks": {{
    "<landmark_id>": {{"visibility": "...", "clarity": "...", "confidence": 0.0}}
  }},
  "relations": {{
    "<relation_id>": {{"<dimension>": "...", "confidence": 0.0}}
  }}
}}
"""


def _format_diagnosis_prompt(profile: LandmarkProfile, view_label: Optional[str] = None) -> str:
    landmark_list = "\n".join(f"- {lm.id}: {lm.definition}" for lm in profile.landmarks)
    relation_list = "\n".join(
        f"- {rs.id}: {rs.landmark_a} relative to {rs.landmark_b} -- dimensions: {', '.join(rs.dimensions)}"
        for rs in profile.relation_specs
    )
    if view_label:
        # Deliberate deviation from the "no label leakage" default: the known
        # view label is given as orientation context so the model doesn't
        # confuse anatomical left/right with image-coordinate left/right. It
        # must still ground every judgment in the pixels, not in expectations
        # about what a typical view of this label looks like.
        view_label_context = (
            f"\nThis image's known camera-view label is: {view_label}. Use it only "
            "to orient yourself (e.g. to avoid confusing anatomical left/right with "
            "image-coordinate left/right) -- every visibility, clarity, and relation "
            "judgment must still be based on what is actually visible in this "
            "specific image, not on assumptions about what a typical view with this "
            "label looks like.\n"
        )
    else:
        view_label_context = ""

    return DIAGNOSIS_PROMPT_TEMPLATE.format(
        object_context=profile.object_context,
        view_label_context=view_label_context,
        landmark_list=landmark_list,
        relation_list=relation_list,
        visibility_values=sorted(VISIBILITY_VALUES),
        clarity_values=sorted(CLARITY_VALUES),
        horizontal_values=sorted(HORIZONTAL_VALUES),
        vertical_values=sorted(VERTICAL_VALUES),
        apparent_size_values=sorted(APPARENT_SIZE_VALUES),
        visibility_dominance_values=sorted(VISIBILITY_DOMINANCE_VALUES),
        overlap_values=sorted(OVERLAP_VALUES),
    )


def validate_signature(signature: dict, profile: LandmarkProfile) -> List[str]:
    """Validate a structured landmark signature against the frozen profile.
    Returns a list of human-readable errors (empty if valid)."""
    errors = []
    landmarks = signature.get("landmarks")
    relations = signature.get("relations")
    if not isinstance(landmarks, dict):
        return ["missing or invalid 'landmarks' object"]
    if not isinstance(relations, dict):
        return ["missing or invalid 'relations' object"]

    profile_landmark_ids = set(profile.landmark_ids())
    extra_landmarks = set(landmarks) - profile_landmark_ids
    missing_landmarks = profile_landmark_ids - set(landmarks)
    if extra_landmarks:
        errors.append(f"unknown landmark ids: {sorted(extra_landmarks)}")
    if missing_landmarks:
        errors.append(f"missing required landmarks: {sorted(missing_landmarks)}")

    landmark_visibility = {}
    for lid, entry in landmarks.items():
        if lid not in profile_landmark_ids:
            continue
        if not isinstance(entry, dict):
            errors.append(f"landmark {lid!r} entry must be an object")
            continue
        visibility = entry.get("visibility")
        clarity = entry.get("clarity")
        confidence = entry.get("confidence")
        if visibility not in VISIBILITY_VALUES:
            errors.append(f"landmark {lid!r} has invalid visibility {visibility!r}")
        if clarity not in CLARITY_VALUES:
            errors.append(f"landmark {lid!r} has invalid clarity {clarity!r}")
        if visibility == "not_visible" and clarity != "not_applicable":
            errors.append(f"landmark {lid!r} is not_visible but clarity is {clarity!r} (must be not_applicable)")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"landmark {lid!r} has invalid confidence {confidence!r}")
        landmark_visibility[lid] = visibility

    profile_relations = {rs.id: rs for rs in profile.relation_specs}
    extra_relations = set(relations) - set(profile_relations)
    missing_relations = set(profile_relations) - set(relations)
    if extra_relations:
        errors.append(f"unknown relation ids: {sorted(extra_relations)}")
    if missing_relations:
        errors.append(f"missing required relations: {sorted(missing_relations)}")

    for rid, entry in relations.items():
        spec = profile_relations.get(rid)
        if spec is None:
            continue
        if not isinstance(entry, dict):
            errors.append(f"relation {rid!r} entry must be an object")
            continue

        confidence = entry.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"relation {rid!r} has invalid confidence {confidence!r}")

        present_dims = set(entry) - {"confidence"}
        expected_dims = set(spec.dimensions)
        if present_dims != expected_dims:
            errors.append(f"relation {rid!r} dimensions mismatch: expected {sorted(expected_dims)}, got {sorted(present_dims)}")

        vis_a = landmark_visibility.get(spec.landmark_a)
        vis_b = landmark_visibility.get(spec.landmark_b)
        for dim in expected_dims:
            value = entry.get(dim)
            allowed = DIMENSION_VALUES.get(dim, set())
            if value not in allowed:
                errors.append(f"relation {rid!r} dimension {dim!r} has invalid value {value!r}")
                continue
            if (vis_a == "not_visible" or vis_b == "not_visible") and value != "not_applicable":
                errors.append(
                    f"relation {rid!r} dimension {dim!r} must be not_applicable because "
                    f"{spec.landmark_a!r} or {spec.landmark_b!r} is not_visible"
                )
            elif (vis_a == "uncertain" or vis_b == "uncertain") and value not in ("unclear", "not_applicable"):
                errors.append(
                    f"relation {rid!r} dimension {dim!r} must be unclear because "
                    f"{spec.landmark_a!r} or {spec.landmark_b!r} is uncertain"
                )

    return errors


def diagnose_image(
    image_path,
    profile: LandmarkProfile,
    view_label: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Diagnose a single image against the frozen profile, retrying with a
    bounded number of correction prompts on invalid model output.

    `view_label` (e.g. "right", "front", "left"), if given, is passed to the
    model as orientation context -- see _format_diagnosis_prompt."""
    base_prompt = _format_diagnosis_prompt(profile, view_label=view_label)
    prompt = base_prompt
    last_error = None

    for _ in range(MAX_VALIDATION_RETRIES + 1):
        response_text = ask_chatgpt(prompt=prompt, screenshot_path=str(image_path), model=model)
        parsed = _extract_json_object(response_text)
        if parsed is None:
            last_error = f"model response was not valid JSON: {response_text!r}"
            prompt = f"{base_prompt}\n\nYour previous response was invalid: {last_error}\nReturn a corrected JSON object only."
            continue

        errors = validate_signature(parsed, profile)
        if not errors:
            return parsed

        last_error = "; ".join(errors)
        prompt = f"{base_prompt}\n\nYour previous response had these validation errors: {last_error}\nReturn a corrected JSON object only."

    raise ValueError(f"Failed to diagnose {image_path} after {MAX_VALIDATION_RETRIES + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Canonical sentence rendering
# ---------------------------------------------------------------------------

LANDMARK_VISIBILITY_SENTENCES = {
    "visible": "{name} is visible.",
    "partially_visible": "{name} is partially visible.",
    "not_visible": "{name} is not visible.",
    "uncertain": "{name} visibility is uncertain.",
}
LANDMARK_CLARITY_SENTENCES = {
    "clear": "{name} is clear.",
    "unclear": "{name} is unclear.",
}
HORIZONTAL_SENTENCES = {
    "left": "{a} is left of {b}.",
    "right": "{a} is right of {b}.",
    "aligned": "{a} is horizontally aligned with {b}.",
}
VERTICAL_SENTENCES = {
    "above": "{a} is above {b}.",
    "below": "{a} is below {b}.",
    "aligned": "{a} is vertically aligned with {b}.",
}
APPARENT_SIZE_SENTENCES = {
    "larger": "{a} appears larger than {b}.",
    "smaller": "{a} appears smaller than {b}.",
    "similar": "{a} and {b} have similar apparent size.",
}
VISIBILITY_DOMINANCE_SENTENCES = {
    "a_more_visible": "{a} is more visible than {b}.",
    "b_more_visible": "{b} is more visible than {a}.",
    "similar": "{a} and {b} are similarly visible.",
}
OVERLAP_SENTENCES = {
    "overlapping": "{a} overlaps {b}.",
    "separate": "{a} is separate from {b}.",
}
DIMENSION_SENTENCE_MAP = {
    "horizontal": HORIZONTAL_SENTENCES,
    "vertical": VERTICAL_SENTENCES,
    "apparent_size": APPARENT_SIZE_SENTENCES,
    "visibility_dominance": VISIBILITY_DOMINANCE_SENTENCES,
    "overlap": OVERLAP_SENTENCES,
}
DIMENSION_UNCLEAR_TEXT = {
    "horizontal": "horizontal",
    "vertical": "vertical",
    "apparent_size": "apparent size",
    "visibility_dominance": "visibility dominance",
    "overlap": "overlap",
}


def render_description_sentences(profile: LandmarkProfile, signature: dict) -> List[str]:
    """Deterministically render canonical sentences from a validated
    structured signature. Order: landmark visibility, landmark clarity (in
    profile order), then relations in profile order and dimension order."""
    sentences = []
    landmarks = signature["landmarks"]
    relations = signature["relations"]

    for lm in profile.landmarks:
        entry = landmarks[lm.id]
        vis_template = LANDMARK_VISIBILITY_SENTENCES.get(entry["visibility"])
        if vis_template:
            sentences.append(vis_template.format(name=lm.name))
        clarity_template = LANDMARK_CLARITY_SENTENCES.get(entry["clarity"])
        if clarity_template:
            sentences.append(clarity_template.format(name=lm.name))

    for spec in profile.relation_specs:
        entry = relations[spec.id]
        name_a = profile.landmark_by_id(spec.landmark_a).name
        name_b = profile.landmark_by_id(spec.landmark_b).name
        for dim in spec.dimensions:
            value = entry.get(dim)
            if value == "not_applicable":
                continue
            if value == "unclear":
                sentences.append(f"The {DIMENSION_UNCLEAR_TEXT[dim]} relationship between {name_a} and {name_b} is unclear.")
                continue
            template = DIMENSION_SENTENCE_MAP[dim].get(value)
            if template:
                sentences.append(template.format(a=name_a, b=name_b))

    return sentences


# ---------------------------------------------------------------------------
# Caching + top-level orchestration
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_cache(reference_views_path: Path) -> Dict[str, dict]:
    if not reference_views_path.exists():
        return {}
    try:
        data = json.loads(reference_views_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        # Pre-existing reference_views.json from the old (pre-README) schema
        # was a bare list of records with no image/profile hashes -- there is
        # nothing cacheable in it, so treat it as an empty cache.
        return {}
    return {r["node_id"]: r for r in data.get("records", []) if "node_id" in r}


def generate_reference_views(
    config: DatasetConfig,
    model: Optional[str] = None,
    force_profile: bool = False,
    force_descriptions: bool = False,
    skip_rotations: bool = False,
    describe_only: bool = False,
) -> dict:
    profile = load_or_build_landmark_profile(config, model=model, force=force_profile)
    profile_hash = profile.content_hash()

    if skip_rotations or describe_only:
        image_records = _existing_image_records(config)
    else:
        image_records = generate_rotated_images(config)

    # Rebuilding the profile invalidates every cached signature: a changed
    # profile_hash means no cached record will match below.
    cache = _load_cache(config.reference_views_path)

    records = []
    for image_record in image_records:
        node_id = image_record["node_id"]
        image_path = Path(image_record["image_path"])
        image_hash = _sha256_file(image_path)

        cached = None if force_descriptions else cache.get(node_id)
        if (
            cached is not None
            and cached.get("image_hash") == image_hash
            and cached.get("profile_hash") == profile_hash
            and cached.get("schema_version") == SCHEMA_VERSION
            and cached.get("prompt_version") == PROMPT_VERSION
        ):
            print(f"[medical_reference_views] Using cached signature for {node_id}")
            record = dict(cached)
            record.update({
                "node_id": node_id,
                "source_node_id": image_record["source_node_id"],
                "image_path": str(image_path),
                "view_label": image_record["view_label"],
                "in_plane_rotation_degrees": image_record["in_plane_rotation_degrees"],
            })
            records.append(record)
            continue

        print(f"[medical_reference_views] Diagnosing {node_id} ({image_path})...")
        signature = diagnose_image(image_path, profile, view_label=image_record["view_label"], model=model)

        record = {
            "node_id": node_id,
            "source_node_id": image_record["source_node_id"],
            "image_path": str(image_path),
            "view_label": image_record["view_label"],
            "in_plane_rotation_degrees": image_record["in_plane_rotation_degrees"],
            "spatial_frame": "image_coordinates",
            "image_hash": image_hash,
            "profile_hash": profile_hash,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "landmarks": signature["landmarks"],
            "relations": signature["relations"],
        }
        record["description_sentences"] = render_description_sentences(profile, record)
        records.append(record)

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "object_id": config.object_id,
        "object_context": config.object_context,
        "landmark_profile_path": "landmark_profile.json",
        "landmark_profile_hash": profile_hash,
        "records": records,
    }

    out_path = config.reference_views_path
    out_path.write_text(json.dumps(dataset, indent=2))
    print(f"[medical_reference_views] Wrote {len(records)} records to {out_path}")
    return dataset
