"""
Generate a camera-relative graph of reference views.

Key properties:

1. Node names stay dataset-agnostic: ``view_000``, ``view_001``, ...
2. The manually chosen root position and focal point are still supported.
3. Root ``ViewUp`` is derived from a stable object-space up direction.
4. Every non-root node is restored from its parent's exact camera state and then
   receives exactly one elementary VTK camera action.
5. Optional per-view roll canonicalization is available for vision-model or
   image-embedding use.
6. Every node receives a short deterministic viewpoint description generated
   from its camera position and the calibrated object axes. No LLM is used to
   label the reference views.

Important distinction:

- ``position`` and ``focal_point`` determine where the camera is relative to the
  object.
- ``object_anterior_world``, ``object_left_world``, and ``object_up_world``
  define the object's canonical orientation in VTK world coordinates.
- ``ViewUp`` controls image-plane roll; it does not change which side of the
  object the camera sees.

For reliable anatomical labels, calibrate the three object axes once for the
current dataset. The per-view descriptions are then computed directly from the
saved camera positions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera_state import get_camera_state, set_camera_state
from .volume_scene import (
    build_isosurface_pipeline,
    load_raw_volume,
    save_screenshot,
)

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RootCameraCalibration:
    """Dataset-specific root camera and object-orientation calibration.

    ``position`` and ``focal_point`` define the manually calibrated root camera.

    ``object_anterior_world`` points from the object center toward the object's
    anatomical front/anterior direction in VTK world coordinates.

    ``object_left_world`` points toward the object's anatomical left direction.
    This explicit vector removes any ambiguity about left versus right.

    ``object_up_world`` points toward the object's canonical top/superior
    direction. It is also used to make the rendered image upright.

    The three object axes do not need to be perfectly normalized, but they must
    be non-zero and approximately mutually perpendicular. They are normalized
    and orthogonalized before use.

    ``fallback_up_world`` is used only when ``object_up_world`` is parallel to
    the viewing direction and therefore cannot define image top.
    """

    position: Vector3
    focal_point: Vector3
    object_anterior_world: Vector3
    object_left_world: Vector3
    object_up_world: Vector3
    fallback_up_world: Vector3 = (0.0, 0.0, 1.0)
    parallel_projection: bool = False
    parallel_scale: Optional[float] = None


@dataclass(frozen=True)
class ViewDescriptionSettings:
    """Controls deterministic conversion from camera position to view label.

    A secondary directional component is included when its absolute camera-axis
    score is at least ``oblique_ratio_threshold`` times the strongest score.
    This makes exact 45-degree views become oblique labels while keeping 90-degree
    views as pure frontal/lateral/superior views.

    A third component is included only when it also satisfies the same ratio and
    ``include_third_component`` is enabled.
    """

    oblique_ratio_threshold: float = 0.55
    minimum_component_score: float = 0.20
    include_third_component: bool = True


# ======================================================================
# DATASET CALIBRATION
#
# EDIT THESE THREE OBJECT AXES ONCE FOR THE CURRENT DATASET.
#
# They determine the semantic description of every generated reference view.
# Example only for a dataset whose:
#   anterior points toward -X,
#   anatomical left points toward +Y,
#   superior points toward +Z.
#
# If left/right descriptions are reversed, negate object_left_world.
# Do not change individual node descriptions manually.
# ======================================================================

ROOT_CAMERA = RootCameraCalibration(
    position=(-624.664, 118.625, 121.945),
    focal_point=(142.426, 118.625, 121.945),

    # REQUIRED DATASET CALIBRATION for the skull volume.
    # The deterministic descriptions in reference_views.json are calculated
    # from these axes. In this dataset, anatomical left points toward -Y.
    object_anterior_world=(-1.0, 0.0, 0.0),
    object_left_world=(0.0, -1.0, 0.0),
    object_up_world=(0.0, 0.0, 1.0),

    # Used for pole views where object_up_world points into/out of the screen.
    fallback_up_world=(0.0, 1.0, 0.0),

    parallel_projection=False,
    parallel_scale=None,
)

VIEW_DESCRIPTION_SETTINGS = ViewDescriptionSettings(
    oblique_ratio_threshold=0.55,
    minimum_component_score=0.20,
    include_third_component=True,
)


# ======================================================================
# CAMERA-RELATIVE VIEW GRAPH
#
# Each non-root node is generated by restoring its parent and applying exactly
# one elementary VTK camera action. Node IDs remain semantically neutral.
# ======================================================================

VIEW_GENERATION_PLAN: List[dict] = [
    {"node_id": "view_000", "parent_id": None, "action": None},

    # Level 1
    {"node_id": "view_001", "parent_id": "view_000", "action": {"type": "azimuth", "degrees": 45.0}},
    {"node_id": "view_002", "parent_id": "view_000", "action": {"type": "azimuth", "degrees": -45.0}},
    {"node_id": "view_003", "parent_id": "view_000", "action": {"type": "azimuth", "degrees": 90.0}},
    {"node_id": "view_004", "parent_id": "view_000", "action": {"type": "azimuth", "degrees": -90.0}},
    {"node_id": "view_005", "parent_id": "view_000", "action": {"type": "elevation", "degrees": 45.0}},
    {"node_id": "view_006", "parent_id": "view_000", "action": {"type": "elevation", "degrees": -45.0}},
    {"node_id": "view_007", "parent_id": "view_000", "action": {"type": "elevation", "degrees": 90.0}},
    {"node_id": "view_008", "parent_id": "view_000", "action": {"type": "elevation", "degrees": -90.0}},

    # Level 2
    {"node_id": "view_009", "parent_id": "view_001", "action": {"type": "elevation", "degrees": 45.0}},
    {"node_id": "view_010", "parent_id": "view_001", "action": {"type": "elevation", "degrees": -45.0}},
    {"node_id": "view_011", "parent_id": "view_002", "action": {"type": "elevation", "degrees": 45.0}},
    {"node_id": "view_012", "parent_id": "view_002", "action": {"type": "elevation", "degrees": -45.0}},
    {"node_id": "view_013", "parent_id": "view_003", "action": {"type": "elevation", "degrees": 45.0}},
    {"node_id": "view_014", "parent_id": "view_003", "action": {"type": "elevation", "degrees": -45.0}},
    {"node_id": "view_015", "parent_id": "view_004", "action": {"type": "elevation", "degrees": 45.0}},
    {"node_id": "view_016", "parent_id": "view_004", "action": {"type": "elevation", "degrees": -45.0}},
    {"node_id": "view_017", "parent_id": "view_005", "action": {"type": "azimuth", "degrees": 45.0}},
]


def _as_vector(vector: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values; got {vector!r}.")
    return result


def _normalize(vector: Sequence[float], name: str) -> np.ndarray:
    result = _as_vector(vector, name)
    norm = float(np.linalg.norm(result))
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero.")
    return result / norm


def _orthogonalized_object_axes(
    calibration: RootCameraCalibration,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized, mutually orthogonal anterior/left/superior axes.

    The explicit left vector is preserved as the semantic left direction. The
    superior vector is projected away from anterior, and left is then projected
    away from both. A final superior vector is reconstructed to eliminate small
    calibration noise while preserving the configured handedness.
    """
    anterior = _normalize(
        calibration.object_anterior_world,
        "object_anterior_world",
    )

    superior_raw = _as_vector(calibration.object_up_world, "object_up_world")
    superior_projected = superior_raw - np.dot(superior_raw, anterior) * anterior
    superior_norm = float(np.linalg.norm(superior_projected))
    if superior_norm < 1e-8:
        raise ValueError(
            "object_up_world must not be parallel to object_anterior_world. "
            "Calibrate the dataset's anatomical/canonical axes before generating "
            "reference descriptions."
        )
    superior = superior_projected / superior_norm

    left_raw = _as_vector(calibration.object_left_world, "object_left_world")
    left_projected = (
        left_raw
        - np.dot(left_raw, anterior) * anterior
        - np.dot(left_raw, superior) * superior
    )
    left_norm = float(np.linalg.norm(left_projected))
    if left_norm < 1e-8:
        raise ValueError(
            "object_left_world must be independent of object_anterior_world and "
            "object_up_world. Calibrate three distinct object axes."
        )
    left = left_projected / left_norm

    # Reconstruct superior using the calibrated semantic left direction. This
    # keeps the basis orthogonal without changing which vector means "left".
    reconstructed_superior = np.cross(anterior, left)
    reconstructed_superior = _normalize(
        reconstructed_superior,
        "reconstructed_object_up_world",
    )

    # Preserve the configured superior sign. If the reconstructed axis points in
    # the opposite direction, flip it rather than silently swapping left/right.
    if float(np.dot(reconstructed_superior, superior)) < 0.0:
        reconstructed_superior = -reconstructed_superior

    return anterior, left, reconstructed_superior


def _project_up_onto_image_plane(
    up_world: Sequence[float],
    direction_of_projection: Sequence[float],
) -> Optional[np.ndarray]:
    """Project a world-space up direction onto the current image plane."""
    up = _normalize(up_world, "up_world")
    dop = _normalize(direction_of_projection, "direction_of_projection")

    projected = up - np.dot(up, dop) * dop
    projected_norm = float(np.linalg.norm(projected))

    if projected_norm < 1e-8:
        return None

    return projected / projected_norm


def _canonical_view_up(
    *,
    direction_of_projection: Sequence[float],
    object_up_world: Sequence[float],
    fallback_up_world: Sequence[float],
) -> np.ndarray:
    """Return a stable image-up vector for the current viewing direction."""
    for candidate in (object_up_world, fallback_up_world, (1.0, 0.0, 0.0)):
        projected = _project_up_onto_image_plane(
            candidate,
            direction_of_projection,
        )
        if projected is not None:
            return projected

    raise ValueError(
        "Unable to derive a valid ViewUp vector. object_up_world and all "
        "fallback directions are parallel to the viewing direction."
    )


def _set_canonical_view_up(camera, calibration: RootCameraCalibration) -> None:
    """Align image top with the configured object-up direction.

    This changes camera roll only. Position, focal point, and therefore the
    actual object viewpoint remain unchanged.
    """
    _anterior, _left, superior = _orthogonalized_object_axes(calibration)
    view_up = _canonical_view_up(
        direction_of_projection=camera.GetDirectionOfProjection(),
        object_up_world=superior,
        fallback_up_world=calibration.fallback_up_world,
    )
    camera.SetViewUp(*map(float, view_up))
    camera.OrthogonalizeViewUp()


def _apply_root_camera(camera, calibration: RootCameraCalibration) -> None:
    camera.SetPosition(*calibration.position)
    camera.SetFocalPoint(*calibration.focal_point)

    camera.SetParallelProjection(calibration.parallel_projection)
    if calibration.parallel_projection and calibration.parallel_scale is not None:
        camera.SetParallelScale(calibration.parallel_scale)

    # Prevent a frontal or otherwise valid root viewpoint from starting with
    # arbitrary image-plane roll.
    _set_canonical_view_up(camera, calibration)


def _apply_elementary_action(camera, action: dict) -> None:
    action_type = action.get("type")
    degrees = float(action.get("degrees"))

    if action_type == "azimuth":
        camera.Azimuth(degrees)
    elif action_type == "elevation":
        camera.Elevation(degrees)
    elif action_type == "roll":
        camera.Roll(degrees)
    else:
        raise ValueError(
            f"Unsupported action type {action_type!r}. "
            "Expected 'azimuth', 'elevation', or 'roll'."
        )

    camera.OrthogonalizeViewUp()


def _validate_plan(plan: List[dict]) -> None:
    if not plan:
        raise ValueError("The generation plan cannot be empty.")

    seen: set[str] = set()
    root_count = 0

    for index, entry in enumerate(plan):
        node_id = entry.get("node_id")
        parent_id = entry.get("parent_id")
        action = entry.get("action")

        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Plan entry {index} has an invalid node_id.")

        if node_id in seen:
            raise ValueError(f"Duplicate node_id in plan: {node_id!r}")

        if parent_id is None:
            root_count += 1
            if action is not None:
                raise ValueError(f"Root node {node_id!r} must have action=None.")
        else:
            if parent_id not in seen:
                raise ValueError(
                    f"Node {node_id!r} references parent {parent_id!r} before "
                    "that parent has been generated."
                )
            if not isinstance(action, dict):
                raise ValueError(f"Non-root node {node_id!r} must define an action.")
            if action.get("type") not in {"azimuth", "elevation", "roll"}:
                raise ValueError(
                    f"Node {node_id!r} has unsupported action type "
                    f"{action.get('type')!r}."
                )
            if not isinstance(action.get("degrees"), (int, float)):
                raise ValueError(
                    f"Node {node_id!r} must provide numeric action degrees."
                )

        seen.add(node_id)

    if root_count != 1:
        raise ValueError(f"The plan must contain exactly one root; found {root_count}.")


def _validate_description_settings(settings: ViewDescriptionSettings) -> None:
    if not 0.0 < settings.oblique_ratio_threshold <= 1.0:
        raise ValueError("oblique_ratio_threshold must be in the interval (0, 1].")
    if not 0.0 <= settings.minimum_component_score <= 1.0:
        raise ValueError("minimum_component_score must be in the interval [0, 1].")


def _direction_label(axis_name: str, signed_score: float) -> str:
    if axis_name == "anterior":
        return "anterior" if signed_score >= 0.0 else "posterior"
    if axis_name == "left":
        return "left" if signed_score >= 0.0 else "right"
    if axis_name == "superior":
        return "superior" if signed_score >= 0.0 else "inferior"
    raise ValueError(f"Unknown object axis: {axis_name!r}")


def _format_view_description(component_labels: List[str]) -> str:
    """Convert selected signed components into one short reference description."""
    if not component_labels:
        return "Unclear view."

    if len(component_labels) == 1:
        label = component_labels[0]
        pure_labels = {
            "anterior": "Frontal view.",
            "posterior": "Posterior view.",
            "left": "Left lateral view.",
            "right": "Right lateral view.",
            "superior": "Superior view.",
            "inferior": "Inferior view.",
        }
        return pure_labels[label]

    display_names = {
        "anterior": "anterior",
        "posterior": "posterior",
        "left": "left",
        "right": "right",
        "superior": "superior",
        "inferior": "inferior",
    }
    joined = "-".join(display_names[label] for label in component_labels)
    return f"{joined.capitalize()} oblique view."


def _classify_camera_view(
    camera,
    calibration: RootCameraCalibration,
    settings: ViewDescriptionSettings,
) -> dict:
    """Classify a camera viewpoint from position relative to calibrated axes.

    The vector from focal point to camera position tells us which side of the
    object the camera occupies. Dot products against the canonical object axes
    produce deterministic anterior/posterior, left/right, and superior/inferior
    scores.
    """
    anterior, left, superior = _orthogonalized_object_axes(calibration)

    camera_position = _as_vector(camera.GetPosition(), "camera_position")
    focal_point = _as_vector(camera.GetFocalPoint(), "camera_focal_point")
    object_to_camera = _normalize(
        camera_position - focal_point,
        "object_to_camera",
    )

    signed_scores = {
        "anterior": float(np.dot(object_to_camera, anterior)),
        "left": float(np.dot(object_to_camera, left)),
        "superior": float(np.dot(object_to_camera, superior)),
    }

    ranked = sorted(
        (
            {
                "axis": axis_name,
                "label": _direction_label(axis_name, signed_score),
                "signed_score": signed_score,
                "absolute_score": abs(signed_score),
            }
            for axis_name, signed_score in signed_scores.items()
        ),
        key=lambda item: item["absolute_score"],
        reverse=True,
    )

    strongest = ranked[0]["absolute_score"]
    selected_components = [ranked[0]]

    for item in ranked[1:]:
        if len(selected_components) == 2 and not settings.include_third_component:
            break

        large_enough = item["absolute_score"] >= settings.minimum_component_score
        relative_enough = (
            strongest > 0.0
            and item["absolute_score"] / strongest
            >= settings.oblique_ratio_threshold
        )
        if large_enough and relative_enough:
            selected_components.append(item)

    # Keep descriptions stable and readable instead of ordering words by tiny
    # floating-point score differences.
    semantic_order = {
        "anterior": 0,
        "posterior": 0,
        "left": 1,
        "right": 1,
        "superior": 2,
        "inferior": 2,
    }
    component_labels = sorted(
        (item["label"] for item in selected_components),
        key=lambda label: semantic_order[label],
    )

    return {
        "description": _format_view_description(component_labels),
        "components": component_labels,
        "axis_scores": {
            "anterior_posterior": signed_scores["anterior"],
            "left_right": signed_scores["left"],
            "superior_inferior": signed_scores["superior"],
        },
        "object_to_camera_unit_vector": list(map(float, object_to_camera)),
        "calibrated_axes": {
            "anterior_world": list(map(float, anterior)),
            "left_world": list(map(float, left)),
            "superior_world": list(map(float, superior)),
        },
    }


def _full_camera_metadata(camera) -> dict:
    parallel = bool(camera.GetParallelProjection())

    return {
        "position": list(map(float, camera.GetPosition())),
        "focal_point": list(map(float, camera.GetFocalPoint())),
        "view_up": list(map(float, camera.GetViewUp())),
        "direction_of_projection": list(
            map(float, camera.GetDirectionOfProjection())
        ),
        "clipping_range": list(map(float, camera.GetClippingRange())),
        "parallel_projection": parallel,
        "parallel_scale": (
            float(camera.GetParallelScale()) if parallel else None
        ),
        "view_angle": float(camera.GetViewAngle()),
        "distance": float(camera.GetDistance()),
        "roll": float(camera.GetRoll()),
    }


def generate_camera_relative_views(
    raw_path: str,
    dimensions: tuple,
    scalar_type: str = "uint8",
    isovalue: float = 80,
    output_dir: str = "reference_views_relative",
    plan: Optional[List[dict]] = None,
    *,
    root_camera: RootCameraCalibration = ROOT_CAMERA,
    canonicalize_each_view: bool = False,
    description_settings: ViewDescriptionSettings = VIEW_DESCRIPTION_SETTINGS,
) -> List[dict]:
    """Render camera-relative graph nodes, metadata, and deterministic labels.

    ``canonicalize_each_view=False``
        Preserve the exact camera-relative pose produced by restoring the parent
        and applying one elementary VTK action. Use this when graph edges must
        correspond exactly to replayable VTK actions.

    ``canonicalize_each_view=True``
        After the elementary action, remove in-plane roll by aligning image top
        with ``root_camera.object_up_world``. This is normally more robust for
        image embeddings and vision-language models.

        In that mode, each edge is explicitly:
            restore parent -> elementary action -> ViewUp canonicalization

    The reference description is independent of roll. It is generated from the
    camera position relative to the calibrated object axes.

    Output:
        {output_dir}/view_000.png ...
        {output_dir}/camera_nodes.json
        {output_dir}/reference_views.json
        {output_dir}/generation_config.json
    """
    selected_plan = plan if plan is not None else VIEW_GENERATION_PLAN
    _validate_plan(selected_plan)
    _validate_description_settings(description_settings)

    # Fail early if the object-axis calibration is invalid.
    _orthogonalized_object_axes(root_camera)

    image_data = load_raw_volume(raw_path, dimensions, scalar_type)
    _actor, renderer, render_window = build_isosurface_pipeline(
        image_data,
        isovalue,
    )
    camera = renderer.GetActiveCamera()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_states: Dict[str, dict] = {}
    records: List[dict] = []
    reference_descriptions: List[dict] = []

    for entry in selected_plan:
        node_id = entry["node_id"]
        parent_id = entry["parent_id"]
        action = entry["action"]

        if parent_id is None:
            _apply_root_camera(camera, root_camera)
            roll_canonicalized = True
        else:
            set_camera_state(camera, saved_states[parent_id])
            _apply_elementary_action(camera, action)

            roll_canonicalized = False
            if canonicalize_each_view:
                _set_canonical_view_up(camera, root_camera)
                roll_canonicalized = True

        renderer.ResetCameraClippingRange()
        render_window.Render()

        image_path = out_dir / f"{node_id}.png"
        save_screenshot(render_window, str(image_path))

        saved_states[node_id] = get_camera_state(camera)
        view_classification = _classify_camera_view(
            camera,
            root_camera,
            description_settings,
        )
        description = view_classification["description"]

        record = {
            "node_id": node_id,
            "description": description,
            "parent_id": parent_id,
            "action": action,
            "post_action_view_up_canonicalized": roll_canonicalized,
            "image_path": str(image_path),
            "camera": _full_camera_metadata(camera),
            "view_classification": view_classification,
        }
        records.append(record)

        # This compact file matches the simple reference-item structure expected
        # by downstream prompts and retrieval code.
        reference_descriptions.append(
            {
                "node_id": node_id,
                "description": description,
            }
        )

        print(
            "[camera_relative_views] "
            f"Saved {node_id}: {description} "
            f"(parent={parent_id}, action={action}, "
            f"canonicalized={roll_canonicalized}) -> {image_path}"
        )

    nodes_path = out_dir / "camera_nodes.json"
    with nodes_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    references_path = out_dir / "reference_views.json"
    with references_path.open("w", encoding="utf-8") as file:
        json.dump(reference_descriptions, file, indent=2)

    config = {
        "root_camera": asdict(root_camera),
        "description_settings": asdict(description_settings),
        "canonicalize_each_view": canonicalize_each_view,
        "node_count": len(records),
    }
    config_path = out_dir / "generation_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(
        f"[camera_relative_views] Wrote {len(records)} records to {nodes_path}"
    )
    print(
        f"[camera_relative_views] Wrote compact descriptions to {references_path}"
    )
    print(
        f"[camera_relative_views] Wrote generation config to {config_path}"
    )

    return records