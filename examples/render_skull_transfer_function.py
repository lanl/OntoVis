"""Rendering of the Visible Male head RAW dataset with VTK -- either direct volume
rendering (a color/opacity transfer function, every voxel contributes to the image) or
isosurface extraction (a single scalar threshold, marching cubes traces one solid surface).

Set RENDER_MODE below to switch between the two:
  "volume"     -- classify intensities through a transfer function (TRANSFER_FUNCTION_PRESET
                  picks "skull" (bone only) or "head" (skin/soft-tissue surface, a moderate
                  opacity ramp with a skin tone)). Thin bone can appear translucent instead
                  of being included/excluded by a single threshold.
  "isosurface" -- no opacity/color setting at all: vtkFlyingEdges3D traces ONE surface at
                  ISOVALUE, exactly like camera_reasoning/volume_scene.py's
                  build_isosurface_actor. ISOVALUE=40 sits right at the background/soft-tissue
                  boundary noted in vis_male_transfer_function.json ("soft-tissue peak sits
                  around 60-80"), i.e. close to the air-to-skin transition -- low enough that
                  the traced surface should be the skin's outer boundary itself, not bone.
"""

import json
import os

import vtk

# --- Dataset ----------------------------------------------------------------
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_128x256x256_uint8.raw"
)
VOLUME_DIMENSIONS = (128, 256, 256)
VOLUME_SPACING = (1.57774, 0.995861, 1.00797)
VOLUME_ORIGIN = (0.0, 0.0, 0.0)

# --- Optional smoothing (denoise before classification) ---------------------
ENABLE_SMOOTHING = True
GAUSSIAN_STANDARD_DEVIATION = 1.0
GAUSSIAN_RADIUS_FACTOR = 2.0

# ShadeOff() gives a flat, uniformly-colored look with no lighting falloff --
# useful to actually see a pure-white color transfer function as white,
# instead of shaded gray in places facing away from the light.
ENABLE_SHADING = True

# --- Render mode --------------------------------------------------------
RENDER_MODE = "isosurface"  # "volume" (transfer function) or "isosurface" (single threshold)
ISOVALUE = 40  # only used when RENDER_MODE == "isosurface"

# --- Transfer function control points (RENDER_MODE == "volume" only) ----
# Loaded from data/vis_male_<preset>_transfer_function.json (kept next to the
# raw dataset it was tuned for) instead of hardcoded here, so tuning it in one
# place (this script, camera_reasoning's CameraReasoningSession, etc.) keeps
# every user of this dataset in sync. See examples/show_hist.py to inspect a
# dataset's histogram before tuning new points for a different dataset.
TRANSFER_FUNCTION_PRESET = "head"  # "skull" (bone only) or "head" (skin/soft-tissue surface)

_TRANSFER_FUNCTION_FILENAMES = {
    "skull": "vis_male_transfer_function.json",
    "head": "vis_male_head_transfer_function.json",
}
TRANSFER_FUNCTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    _TRANSFER_FUNCTION_FILENAMES[TRANSFER_FUNCTION_PRESET],
)


def load_transfer_function_json(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Transfer function file not found at: {file_path}")

    with open(file_path) as f:
        spec = json.load(f)

    try:
        opacity_points = [tuple(point) for point in spec["opacity_points"]]
        color_points = [tuple(point) for point in spec["color_points"]]
    except KeyError as error:
        raise ValueError(f"Transfer function file {file_path} is missing key {error}.") from error

    return opacity_points, color_points


if RENDER_MODE == "volume":
    OPACITY_POINTS, COLOR_POINTS = load_transfer_function_json(TRANSFER_FUNCTION_PATH)


def load_raw_volume(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"RAW dataset not found at: {file_path}")

    reader = vtk.vtkImageReader2()
    reader.SetFileName(file_path)
    reader.SetDataScalarTypeToUnsignedChar()
    reader.SetNumberOfScalarComponents(1)
    reader.SetFileDimensionality(3)
    reader.SetDataExtent(0, VOLUME_DIMENSIONS[0] - 1, 0, VOLUME_DIMENSIONS[1] - 1, 0, VOLUME_DIMENSIONS[2] - 1)
    reader.SetDataSpacing(*VOLUME_SPACING)
    reader.SetDataOrigin(*VOLUME_ORIGIN)
    reader.SetDataByteOrderToLittleEndian()
    reader.Update()
    return reader


def apply_optional_smoothing(reader):
    """Returns the upstream algorithm (reader or smoother), kept alive by the
    caller -- VTK's Python bindings don't keep a producer alive from its
    output port alone, so dropping this reference crashes the mapper later."""
    if not ENABLE_SMOOTHING:
        return reader

    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputConnection(reader.GetOutputPort())
    smoother.SetStandardDeviation(GAUSSIAN_STANDARD_DEVIATION)
    smoother.SetRadiusFactor(GAUSSIAN_RADIUS_FACTOR)
    smoother.Update()
    return smoother


def create_volume(image_source):
    opacity_function = vtk.vtkPiecewiseFunction()
    for scalar, opacity in OPACITY_POINTS:
        opacity_function.AddPoint(scalar, opacity)

    color_function = vtk.vtkColorTransferFunction()
    for scalar, r, g, b in COLOR_POINTS:
        color_function.AddRGBPoint(scalar, r, g, b)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_function)
    volume_property.SetScalarOpacity(opacity_function)
    volume_property.SetInterpolationTypeToLinear()
    if ENABLE_SHADING:
        volume_property.ShadeOn()
    else:
        volume_property.ShadeOff()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputConnection(image_source.GetOutputPort())

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    return volume


def create_isosurface_actor(image_source):
    """Trace ONE surface at ISOVALUE via marching cubes -- no opacity/color transfer
    function at all, unlike create_volume(). Mirrors
    camera_reasoning/volume_scene.py's build_isosurface_actor."""
    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputConnection(image_source.GetOutputPort())
    surface.SetValue(0, ISOVALUE)
    surface.ComputeNormalsOn()
    surface.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.75, 0.65)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    return actor


def set_front_facing_camera(renderer, prop):
    """Point the camera at prop's (the volume or isosurface actor's) front (face) side,
    framed upright.

    renderer.ResetCamera() alone gives an unhelpful bottom-up/inferior view for THIS
    dataset (vis_male_128x256x256_uint8.raw) -- verified by rendering both this preset and
    the skull-only preset from ResetCamera()'s default position and finding the same odd
    angle in both, and then by testing camera positions along each axis directly. This
    (center + diag*Y, view_up=-Z) direction is the one confirmed by eye to show the face.
    """
    bounds = prop.GetBounds()
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    diag = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 3

    camera = renderer.GetActiveCamera()
    camera.SetPosition(center[0], center[1] + diag, center[2])
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, -1)
    renderer.ResetCameraClippingRange()


def main():
    reader = load_raw_volume(DATASET_PATH)
    image_source = apply_optional_smoothing(reader)  # must stay referenced

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)

    if RENDER_MODE == "isosurface":
        prop = create_isosurface_actor(image_source)
        renderer.AddActor(prop)
    else:
        prop = create_volume(image_source)
        renderer.AddVolume(prop)

    set_front_facing_camera(renderer, prop)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(900, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
