"""Direct volume rendering of the Visible Male head RAW dataset with VTK.

Loads the raw uint8 volume, classifies intensities through a color/opacity
transfer function, and displays it in an interactive render window. No
isosurface extraction -- every voxel contributes to the image according to
the transfer function, so thin bone can appear translucent instead of being
included/excluded by a single threshold.
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

# --- Transfer function control points -----------------------------------
# Loaded from data/vis_male_transfer_function.json (kept next to the raw
# dataset it was tuned for) instead of hardcoded here, so tuning it in one
# place (this script, camera_reasoning's CameraReasoningSession, etc.) keeps
# every user of this dataset in sync. See examples/show_hist.py to inspect a
# dataset's histogram before tuning new points for a different dataset.
TRANSFER_FUNCTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_transfer_function.json"
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


def main():
    reader = load_raw_volume(DATASET_PATH)
    image_source = apply_optional_smoothing(reader)  # must stay referenced
    volume = create_volume(image_source)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

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
