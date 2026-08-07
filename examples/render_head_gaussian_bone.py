"""Direct volume rendering of the Visible Male head (vis_male_128x256x256_uint8.raw) where
soft tissue/skin uses the usual flat opacity ramp, but BONE uses a Gaussian opacity function
of intensity instead -- opacity(x) = GAUSSIAN_PEAK_OPACITY * exp(-0.5 * ((x - GAUSSIAN_MEAN)
/ GAUSSIAN_SIGMA) ** 2).

Why Gaussian instead of the plateau in data/vis_male_transfer_function.json (skull preset,
flat opacity=1.0 from ~110 up through 254): a plateau treats every bone-range intensity as
equally solid, so denser cortical bone and fainter/thinner bone read identically. A Gaussian
bump instead peaks at one target density (GAUSSIAN_MEAN) and fades opacity for voxels further
from it in either direction, so it acts as a soft band-pass filter over intensity -- useful
for emphasizing one bone density band without a hard cutoff. See
data/vis_male_transfer_function.json's notes for the underlying histogram (bone climbs from
~90 up through the low 200s for this dataset).

The Gaussian is not a single vtkPiecewiseFunction call -- VTK's opacity function is a
piecewise-linear interpolation between explicit points, so the curve is approximated by
sampling exp(...) at fine, regular intensity steps (GAUSSIAN_SAMPLE_STEP) across the bone
range and adding each as its own point.

Run: python examples/render_head_gaussian_bone.py
Opens an interactive window (drag to orbit) and also saves a screenshot to
output/render_head_gaussian_bone.png before the window opens.
"""

import math
import os

import vtk

# --- Dataset ------------------------------------------------------------
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_128x256x256_uint8.raw"
)
VOLUME_DIMENSIONS = (128, 256, 256)
VOLUME_SPACING = (1.57774, 0.995861, 1.00797)
VOLUME_ORIGIN = (0.0, 0.0, 0.0)

# --- Optional smoothing (denoise before classification) ------------------
ENABLE_SMOOTHING = True
GAUSSIAN_SMOOTH_STANDARD_DEVIATION = 1.0
GAUSSIAN_SMOOTH_RADIUS_FACTOR = 2.0

ENABLE_SHADING = True

# --- Skin/soft-tissue opacity (flat ramp, same idea as vis_male_head_transfer_function.json)
SKIN_OPACITY_POINTS = [
    (0, 0.00),
    (30, 0.00),
    (50, 0.05),
    (70, 0.12),
]
SKIN_COLOR_POINTS = [
    (0, 0.0, 0.0, 0.0),
    (40, 0.4, 0.2, 0.15),
    (70, 0.93, 0.76, 0.65),
]

# --- Bone opacity: Gaussian function of intensity, not a flat ramp -------
GAUSSIAN_MEAN = 150.0          # intensity where bone opacity peaks
GAUSSIAN_SIGMA = 35.0          # controls how quickly opacity fades away from the mean
GAUSSIAN_PEAK_OPACITY = 0.85   # opacity at x == GAUSSIAN_MEAN
GAUSSIAN_RANGE = (71, 254)     # intensities the Gaussian is sampled over (picks up where skin leaves off)
GAUSSIAN_SAMPLE_STEP = 2       # smaller = smoother curve, more vtkPiecewiseFunction points

# Bone color: transitions from skin tone into an ivory/off-white bone tone around the peak
BONE_COLOR_POINTS = [
    (GAUSSIAN_MEAN - GAUSSIAN_SIGMA, 0.85, 0.55, 0.45),
    (GAUSSIAN_MEAN, 0.95, 0.92, 0.85),
    (254, 0.95, 0.92, 0.85),
]

# --- Render mode ("volume" always here) + camera --------------------------
OUTPUT_SCREENSHOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "render_head_gaussian_bone.png"
)


def gaussian_opacity_samples(low, high, step, mean, sigma, peak_opacity):
    """Sample opacity(x) = peak_opacity * exp(-0.5 * ((x - mean) / sigma) ** 2) at regular
    steps across [low, high] -- approximates a smooth Gaussian bump as a series of
    piecewise-linear points, since vtkPiecewiseFunction has no native Gaussian primitive."""
    points = []
    x = low
    while x < high:
        opacity = peak_opacity * math.exp(-0.5 * ((x - mean) / sigma) ** 2)
        points.append((x, round(opacity, 4)))
        x += step
    opacity_at_high = peak_opacity * math.exp(-0.5 * ((high - mean) / sigma) ** 2)
    points.append((high, round(opacity_at_high, 4)))
    return points


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
    """Returns the upstream algorithm (reader or smoother), kept alive by the caller -- VTK's
    Python bindings don't keep a producer alive from its output port alone, so dropping this
    reference crashes the mapper later."""
    if not ENABLE_SMOOTHING:
        return reader

    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputConnection(reader.GetOutputPort())
    smoother.SetStandardDeviation(GAUSSIAN_SMOOTH_STANDARD_DEVIATION)
    smoother.SetRadiusFactor(GAUSSIAN_SMOOTH_RADIUS_FACTOR)
    smoother.Update()
    return smoother


def create_volume(image_source):
    opacity_points = SKIN_OPACITY_POINTS + gaussian_opacity_samples(
        GAUSSIAN_RANGE[0], GAUSSIAN_RANGE[1], GAUSSIAN_SAMPLE_STEP,
        GAUSSIAN_MEAN, GAUSSIAN_SIGMA, GAUSSIAN_PEAK_OPACITY,
    )
    color_points = SKIN_COLOR_POINTS + BONE_COLOR_POINTS

    opacity_function = vtk.vtkPiecewiseFunction()
    for scalar, opacity in opacity_points:
        opacity_function.AddPoint(scalar, opacity)

    color_function = vtk.vtkColorTransferFunction()
    for scalar, r, g, b in color_points:
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


def set_front_facing_camera(renderer, prop):
    """Point the camera at prop's front (face) side, framed upright -- same recipe verified
    by eye in examples/render_skull_transfer_function.py (renderer.ResetCamera() alone gives
    an unhelpful bottom-up/inferior view for this dataset)."""
    bounds = prop.GetBounds()
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    diag = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 3

    camera = renderer.GetActiveCamera()
    camera.SetPosition(center[0], center[1] + diag, center[2])
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, -1)
    renderer.ResetCameraClippingRange()


def save_screenshot(render_window, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


def main():
    reader = load_raw_volume(DATASET_PATH)
    image_source = apply_optional_smoothing(reader)  # must stay referenced

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.0, 0.0, 0.0)

    volume = create_volume(image_source)
    renderer.AddVolume(volume)

    set_front_facing_camera(renderer, volume)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(900, 900)

    render_window.Render()
    save_screenshot(render_window, OUTPUT_SCREENSHOT_PATH)
    print(f"Screenshot saved to: {OUTPUT_SCREENSHOT_PATH}")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    interactor.Start()


if __name__ == "__main__":
    main()
