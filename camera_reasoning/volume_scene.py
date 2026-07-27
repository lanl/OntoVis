import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support


def load_transfer_function_json(path) -> tuple:
    """Load (opacity_points, color_points) from a transfer-function JSON file.

    Expects "opacity_points" (list of [scalar, opacity]) and "color_points"
    (list of [scalar, r, g, b]) keys -- see data/vis_male_transfer_function.json
    for a worked example, and build_volume_rendering_pipeline() below for how
    these are consumed. Kept next to the raw dataset it was tuned for, rather
    than hardcoded in Python, so a new object's points never risk overwriting
    a previous one's.
    """
    spec = json.loads(Path(path).read_text())
    opacity_points = [tuple(point) for point in spec["opacity_points"]]
    color_points = [tuple(point) for point in spec["color_points"]]
    return opacity_points, color_points


def load_raw_volume(
    path: str, dimensions: tuple, scalar_type: str = "uint8", spacing: tuple = (1.0, 1.0, 1.0)
) -> vtk.vtkImageData:
    """Read a raw binary volume file into a vtkImageData object.

    `spacing` defaults to isotropic (1.0, 1.0, 1.0) -- pass the dataset's real
    per-axis voxel spacing for anisotropic volumes (e.g. vis_male_128x256x256's
    (1.57774, 0.995861, 1.00797), see data/vis_male_transfer_function.json /
    examples/render_skull_transfer_function.py). Getting this wrong silently
    distorts geometry rather than raising an error -- e.g. defaulting to (1,1,1)
    for that dataset compresses its 128-slice axis to half the width of the
    256-slice axes, since 128*1.0 is half of 256*1.0 even though the real
    physical spacing (1.57774 vs ~1.0) was meant to compensate for it.
    """
    dx, dy, dz = dimensions
    dtype = np.dtype(scalar_type)
    data = np.fromfile(path, dtype=dtype)
    expected = dx * dy * dz
    if data.size != expected:
        raise ValueError(
            f"Raw file has {data.size} values but dimensions {dimensions} require {expected}."
        )
    # VTK expects Fortran-order (x varies fastest) — raw volumes are typically C-order
    data = data.reshape((dz, dy, dx))

    image = vtk.vtkImageData()
    image.SetDimensions(dx, dy, dz)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(*spacing)

    vtk_array = numpy_support.numpy_to_vtk(
        data.ravel(order="C"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image.GetPointData().SetScalars(vtk_array)
    return image


def build_isosurface_actor(image: vtk.vtkImageData, isovalue: float) -> vtk.vtkActor:
    """Build just the isosurface actor for `image` at `isovalue`, with no renderer/window.

    Factored out of build_isosurface_pipeline() so a specialist (e.g. an isovalue-adjustment
    agent) can rebuild the actor in place at a new isovalue and swap it into an existing
    renderer, without disturbing that renderer's camera state -- see
    CameraReasoningSession.set_isovalue().
    """
    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(image)
    mc.SetValue(0, isovalue)
    mc.ComputeNormalsOn()
    mc.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.85, 0.75, 0.65)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    return actor


def build_isosurface_pipeline(image: vtk.vtkImageData, isovalue: float):
    """Return (actor, renderer, render_window) for an isosurface render."""
    actor = build_isosurface_actor(image, isovalue)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    render_window.SetOffScreenRendering(1)

    return actor, renderer, render_window


def build_volume_actor(
    image: vtk.vtkImageData,
    opacity_points: list,
    color_points: list,
    enable_smoothing: bool = False,
    gaussian_standard_deviation: float = 1.0,
    gaussian_radius_factor: float = 2.0,
    enable_shading: bool = True,
) -> vtk.vtkVolume:
    """Build just the direct-volume-rendering prop for `image` under the given transfer
    function, with no renderer/window.

    Unlike build_isosurface_actor (single isovalue -> polygonal surface via
    vtkFlyingEdges3D), every voxel is classified through opacity_points and color_points
    transfer functions, so intensity ranges can fade in or out instead of being an in/out
    binary decision at one threshold. This makes it possible to reveal thin/low-density
    structures that a single isovalue would either clip or flood with noise.

    opacity_points: list of (scalar_intensity, opacity) with opacity in [0, 1].
    color_points: list of (scalar_intensity, r, g, b) with each in [0, 1].

    Factored out of build_volume_rendering_pipeline() so a specialist (e.g. an isovalue
    agent deriving an opacity ramp from the volume's own histogram) can rebuild the volume
    prop in place under a new transfer function and swap it into an existing renderer,
    without disturbing that renderer's camera state -- see
    CameraReasoningSession.set_transfer_function().

    enable_shading: when True (default), surfaces facing away from the light shade darker
    regardless of color_points -- gives depth cues but can make a pure-white color transfer
    function still look gray in places. Set False for a flat, uniformly-colored look with
    no lighting falloff at all.
    """
    source_image = image
    if enable_smoothing:
        smoother = vtk.vtkImageGaussianSmooth()
        smoother.SetInputData(image)
        smoother.SetStandardDeviation(gaussian_standard_deviation)
        smoother.SetRadiusFactor(gaussian_radius_factor)
        smoother.Update()
        source_image = smoother.GetOutput()

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
    if enable_shading:
        volume_property.ShadeOn()
    else:
        volume_property.ShadeOff()

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(source_image)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volume_property)
    return volume


def build_volume_rendering_pipeline(
    image: vtk.vtkImageData,
    opacity_points: list,
    color_points: list,
    enable_smoothing: bool = False,
    gaussian_standard_deviation: float = 1.0,
    gaussian_radius_factor: float = 2.0,
    enable_shading: bool = True,
):
    """Return (volume, renderer, render_window) for a direct volume render. See
    build_volume_actor for the transfer-function/smoothing/shading details -- this just
    wraps it in its own renderer/window.

    There is no automatic derivation of opacity_points/color_points from the volume's
    histogram in THIS function -- callers choose them per dataset (see
    examples/render_skull_transfer_function.py for a worked example,
    examples/show_hist.py for inspecting a volume's histogram first, and
    visualization_orchestrator/specialists/isovalue_adapter.py's
    build_opacity_ramp_for_band for an automatic, histogram-derived alternative).
    """
    volume = build_volume_actor(
        image, opacity_points, color_points,
        enable_smoothing=enable_smoothing,
        gaussian_standard_deviation=gaussian_standard_deviation,
        gaussian_radius_factor=gaussian_radius_factor,
        enable_shading=enable_shading,
    )

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    render_window.SetOffScreenRendering(1)

    return volume, renderer, render_window


def save_screenshot(render_window: vtk.vtkRenderWindow, path: str):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    render_window.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
