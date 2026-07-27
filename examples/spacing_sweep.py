"""Diagnostic: render the SAME front-facing isosurface view of
vis_male_128x256x256_uint8.raw under several candidate voxel spacings, tiled side by side
-- so which spacing (if any) actually removes the squeeze along the 128-slice axis can be
judged by eye, instead of trusting one hardcoded value blindly.

Uses the same "front facing camera" recipe already verified by eye in
render_skull_transfer_function.py (center + diag*Y, view_up=-Z) -- centered on each
candidate's OWN actor bounds, so every tile stays front-on regardless of how that
candidate's spacing reshapes those bounds.

Run: python examples/spacing_sweep.py
Output: output/spacing_sweep/spacing_sweep_grid.png (one labeled tile per candidate)
"""
import math
import os
from pathlib import Path

import vtk
from PIL import Image as PILImage, ImageDraw

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_128x256x256_uint8.raw")
DIMENSIONS = (128, 256, 256)
ISOVALUE = 40
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "spacing_sweep")

# Candidates to compare -- not just "is the documented value right", but also whether the
# squeeze might actually be an axis-order mixup (swapped_*) rather than a magnitude issue.
SPACING_CANDIDATES = {
    "isotropic_1_1_1": (1.0, 1.0, 1.0),
    "documented_1.578_0.996_1.008": (1.57774, 0.995861, 1.00797),
    "naive_exact_2x": (2.0, 1.0, 1.0),
    "swapped_yxz": (0.995861, 1.57774, 1.00797),
    "swapped_zyx": (1.00797, 0.995861, 1.57774),
    "inverse_documented": (1 / 1.57774, 1 / 0.995861, 1 / 1.00797),
}

TILE_SIZE = 500
LABEL_HEIGHT = 30
COLUMNS = 3


def render_one(spacing, output_path):
    reader = vtk.vtkImageReader2()
    reader.SetFileName(DATASET_PATH)
    reader.SetDataScalarTypeToUnsignedChar()
    reader.SetNumberOfScalarComponents(1)
    reader.SetFileDimensionality(3)
    reader.SetDataExtent(0, DIMENSIONS[0] - 1, 0, DIMENSIONS[1] - 1, 0, DIMENSIONS[2] - 1)
    reader.SetDataSpacing(*spacing)
    reader.SetDataByteOrderToLittleEndian()
    reader.Update()

    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputConnection(reader.GetOutputPort())
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

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.05, 0.05, 0.05)
    renderer.AddActor(actor)

    bounds = actor.GetBounds()
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
    diag = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 3

    camera = renderer.GetActiveCamera()
    camera.SetPosition(center[0], center[1] + diag, center[2])
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0, 0, -1)
    renderer.ResetCameraClippingRange()

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(TILE_SIZE, TILE_SIZE)
    render_window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    tile_paths = {}
    for label, spacing in SPACING_CANDIDATES.items():
        path = os.path.join(OUTPUT_DIR, f"{label}.png")
        render_one(spacing, path)
        tile_paths[label] = path
        print(f"rendered {label} spacing={spacing} -> {path}")

    rows = math.ceil(len(tile_paths) / COLUMNS)
    canvas = PILImage.new("RGB", (TILE_SIZE * COLUMNS, (TILE_SIZE + LABEL_HEIGHT) * rows), color=(15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for i, (label, path) in enumerate(tile_paths.items()):
        col, row = i % COLUMNS, i // COLUMNS
        x, y = col * TILE_SIZE, row * (TILE_SIZE + LABEL_HEIGHT)
        draw.rectangle([x, y, x + TILE_SIZE, y + LABEL_HEIGHT], fill=(15, 15, 15))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
        canvas.paste(PILImage.open(path).convert("RGB"), (x, y + LABEL_HEIGHT))

    grid_path = os.path.join(OUTPUT_DIR, "spacing_sweep_grid.png")
    canvas.save(grid_path)
    print(f"\nComparison grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
