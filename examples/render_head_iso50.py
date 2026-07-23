import os

import vtk

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vis_male_128x256x256_uint8.raw")
DIMENSIONS = (128, 256, 256)
SPACING = (1.57774, 0.995861, 1.00797)
ISOVALUE = 150

reader = vtk.vtkImageReader2()
reader.SetFileName(DATASET_PATH)
reader.SetDataScalarTypeToUnsignedChar()
reader.SetNumberOfScalarComponents(1)
reader.SetFileDimensionality(3)
reader.SetDataExtent(0, DIMENSIONS[0] - 1, 0, DIMENSIONS[1] - 1, 0, DIMENSIONS[2] - 1)
reader.SetDataSpacing(*SPACING)
reader.SetDataByteOrderToLittleEndian()
reader.Update()

surface = vtk.vtkFlyingEdges3D()
surface.SetInputConnection(reader.GetOutputPort())
surface.SetValue(0, ISOVALUE)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(surface.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)
renderer.ResetCamera()

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

render_window.Render()
interactor.Start()
