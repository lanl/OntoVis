---
definitions:
 - Dataset: A collection of scientific data describing one or more quantities over a spatial or spatial-temporal domain.
 - Field: A quantity contained in a dataset, such as pressure, temperature, density, or velocity.
 - Grid: The spatial structure on which fields are defined.
 - Scalar: A field containing one scalar value per associated grid element.
 - Vector: A field containing multiple components representing a vector quantity.
 - Tensor: A field containing multiple components representing a tensor quantity.
 - Point: A location defined by the grid.
 - Cell: A spatial element defined by a collection of grid points.
 - UniformGrid: A regular grid with uniform spacing.
 - RectilinearGrid: A structured grid with independently varying coordinates along each axis.
 - StructuredGrid:  A grid with regular topology and arbitrary coordinates.
 - UnstructuredGrid: A grid with arbitrary topology and geometry.
 - Static: A dataset containing a single state in time.
 - TimeVarying: A dataset containing multiple states in time.
 - Spacing: For grid datasets, the space between x, y, z with with the dataset was captured
 - Capture_method: the method used to capture the dataset e.g. CT Scan, X Ray, ...
 - scalar_range: the range for each field
 - Description: description of the dataset

---

Dataset <has> [1] Grid
Dataset <has> [1..*] Field
Dataset <has> [1] Description

Dataset <temporal_type> [1] {Static, TimeVarying}

Dataset <has> [0..1] Capture_method

Grid <has> [0..1] Spacing

Grid <grid_type> [1] {
    UniformGrid,
    RectilinearGrid,
    StructuredGrid,
    UnstructuredGrid
}

Grid <spatial_dimensions> [1] {1, 2, 3}

Grid <size_x> [0..1] integer
Grid <size_y> [0..1] integer
Grid <size_z> [0..1] integer

Field <data_type> [1] {Scalar, Vector, Tensor}
Field <association> [1] {Point, Cell}

Field <name> [1] string
Field <components> [1] integer
Field <units> [0..1] string

Field <minimum> [0..1] number
Field <maximum> [0..1] number
