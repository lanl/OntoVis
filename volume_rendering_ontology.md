---
definitions:
 - VolumeRendering: A family of techniques for visualizing 3D scalar or vector fields by displaying samples of the volumetric data directly, without first converting it to an explicit surface representation.
 - DirectVolumeRendering: An approach that renders a volume by compositing contributions from samples throughout the 3D domain, using a transfer function to map data values to color and opacity.
 - IndirectVolumeRendering: An approach that first extracts an explicit geometric surface from the volume and then renders that surface with standard polygon rendering.
 - RayCasting: A direct volume rendering algorithm that casts a ray from the camera through each pixel, samples the volume along the ray, and composites the samples into a final pixel color.
 - Splatting: A direct volume rendering algorithm that projects each voxel onto the image plane as a reconstruction kernel ("splat") and accumulates overlapping splats.
 - TextureBasedRendering: A direct volume rendering algorithm that resamples the volume onto textured slices rendered and blended by graphics hardware.
 - ShearWarpRendering: A direct volume rendering algorithm that shears the volume into an object space where viewing rays are axis-aligned, enabling fast slice-based compositing before warping the result into image space.
 - IsosurfaceExtraction: An indirect volume rendering technique that extracts a polygonal surface representing all points in the volume equal to a given scalar threshold.
 - MarchingCubes: An isosurface extraction algorithm that processes the volume cell by cell, generating triangles based on which cell corners lie above or below the threshold.
 - TransferFunction: A mapping from scalar data values to visual properties, typically color and opacity, that controls what is seen in a direct volume rendering.
 - ColorMapping: The component of a transfer function that assigns a color to each data value.
 - OpacityMapping: The component of a transfer function that assigns an opacity to each data value, controlling the visibility of structures.
 - PreIntegratedTransferFunction: An optimized transfer function evaluation that precomputes the color/opacity integral between consecutive sample pairs to reduce sampling artifacts.
 - Sampling: The process of evaluating the volume's scalar field at discrete points along a viewing ray.
 - Interpolation: The method used to estimate a scalar value at a non-grid-aligned sample point, such as trilinear interpolation.
 - Compositing: The process of accumulating color and opacity contributions from samples along a ray into a single pixel value.
 - FrontToBackCompositing: A compositing order that accumulates samples starting nearest the camera, enabling early ray termination.
 - BackToFrontCompositing: A compositing order that accumulates samples starting farthest from the camera.
 - AlphaBlending: The standard compositing operator that blends a new sample's color and opacity into the accumulated result using the "over" operator.
 - MaximumIntensityProjection: A compositing method that selects the maximum data value along each ray instead of blending opacity-weighted contributions.
 - GradientEstimation: The computation of the local gradient of the scalar field, used to estimate surface normals for shading.
 - CentralDifferenceGradient: A gradient estimation method that approximates the gradient using finite differences between neighboring samples.
 - Shading: The application of a lighting model to volume samples, using estimated normals to enhance depth and structure perception.
 - PhongShading: A local illumination model combining ambient, diffuse, and specular reflection terms, commonly applied in volume shading.
 - AmbientOcclusion: A shading technique that approximates soft shadowing from nearby occluding structures within the volume to improve depth perception.
 - EmptySpaceSkipping: An acceleration technique that skips regions of the volume known to be empty or fully transparent.
 - EarlyRayTermination: An acceleration technique that stops sampling along a ray once accumulated opacity reaches full occlusion.
 - SpatialAccelerationStructure: A data structure that organizes the volume to quickly identify and skip empty or irrelevant regions.
 - Octree: A hierarchical spatial acceleration structure that recursively subdivides the volume into eight octants.
 - GPUAcceleration: The use of graphics hardware, such as fragment or compute shaders, to parallelize sampling, compositing, and shading.
 - LevelOfDetail: A technique that renders coarser representations of the volume at a distance or during interaction, trading quality for performance.
---

VolumeRendering <hasApproach> [1..*] {
    DirectVolumeRendering,
    IndirectVolumeRendering
}

DirectVolumeRendering <includesAlgorithm> [1..*] {
    RayCasting,
    Splatting,
    TextureBasedRendering,
    ShearWarpRendering
}

IndirectVolumeRendering <includesAlgorithm> [1..*] { IsosurfaceExtraction }

IsosurfaceExtraction <implementedBy> [1..*] { MarchingCubes }

RayCasting <uses> [1] TransferFunction
RayCasting <uses> [1] Sampling
RayCasting <uses> [1] Compositing
RayCasting <uses> [0..1] GradientEstimation

RayCasting <acceleratedBy> [0..*] {
    EmptySpaceSkipping,
    EarlyRayTermination,
    SpatialAccelerationStructure,
    GPUAcceleration
}

Splatting <uses> [1] TransferFunction
TextureBasedRendering <uses> [1] TransferFunction
ShearWarpRendering <uses> [1] TransferFunction

Sampling <uses> [1] Interpolation

TransferFunction <has> [1] ColorMapping
TransferFunction <has> [1] OpacityMapping
TransferFunction <variant> [0..1] PreIntegratedTransferFunction

Compositing <order> [1] { FrontToBackCompositing, BackToFrontCompositing }
Compositing <operator> [1] { AlphaBlending, MaximumIntensityProjection }

GradientEstimation <method> [1] { CentralDifferenceGradient }
GradientEstimation <enables> [1] Shading

Shading <model> [1..*] { PhongShading, AmbientOcclusion }

SpatialAccelerationStructure <implementedBy> [1..*] { Octree }

GPUAcceleration <enables> [0..1] LevelOfDetail
