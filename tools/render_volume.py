#!/usr/bin/env python3
"""
Render a volume from a specific camera angle.

This tool allows you to render a volume dataset from any viewpoint by specifying
the azimuth (horizontal rotation), elevation (vertical angle), and distance.

Usage:
    python tools/render_volume.py data/foot.raw --azimuth 45 --elevation 30 --distance 2.0
    python tools/render_volume.py data/foot.raw -a 0 -e 90 -r 1.5 -o top_view.png
    python tools/render_volume.py data/foot.raw --dimensions 256,256,256 --dtype uint8
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def spherical_to_cartesian(azimuth_deg, elevation_deg, radius):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        azimuth_deg: Azimuth angle in degrees (0-360)
                     0° = +X axis, 90° = +Y axis
        elevation_deg: Elevation angle in degrees (-90 to 90)
                      0° = XY plane, 90° = +Z axis
        radius: Distance from origin

    Returns:
        tuple: (x, y, z) Cartesian coordinates
    """
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    return x, y, z


def load_volume(volume_path, metadata=None):
    """
    Load volume data from file.

    Args:
        volume_path: Path to volume file
        metadata: Dict with 'dimensions' and 'dtype' for .raw files

    Returns:
        numpy.ndarray: Volume data
    """
    path = Path(volume_path)
    ext = path.suffix.lower()

    if ext == '.raw':
        if not metadata or 'dimensions' not in metadata:
            raise ValueError("For .raw files, metadata must include 'dimensions'")

        dims = metadata['dimensions']
        dtype = metadata.get('dtype', 'uint8')
        np_dtype = getattr(np, dtype)

        volume_data = np.fromfile(volume_path, dtype=np_dtype)
        volume_data = volume_data.reshape(dims)

    elif ext == '.npy':
        volume_data = np.load(volume_path)

    elif ext == '.dat':
        import struct
        with open(volume_path, 'rb') as f:
            dims = struct.unpack('III', f.read(12))
            dtype = metadata.get('dtype', 'uint8') if metadata else 'uint8'
            np_dtype = getattr(np, dtype)
            volume_data = np.fromfile(f, dtype=np_dtype)
            volume_data = volume_data.reshape(dims)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return volume_data


def render_volume(volume_path, azimuth, elevation, distance,
                  output_path=None, metadata=None, window_size=(800, 800),
                  background_color='black', show_axes=False, verbose=False, roll=0):
    """
    Render a volume from a specific camera angle.

    Args:
        volume_path: Path to volume file
        azimuth: Azimuth angle in degrees (0-360)
        elevation: Elevation angle in degrees (-90 to 90)
        distance: Camera distance from volume center (multiplier, typically 1.0-3.0)
        output_path: Path to save rendered image (optional)
        metadata: Dict with 'dimensions' and 'dtype' for .raw files
        window_size: Tuple of (width, height) for render window
        background_color: Background color ('white', 'black', or RGB tuple)
        show_axes: Whether to show coordinate axes
        verbose: Print detailed statistics
        roll: Camera roll angle in degrees - rotates around viewing axis (positive=clockwise)

    Returns:
        numpy.ndarray: Rendered image as RGB array
    """
    # Load volume
    print(f"Loading volume from {volume_path}...")
    volume_data = load_volume(volume_path, metadata)
    print(f"Volume shape: {volume_data.shape}")
    print(f"Value range: {volume_data.min()} to {volume_data.max()}")

    if verbose:
        print(f"\nVolume Statistics:")
        print(f"  Mean: {volume_data.mean():.2f}")
        print(f"  Median: {np.median(volume_data):.2f}")
        print(f"  Std Dev: {volume_data.std():.2f}")
        print(f"  Percentiles:")
        for p in [1, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(volume_data, p):.2f}")

    # Create PyVista volume
    grid = pv.ImageData()
    grid.dimensions = np.array(volume_data.shape) + 1
    grid.cell_data["values"] = volume_data.flatten(order="F")

    # Create plotter
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = background_color

    # Add volume with inverted grayscale colormap
    print("Rendering volume...")

    # Use a simple linear opacity that PyVista understands
    # Inverted colormap: high values appear bright (like X-ray)
    volume_actor = plotter.add_volume(
        grid,
        scalars="values",
        opacity="linear",  # Simple linear ramp from transparent to opaque
        cmap='gray',  # Inverted: bright structures on dark background
        shade=True,
        show_scalar_bar=False
    )

    # Set lighting
    volume_actor.prop.ambient = 0.2
    volume_actor.prop.diffuse = 0.7
    volume_actor.prop.specular = 0.3

    # Set camera position
    center = np.array(volume_data.shape) / 2

    # Calculate camera position relative to volume center
    # Scale distance by max dimension for consistent behavior across volumes
    max_dim = max(volume_data.shape)
    scaled_distance = distance * max_dim

    cam_x, cam_y, cam_z = spherical_to_cartesian(azimuth, elevation, scaled_distance)
    camera_position = center + np.array([cam_x, cam_y, cam_z])

    # Calculate proper up vector with roll
    # The up vector is perpendicular to the viewing direction and rotated by roll angle

    # Viewing direction (from camera to center)
    view_dir = center - camera_position
    view_dir = view_dir / np.linalg.norm(view_dir)  # Normalize

    # Default up vector (Z-axis)
    base_up = np.array([0.0, 0.0, 1.0])

    # Make sure base_up is perpendicular to view_dir
    # If view_dir is parallel to Z, use Y as base_up
    if abs(np.dot(view_dir, base_up)) > 0.99:
        base_up = np.array([0.0, 1.0, 0.0])

    # Calculate right vector (perpendicular to both view_dir and base_up)
    right = np.cross(base_up, view_dir)
    right = right / np.linalg.norm(right)

    # Calculate corrected up vector (perpendicular to view_dir and right)
    up = np.cross(view_dir, right)
    up = up / np.linalg.norm(up)

    # Apply roll rotation around the viewing axis using Rodrigues' rotation formula
    if roll != 0:
        roll_rad = np.radians(roll)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)

        # Rotate up vector around view_dir axis
        # Rodrigues' formula: v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
        # where k is the axis (view_dir), v is the vector to rotate (up)
        k_dot_v = np.dot(view_dir, up)
        k_cross_v = np.cross(view_dir, up)

        up = up * cos_roll + k_cross_v * sin_roll + view_dir * k_dot_v * (1 - cos_roll)
        up = up / np.linalg.norm(up)  # Normalize

    plotter.camera_position = [
        tuple(camera_position),
        tuple(center),
        tuple(up)
    ]

    # Add axes if requested
    if show_axes:
        plotter.add_axes()

    # Render
    plotter.show(auto_close=False)

    # Get screenshot
    img_array = plotter.screenshot(return_img=True)
    plotter.close()

    # Save if output path specified
    if output_path:
        from PIL import Image
        img = Image.fromarray(img_array)
        img.save(output_path)
        print(f"Saved render to {output_path}")

    return img_array


def main():
    parser = argparse.ArgumentParser(
        description="Render a volume from a specific camera angle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Front view of foot
  %(prog)s data/foot.raw --azimuth 0 --elevation 0 --distance 2.0

  # Top-down view
  %(prog)s data/foot.raw --azimuth 0 --elevation 90 --distance 1.5

  # Diagonal view (common for 3D visualization)
  %(prog)s data/foot.raw --azimuth 45 --elevation 30 --distance 2.5

  # Save to specific file
  %(prog)s data/foot.raw -a 180 -e 0 -r 2.0 -o back_view.png

  # Specify dimensions for .raw file
  %(prog)s data/foot.raw -a 45 -e 30 -r 2.0 --dimensions 256,256,256 --dtype uint8

Camera Coordinate System:
  Azimuth: Horizontal rotation (0-360°)
    0°   = View from +X axis (front)
    90°  = View from +Y axis (right side)
    180° = View from -X axis (back)
    270° = View from -Y axis (left side)

  Elevation: Vertical angle (-90 to 90°)
    90°  = View from top (+Z axis)
    0°   = View from side (XY plane)
    -90° = View from bottom (-Z axis)

  Distance: Camera distance multiplier (1.0-3.0 typical)
    1.0  = Close-up
    2.0  = Standard view (recommended)
    3.0  = Far view
        """
    )

    parser.add_argument(
        'volume_path',
        help='Path to volume file (.raw, .npy, .dat)'
    )

    parser.add_argument(
        '--azimuth', '-a',
        type=float,
        default=45.0,
        help='Azimuth angle in degrees (0-360, default: 45)'
    )

    parser.add_argument(
        '--elevation', '-e',
        type=float,
        default=30.0,
        help='Elevation angle in degrees (-90 to 90, default: 30)'
    )

    parser.add_argument(
        '--distance', '-r',
        type=float,
        default=2.0,
        help='Camera distance multiplier (default: 2.0)'
    )

    parser.add_argument(
        '--roll',
        type=float,
        default=0.0,
        help='Camera roll in degrees - rotates around viewing axis, positive=clockwise (default: 0)'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output image path (PNG)'
    )

    parser.add_argument(
        '--dimensions',
        help='Volume dimensions for .raw files (e.g., "256,256,256")'
    )

    parser.add_argument(
        '--dtype',
        default='uint8',
        help='Data type for .raw files (default: uint8)'
    )

    parser.add_argument(
        '--size',
        default='800,800',
        help='Window size as "width,height" (default: 800,800)'
    )

    parser.add_argument(
        '--background',
        choices=['white', 'black'],
        default='black',
        help='Background color (default: black)'
    )

    parser.add_argument(
        '--axes',
        action='store_true',
        help='Show coordinate axes'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed volume statistics'
    )

    args = parser.parse_args()

    # Check volume file exists
    volume_path = Path(args.volume_path)
    if not volume_path.exists():
        print(f"❌ Volume file not found: {args.volume_path}")
        sys.exit(1)

    # Prepare metadata for .raw files
    metadata = None
    if volume_path.suffix == '.raw':
        if args.dimensions:
            dims = tuple(map(int, args.dimensions.split(',')))
            metadata = {'dimensions': dims, 'dtype': args.dtype}
        else:
            # Try to infer from filename (e.g., "foot_256x256x256_uint8.raw")
            stem = volume_path.stem
            parts = stem.split('_')
            for part in parts:
                if 'x' in part:
                    try:
                        dims = tuple(map(int, part.split('x')))
                        if len(dims) == 3:
                            dtype_part = parts[parts.index(part) + 1] if parts.index(part) + 1 < len(parts) else 'uint8'
                            metadata = {'dimensions': dims, 'dtype': dtype_part}
                            print(f"ℹ️  Inferred dimensions from filename: {dims}, dtype: {dtype_part}")
                            break
                    except:
                        pass

            if not metadata:
                print("❌ Could not infer dimensions from filename.")
                print("Please provide --dimensions for .raw files")
                print("Example: --dimensions 256,256,256")
                sys.exit(1)

    # Parse window size
    window_size = tuple(map(int, args.size.split(',')))

    # Auto-generate output filename if not specified
    output_path = args.output
    if not output_path:
        output_path = f"{volume_path.stem}_azim{args.azimuth}_elev{args.elevation}.png"

    # Print configuration
    print("="*80)
    print("VOLUME RENDERER")
    print("="*80)
    print(f"Volume:     {args.volume_path}")
    print(f"Azimuth:    {args.azimuth}°")
    print(f"Elevation:  {args.elevation}°")
    print(f"Distance:   {args.distance}x")
    if args.roll != 0:
        print(f"Roll:       {args.roll}°")
    print(f"Output:     {output_path}")
    print("="*80)
    print()

    try:
        # Render volume
        render_volume(
            volume_path=str(volume_path),
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            output_path=output_path,
            metadata=metadata,
            window_size=window_size,
            background_color=args.background,
            show_axes=args.axes,
            verbose=args.verbose,
            roll=args.roll
        )

        print()
        print("✅ Rendering complete!")
        print(f"📁 Saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
