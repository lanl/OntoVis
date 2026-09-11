#!/usr/bin/env python3
"""
Match camera angle to a reference image using vision-guided iterative search.

Usage:
    python tools/match_angle.py <volume_path> <reference_id> [options]

Examples:
    python tools/match_angle.py data/vis_male.raw a_skull_front_view --dimensions 256,256,128
    python tools/match_angle.py data/foot.raw a_skull_side_view --distance 3.0
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pyvista as pv
from PIL import Image
import io

from src.ontovis.angle_matcher import AngleMatcher
from src.ontovis.multimodal_kg import MultimodalKnowledgeGraph


def create_render_function(volume_path: str, dimensions: tuple, dtype: str):
    """Create a rendering function for the angle matcher.

    Args:
        volume_path: Path to volume file
        dimensions: (width, height, depth) tuple
        dtype: Data type (e.g., 'uint8')

    Returns:
        Function that renders volume at given angles
    """
    # Load volume once
    volume_data = np.fromfile(volume_path, dtype=dtype)
    volume_data = volume_data.reshape(dimensions)

    def render_at_angle(azimuth: float, elevation: float, roll: float, distance: float) -> Image.Image:
        """Render volume at specified camera angles.

        Args:
            azimuth: Horizontal rotation (0-360°)
            elevation: Vertical angle (-90 to 90°)
            roll: Camera roll (-180 to 180°)
            distance: Camera distance multiplier

        Returns:
            PIL Image of the render
        """
        # Create plotter (offscreen)
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))

        # Create volume
        grid = pv.ImageData()
        grid.dimensions = np.array(dimensions) + 1
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid.point_data["values"] = volume_data.flatten(order="F")

        # Add volume with linear opacity
        plotter.add_volume(
            grid,
            cmap="gray_r",
            opacity="linear",
            shade=True
        )

        # Set background to black
        plotter.set_background("black")

        # Calculate camera position using spherical coordinates
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        # Center of volume
        center = np.array([d / 2.0 for d in dimensions])

        # Calculate radius based on volume size
        max_dim = max(dimensions)
        radius = max_dim * distance

        # Convert spherical to Cartesian
        x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = radius * np.sin(elevation_rad)

        camera_position = center + np.array([x, y, z])

        # Calculate up vector with roll
        view_dir = center - camera_position
        view_dir = view_dir / np.linalg.norm(view_dir)

        base_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, base_up)) > 0.99:
            base_up = np.array([0.0, 1.0, 0.0])

        right = np.cross(base_up, view_dir)
        right = right / np.linalg.norm(right)

        up = np.cross(view_dir, right)
        up = up / np.linalg.norm(up)

        # Apply roll using Rodrigues rotation
        if roll != 0:
            roll_rad = np.radians(roll)
            cos_roll = np.cos(roll_rad)
            sin_roll = np.sin(roll_rad)
            k_dot_v = np.dot(view_dir, up)
            k_cross_v = np.cross(view_dir, up)
            up = up * cos_roll + k_cross_v * sin_roll + view_dir * k_dot_v * (1 - cos_roll)
            up = up / np.linalg.norm(up)

        # Set camera
        plotter.camera_position = [
            tuple(camera_position),
            tuple(center),
            tuple(up)
        ]

        # Render to image
        img_array = plotter.screenshot(return_img=True)
        plotter.close()

        # Convert to PIL Image
        return Image.fromarray(img_array)

    return render_at_angle


def main():
    parser = argparse.ArgumentParser(
        description="Match camera angle to reference image using vision-guided search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/vis_male.raw a_skull_front_view --dimensions 256,256,128
  %(prog)s data/foot.raw a_skull_side_view --distance 3.0 --kg-path .kg

Reference IDs in KG:
  - a_skull_front_view
  - a_skull_side_view
        """
    )

    parser.add_argument('volume_path', help='Path to raw volume file')
    parser.add_argument('reference_id', help='Reference ID from KG (e.g., a_skull_front_view)')

    parser.add_argument('--dimensions', type=str, help='Volume dimensions as "width,height,depth" (e.g., "256,256,128")')
    parser.add_argument('--dtype', default='uint8', help='Data type (default: uint8)')
    parser.add_argument('--distance', type=float, default=2.0, help='Camera distance multiplier (default: 2.0)')
    parser.add_argument('--kg-path', default='.kg', help='Path to knowledge graph (default: .kg)')
    parser.add_argument('--output', '-o', help='Save final render to file')

    args = parser.parse_args()

    # Parse dimensions
    if args.dimensions:
        try:
            dims = tuple(map(int, args.dimensions.split(',')))
            if len(dims) != 3:
                raise ValueError()
        except:
            print(f"Error: Invalid dimensions format. Use 'width,height,depth' (e.g., '256,256,128')")
            sys.exit(1)
    else:
        # Try to infer from filename
        import re
        match = re.search(r'(\d+)x(\d+)x(\d+)', args.volume_path)
        if match:
            dims = tuple(map(int, match.groups()))
            print(f"Inferred dimensions from filename: {dims}")
        else:
            print("Error: Could not infer dimensions. Please specify --dimensions")
            sys.exit(1)

    print("="*70)
    print("VISION-GUIDED ANGLE MATCHING")
    print("="*70)
    print(f"Volume:    {args.volume_path}")
    print(f"Dimensions: {dims}")
    print(f"Reference: {args.reference_id}")
    print(f"Distance:  {args.distance}x")
    print(f"KG Path:   {args.kg_path}")
    print("="*70)

    # Check volume exists
    if not Path(args.volume_path).exists():
        print(f"\nError: Volume file not found: {args.volume_path}")
        sys.exit(1)

    # Initialize matcher
    print("\nInitializing AngleMatcher...")
    matcher = AngleMatcher(kg_path=args.kg_path)

    # Create render function
    print("Creating render function...")
    render_fn = create_render_function(args.volume_path, dims, args.dtype)

    # Find best angle
    print("\nStarting angle search...")
    result = matcher.find_best_angle(
        render_fn=render_fn,
        reference_id=args.reference_id,
        volume_path=args.volume_path,
        distance=args.distance
    )

    # Print results
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Match Score:  {result['match_score']:.1f}/10")
    print(f"Strategy:     {result['search_strategy']}")
    print(f"Iterations:   {result['iterations']}")
    print(f"From Cache:   {result.get('from_cache', False)}")
    print(f"\nBest Angles:")
    print(f"  Azimuth:   {result['final_angles']['azimuth']}°")
    print(f"  Elevation: {result['final_angles']['elevation']}°")
    print(f"  Roll:      {result['final_angles']['roll']}°")
    print(f"  Distance:  {result['final_angles']['distance']}")
    print("="*70)

    # Save final render if requested
    if args.output:
        print(f"\nRendering final image...")
        final_image = render_fn(
            result['final_angles']['azimuth'],
            result['final_angles']['elevation'],
            result['final_angles']['roll'],
            result['final_angles']['distance']
        )
        final_image.save(args.output)
        print(f"✓ Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
