#!/usr/bin/env python3
"""
Find the optimal camera angle by matching against a reference image.

This tool automatically tests multiple camera angles and uses vision AI to
find the angle that best matches a reference image from the knowledge graph.

Usage:
    python tools/find_camera_angle.py vis_male.raw skull
    python tools/find_camera_angle.py vis_male.raw skull --strategy grid
    python tools/find_camera_angle.py vis_male.raw skull --output results/
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.angle_matcher import AngleMatcher
from src.ontovis.multimodal_kg import MultimodalKnowledgeGraph


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal camera angle by matching against KG reference image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/volumes/vis_male.raw skull
    Find angle matching skull reference

  %(prog)s data/volumes/vis_male.raw skull --strategy grid
    Use exhaustive grid search instead of coarse-to-fine

  %(prog)s data/volumes/vis_male.raw bone --output my_results/
    Save results to custom directory

Search Strategies:
  coarse_to_fine (default): Fast two-phase search
    - Phase 1: Test at 45° increments
    - Phase 2: Refine around best match at 15° increments

  grid: Exhaustive search at 30° increments
    - Tests more angles but slower
        """
    )

    parser.add_argument(
        'volume_path',
        help='Path to volume file (.raw, .npy, .dat)'
    )

    parser.add_argument(
        'reference_category',
        help='Category of reference image in KG (e.g., "skull", "bone")'
    )

    parser.add_argument(
        '--strategy',
        '-s',
        choices=['coarse_to_fine', 'grid'],
        default='coarse_to_fine',
        help='Search strategy (default: coarse_to_fine)'
    )

    parser.add_argument(
        '--output',
        '-o',
        help='Output directory for test renders and results'
    )

    parser.add_argument(
        '--kg-path',
        default='.kg',
        help='Path to knowledge graph directory (default: .kg)'
    )

    parser.add_argument(
        '--dimensions',
        help='Volume dimensions for .raw files (e.g., "256,256,128")'
    )

    parser.add_argument(
        '--dtype',
        default='uint8',
        help='Data type for .raw files (default: uint8)'
    )

    args = parser.parse_args()

    # Check volume file exists
    volume_path = Path(args.volume_path)
    if not volume_path.exists():
        print(f"❌ Volume file not found: {args.volume_path}")
        sys.exit(1)

    # Check KG exists
    kg_path = Path(args.kg_path)
    if not kg_path.exists():
        print(f"❌ Knowledge graph not found: {args.kg_path}")
        print("\nCreate a knowledge graph first:")
        print("  python tools/import_kg_markdown.py data/KG/medical_kg.md")
        sys.exit(1)

    # Load KG and find reference image
    kg = MultimodalKnowledgeGraph(kg_path=str(kg_path))
    refs = kg.get_reference_renders(category=args.reference_category, quality="good")

    if not refs:
        print(f"❌ No reference images found for category: {args.reference_category}")
        print("\nAvailable categories:")
        all_refs = kg.get_reference_renders()
        categories = set(ref.get('category') for ref in all_refs)
        for cat in sorted(categories):
            print(f"  - {cat}")
        sys.exit(1)

    ref = refs[0]
    ref_image_path = kg.kg_path / ref['image_path']

    if not ref_image_path.exists():
        print(f"❌ Reference image not found: {ref_image_path}")
        print("\nThe KG has a reference but the image file is missing.")
        print("Reimport your KG to fix:")
        print("  python tools/import_kg_markdown.py data/KG/medical_kg.md")
        sys.exit(1)

    print("="*80)
    print("CAMERA ANGLE FINDER")
    print("="*80)
    print(f"Volume: {args.volume_path}")
    print(f"Reference: {args.reference_category} ({ref_image_path.name})")
    print(f"Strategy: {args.strategy}")
    print("="*80)
    print()

    # Prepare metadata
    metadata = None
    if volume_path.suffix == '.raw':
        if args.dimensions:
            dims = tuple(map(int, args.dimensions.split(',')))
            metadata = {'dimensions': dims, 'dtype': args.dtype}
        else:
            # Try to infer from filename
            stem = volume_path.stem
            parts = stem.split('_')
            if len(parts) >= 2:
                try:
                    dims_str = parts[1]
                    dims = tuple(map(int, dims_str.split('x')))
                    dtype = parts[2] if len(parts) > 2 else 'uint8'
                    metadata = {'dimensions': dims, 'dtype': dtype}
                    print(f"ℹ️  Inferred dimensions from filename: {dims}")
                except:
                    print("❌ Could not infer dimensions from filename.")
                    print("Please provide --dimensions for .raw files")
                    sys.exit(1)

    # Run angle matcher
    matcher = AngleMatcher()

    try:
        result = matcher.find_best_angle(
            volume_path=str(volume_path),
            reference_image_path=str(ref_image_path),
            metadata=metadata,
            output_dir=args.output,
            search_strategy=args.strategy
        )

        print("\n" + "="*80)
        print("RESULT")
        print("="*80)

        best = result['best_match']
        print(f"Best Camera Angle:")
        print(f"  Elevation: {best['elevation']}°")
        print(f"  Azimuth: {best['azimuth']}°")
        print(f"  Match Score: {best['match_score']}/10")
        print()
        print(f"Best Render: {best['render_path']}")
        print(f"Summary: {Path(args.output or 'angle_search_results') / 'angle_search_summary.json'}")
        print()
        print("Top 3 matches:")
        for i, res in enumerate(result['all_results'][:3], 1):
            print(f"  {i}. elev={res['elevation']}°, azim={res['azimuth']}° → score={res['match_score']}/10")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
