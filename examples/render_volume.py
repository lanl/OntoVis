"""CLI tool for volume rendering with natural language prompts."""

import sys
import argparse
from pathlib import Path
from ontovis import VolumeRenderAgent


def main():
    parser = argparse.ArgumentParser(
        description='Render 3D volumes using natural language prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render bones in white
  python render_volume.py data/foot.raw -x 256 -y 256 -z 256 \\
      -p "Show the bones in white, make everything else transparent" \\
      -o foot_bones.png

  # Create a heatmap
  python render_volume.py data/volume.npy \\
      -p "Create a density heatmap using blue to red colors" \\
      -o heatmap.png

  # Show bones and tissue
  python render_volume.py data/scan.raw -x 512 -y 512 -z 300 --dtype uint16 \\
      -p "Show bones in white and soft tissue in semi-transparent pink" \\
      -o composite.png

Prompt Examples:
  - "Show the bones in white, make everything else transparent"
  - "Create a heat map visualization with blue-to-red color scale"
  - "Highlight high-density structures, view from above"
  - "Show bones opaque and soft tissue semi-transparent"
  - "Make a grayscale rendering showing all structures"
  - "Render only the dense materials in the scan"
        """
    )

    parser.add_argument('path', type=str, help='Path to volume data file')
    parser.add_argument('-p', '--prompt', type=str, required=True,
                       help='Natural language description of desired rendering')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output image path (PNG)')

    # .raw file parameters
    parser.add_argument('-x', '--dim-x', type=int, help='X dimension (for .raw files)')
    parser.add_argument('-y', '--dim-y', type=int, help='Y dimension (for .raw files)')
    parser.add_argument('-z', '--dim-z', type=int, help='Z dimension (for .raw files)')
    parser.add_argument('--dtype', type=str, default='uint8',
                       help='Data type (for .raw files): uint8, uint16, float32, etc.')

    # Optional metadata
    parser.add_argument('-n', '--name', type=str, help='Dataset name')
    parser.add_argument('-m', '--modality', type=str, help='Imaging modality')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        sys.exit(1)

    # Prepare metadata
    metadata = {}

    # Handle .raw files
    if args.path.endswith('.raw'):
        if not all([args.dim_x, args.dim_y, args.dim_z]):
            print("Error: .raw files require -x, -y, -z dimensions")
            print("Example: python render_volume.py data.raw -x 256 -y 256 -z 256 -p 'prompt' -o out.png")
            sys.exit(1)
        metadata['dimensions'] = [args.dim_x, args.dim_y, args.dim_z]
        metadata['dtype'] = args.dtype

    # Add optional metadata
    if args.name:
        metadata['name'] = args.name
    if args.modality:
        metadata['modality'] = args.modality

    # Initialize agent
    print("Initializing VolumeRenderAgent...")
    agent = VolumeRenderAgent()

    # Render
    print(f"Loading volume: {args.path}")
    print(f"Prompt: {args.prompt}")
    print("\nAnalyzing volume and determining rendering parameters...")

    try:
        results = agent.render(
            volume_path=args.path,
            prompt=args.prompt,
            metadata=metadata,
            save_image=args.output
        )

        print("\n" + "="*70)
        print("RENDERING COMPLETE")
        print("="*70)
        print(f"\nOutput saved to: {args.output}")
        print(f"\nExplanation: {results['explanation']}")

        print("\nRendering Parameters:")
        instructions = results.get('instructions', {})
        if instructions:
            print(f"  Method: {instructions.get('method', 'N/A')}")
            print(f"  Threshold: {instructions.get('threshold', 'N/A')}")
            print(f"  Color (RGB): {instructions.get('color', 'N/A')}")
            print(f"  Opacity: {instructions.get('opacity', 'N/A')}")
            print(f"  Camera distance: {instructions.get('camera_distance', 'N/A')}")
            print(f"  Background (RGB): {instructions.get('background', 'N/A')}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\nError during rendering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
