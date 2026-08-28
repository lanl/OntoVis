"""Analyze any volume dataset."""

import sys
import argparse
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from ontovis import VolumeAnalysisAgent


def display_histogram(histogram_base64):
    """Display histogram image from base64 data."""
    # Decode base64 to image
    image_data = base64.b64decode(histogram_base64)
    image = Image.open(BytesIO(image_data))

    # Display using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze volume datasets with AI-powered feature identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a .npy file (no metadata needed)
  python analyze_foot.py dataset.npy -d "CT scan of skull"

  # Analyze a .raw file (requires dimensions and dtype)
  python analyze_foot.py data.raw -x 256 -y 256 -z 256 --dtype uint8 -d "Foot CT scan"

  # Full example with all metadata
  python analyze_foot.py foot.raw -x 256 -y 256 -z 256 --dtype uint8 \\
      -d "Medical CT of human foot with bones and soft tissue" \\
      -n "Foot CT" -m "Medical CT" -o foot_histogram.png
        """
    )

    parser.add_argument('path', type=str, help='Path to volume data file (.raw, .npy, .dat)')
    parser.add_argument('-d', '--description', type=str, required=True,
                       help='Description of what the dataset contains')
    parser.add_argument('-x', '--dim-x', type=int, help='X dimension (required for .raw files)')
    parser.add_argument('-y', '--dim-y', type=int, help='Y dimension (required for .raw files)')
    parser.add_argument('-z', '--dim-z', type=int, help='Z dimension (required for .raw files)')
    parser.add_argument('--dtype', type=str, default='uint8',
                       help='Data type: uint8, uint16, float32, etc. (default: uint8)')
    parser.add_argument('-n', '--name', type=str, help='Dataset name for plot title')
    parser.add_argument('-m', '--modality', type=str, help='Imaging modality (e.g., CT, MRI)')
    parser.add_argument('-o', '--output', type=str, help='Output path for histogram PNG')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display histogram window')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        sys.exit(1)

    # Prepare metadata
    metadata = {}

    # Handle .raw files - require dimensions
    if args.path.endswith('.raw'):
        if not all([args.dim_x, args.dim_y, args.dim_z]):
            print("Error: .raw files require -x, -y, -z dimensions")
            print("Example: python analyze_foot.py data.raw -x 256 -y 256 -z 256 -d 'description'")
            sys.exit(1)
        metadata['dimensions'] = [args.dim_x, args.dim_y, args.dim_z]
        metadata['dtype'] = args.dtype

    # Add optional metadata
    if args.name:
        metadata['name'] = args.name
    if args.modality:
        metadata['modality'] = args.modality

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate from input filename
        input_name = Path(args.path).stem
        output_path = f"{input_name}_histogram.png"

    # Initialize the agent
    print("Initializing VolumeAnalysisAgent...")
    agent = VolumeAnalysisAgent()

    # Analyze the dataset
    print(f"\nAnalyzing {args.path}...")
    results = agent.analyze_volume(
        volume_path=args.path,
        user_description=args.description,
        metadata=metadata,
        save_histogram=output_path
    )

    # Display results
    print("\n" + "="*70)
    print("VOLUME ANALYSIS RESULTS")
    print("="*70)

    print("\nVolume Statistics:")
    stats = results['volume_stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if 'peaks' in results.get('histogram_data', {}):
        peaks = results['histogram_data']['peaks']
        if peaks:
            print(f"\nDetected Intensity Peaks: {[f'{p:.1f}' for p in peaks]}")

    print("\n" + "="*70)
    print("AI FEATURE ANALYSIS")
    print("="*70)
    print(results['feature_analysis'])

    print("\n" + "="*70)
    print(f"✓ Analysis complete!")
    print(f"✓ Histogram saved to: {output_path}")
    print("="*70)

    # Display histogram
    if not args.no_display and results.get('histogram_image_base64'):
        print("\nDisplaying histogram...")
        display_histogram(results['histogram_image_base64'])


if __name__ == "__main__":
    main()
