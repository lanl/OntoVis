"""Example usage of the VolumeAnalysisAgent for analyzing 3D volume datasets."""

from ontovis import VolumeAnalysisAgent
import numpy as np


def create_synthetic_volume(output_path='test_volume.npy', file_format='npy'):
    """Create a synthetic volume dataset for testing.

    Simulates a multi-material volume with:
    - Background (low intensity)
    - Material 1 (medium intensity)
    - Material 2 (high intensity)
    - Some noise

    Args:
        output_path: Path to save the volume
        file_format: 'npy' or 'raw'
    """
    print("Creating synthetic volume dataset...")

    # Create a 128x128x128 volume
    volume = np.zeros((128, 128, 128), dtype=np.uint8)

    # Background (air/void): intensity 10-20
    volume[:] = np.random.randint(10, 20, size=volume.shape)

    # Sphere of Material 1 (tissue/material): intensity 80-100
    center = np.array([64, 64, 64])
    radius = 30
    x, y, z = np.ogrid[:128, :128, :128]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask1 = distance <= radius
    volume[mask1] = np.random.randint(80, 100, size=np.sum(mask1))

    # Smaller sphere of Material 2 (dense material/bone): intensity 180-220
    radius2 = 15
    mask2 = distance <= radius2
    volume[mask2] = np.random.randint(180, 220, size=np.sum(mask2))

    # Add some noise
    noise = np.random.normal(0, 3, size=volume.shape)
    volume = np.clip(volume.astype(float) + noise, 0, 255).astype(np.uint8)

    # Save in requested format
    if file_format == 'raw':
        # Save as raw binary
        if not output_path.endswith('.raw'):
            output_path = output_path.replace('.npy', '.raw')
        volume.tofile(output_path)
        print(f"Synthetic volume saved to {output_path} (raw binary format)")
    else:
        # Save as .npy file
        if not output_path.endswith('.npy'):
            output_path = output_path.replace('.raw', '.npy')
        np.save(output_path, volume)
        print(f"Synthetic volume saved to {output_path} (NumPy format)")

    print(f"Shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"Value range: {volume.min()} to {volume.max()}")

    return output_path, volume.shape, volume.dtype


def analyze_synthetic_volume():
    """Example: Analyze a synthetic volume dataset (NumPy format)."""
    print("\n=== Synthetic Volume Analysis Example (NPY format) ===\n")

    # Create synthetic data in NumPy format
    volume_path, shape, dtype = create_synthetic_volume(file_format='npy')

    # Initialize the agent
    print("\nInitializing VolumeAnalysisAgent...")
    agent = VolumeAnalysisAgent()

    # Analyze the volume
    print("\nAnalyzing volume dataset...")
    results = agent.analyze_volume(
        volume_path=volume_path,
        user_description="""
        This is a synthetic medical imaging dataset simulating a CT scan.
        It contains three main components:
        - Background/air (very low intensity)
        - Soft tissue (medium intensity)
        - Dense bone material (high intensity)
        """,
        metadata={
            'name': 'Synthetic CT Scan',
            'modality': 'CT',
            'units': 'Hounsfield Units (normalized to 0-255)'
        },
        save_histogram='synthetic_volume_histogram.png'
    )

    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print("\nVolume Statistics:")
    stats = results['volume_stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)
    print("Feature Analysis:")
    print("-"*60)
    print(results['feature_analysis'])

    print(f"\nHistogram saved to: synthetic_volume_histogram.png")


def analyze_raw_volume_example():
    """Example: Analyze a .raw volume file (requires dimensions in metadata)."""
    print("\n=== Raw Volume Analysis Example ===\n")

    # Create synthetic data in RAW format
    print("Creating synthetic RAW volume...")
    volume_path, shape, dtype = create_synthetic_volume(
        output_path='test_volume.raw',
        file_format='raw'
    )

    # Initialize the agent
    print("\nInitializing VolumeAnalysisAgent...")
    agent = VolumeAnalysisAgent()

    # Analyze the RAW volume
    # IMPORTANT: .raw files REQUIRE dimensions and dtype in metadata
    print("\nAnalyzing RAW volume dataset...")
    results = agent.analyze_volume(
        volume_path=volume_path,
        user_description="""
        This is a synthetic medical imaging dataset simulating a CT scan.
        It contains three main components:
        - Background/air (very low intensity)
        - Soft tissue (medium intensity)
        - Dense bone material (high intensity)
        """,
        metadata={
            'name': 'Synthetic CT Scan (RAW)',
            'dimensions': list(shape),  # REQUIRED for .raw files
            'dtype': str(dtype),        # REQUIRED for .raw files
            'modality': 'CT',
            'units': 'Hounsfield Units (normalized to 0-255)'
        },
        save_histogram='synthetic_raw_histogram.png'
    )

    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print("\nVolume Statistics:")
    stats = results['volume_stats']
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)
    print("Feature Analysis:")
    print("-"*60)
    print(results['feature_analysis'])

    print(f"\nHistogram saved to: synthetic_raw_histogram.png")

    # Info about real datasets
    print("\n" + "="*60)
    print("For real .raw datasets:")
    print("="*60)
    print("Download from: http://klacansky.com/open-scivis-datasets/")
    print("\nExample datasets:")
    print("  - Bonsai (256^3, uint8)")
    print("  - Engine (256^3, uint8)")
    print("  - Foot (256^3, uint8)")
    print("  - Skull (256^3, uint8)")
    print("\nUse: python download_and_analyze.py <dataset_name>")


def analyze_with_custom_annotations():
    """Example: Create annotated histogram with feature labels."""
    print("\n=== Custom Annotation Example ===\n")

    # Create and analyze synthetic volume
    volume_path = create_synthetic_volume('annotated_test.npy')
    agent = VolumeAnalysisAgent()

    results = agent.analyze_volume(
        volume_path=volume_path,
        user_description="Synthetic multi-material volume",
        metadata={'name': 'Multi-Material Sample'}
    )

    # Create custom annotations based on known features
    feature_annotations = {
        15.0: 'Background/Air',
        90.0: 'Soft Tissue',
        200.0: 'Bone'
    }

    print("\nCreating annotated histogram...")
    agent.save_annotated_histogram(
        histogram_data=results['histogram_data'],
        feature_annotations=feature_annotations,
        output_path='annotated_histogram.png',
        metadata={'name': 'Multi-Material Sample'}
    )

    print("Annotated histogram saved to: annotated_histogram.png")


if __name__ == "__main__":
    # Run examples
    #analyze_synthetic_volume()

    print("\n" + "="*60 + "\n")
    analyze_raw_volume_example()

    #print("\n" + "="*60 + "\n")
    #analyze_with_custom_annotations()

    print("\n✓ Examples complete!")
