"""Test script for VolumeAnalysisAgent - creates synthetic data and runs analysis."""

import numpy as np
from pathlib import Path


def test_volume_agent():
    """Test the volume analysis agent with synthetic data."""
    print("Testing VolumeAnalysisAgent...")
    print("="*70)

    # Create synthetic test data
    print("\n1. Creating synthetic volume data...")
    volume = np.zeros((64, 64, 64), dtype=np.uint8)

    # Background: 5-15
    volume[:] = np.random.randint(5, 15, size=volume.shape)

    # Central sphere: 80-120
    center = np.array([32, 32, 32])
    x, y, z = np.ogrid[:64, :64, :64]
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    mask = distance <= 20
    volume[mask] = np.random.randint(80, 120, size=np.sum(mask))

    # Inner core: 180-220
    mask_core = distance <= 10
    volume[mask_core] = np.random.randint(180, 220, size=np.sum(mask_core))

    # Save test volume as RAW binary file
    test_path = Path('test_volume_data.raw')
    volume.tofile(test_path)
    print(f"   ✓ Created test volume: {volume.shape}, {volume.dtype}")
    print(f"   ✓ Saved to: {test_path} (raw binary format)")

    # Test the agent
    print("\n2. Initializing VolumeAnalysisAgent...")
    try:
        from ontovis import VolumeAnalysisAgent
        agent = VolumeAnalysisAgent()
        print("   ✓ Agent initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize agent: {e}")
        return False

    # Run analysis
    print("\n3. Running volume analysis...")
    try:
        results = agent.analyze_volume(
            volume_path=str(test_path),
            user_description="""
            Synthetic test dataset with three distinct regions:
            - Background/air (very low intensity)
            - Outer sphere material (medium intensity)
            - Inner core material (high intensity)
            """,
            metadata={
                'name': 'Test Volume',
                'dimensions': [64, 64, 64],  # Required for .raw files
                'dtype': 'uint8',             # Required for .raw files
                'modality': 'Synthetic',
                'purpose': 'Testing multi-material segmentation'
            },
            save_histogram='test_histogram.png'
        )
        print("   ✓ Analysis complete")
    except Exception as e:
        print(f"   ✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display results
    print("\n4. Results:")
    print("-"*70)

    stats = results.get('volume_stats', {})
    if stats:
        print("\nVolume Statistics:")
        print(f"   Shape:        {stats.get('shape')}")
        print(f"   Data type:    {stats.get('dtype')}")
        print(f"   Value range:  {stats.get('min')} - {stats.get('max')}")
        print(f"   Mean:         {stats.get('mean'):.2f}")
        print(f"   Median:       {stats.get('median'):.2f}")
        print(f"   Std dev:      {stats.get('std'):.2f}")

    hist_data = results.get('histogram_data', {})
    if 'peaks' in hist_data:
        print(f"\nDetected peaks: {hist_data['peaks']}")

    print("\nFeature Analysis:")
    print("-"*70)
    print(results.get('feature_analysis', 'No analysis available'))
    print("-"*70)

    print("\n✓ Test completed successfully!")
    print(f"✓ Histogram saved to: test_histogram.png")

    # Cleanup
    print("\nCleaning up test files...")
    test_path.unlink()
    print("   ✓ Cleanup complete")

    return True


if __name__ == "__main__":
    try:
        success = test_volume_agent()
        if success:
            print("\n" + "="*70)
            print("ALL TESTS PASSED")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("TESTS FAILED")
            print("="*70)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
