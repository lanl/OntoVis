"""Helper script to download and analyze volume datasets from open-scivis-datasets."""

import urllib.request
import gzip
import shutil
import ssl
import time
import subprocess
from pathlib import Path
from ontovis import VolumeAnalysisAgent


# Dataset catalog from klacansky.com/open-scivis-datasets
DATASETS = {
    'bonsai': {
        'url': 'https://klacansky.com/open-scivis-datasets/bonsai/bonsai_256x256x256_uint8.raw.gz',
        'dimensions': [256, 256, 256],
        'dtype': 'uint8',
        'description': '''
        Micro-CT scan of a bonsai tree showing internal wood structure.
        Contains wood density variations, bark, air spaces, and growth rings.
        Useful for studying botanical structures and wood anatomy.
        ''',
        'name': 'Bonsai Tree',
        'modality': 'micro-CT'
    },
    'engine': {
        'url': 'https://klacansky.com/open-scivis-datasets/engine/engine_256x256x128_uint8.raw.gz',
        'dimensions': [256, 256, 128],
        'dtype': 'uint8',
        'description': '''
        CT scan of an internal combustion engine.
        Shows metal engine block, cylinders, cooling channels, and air spaces.
        Different metal densities and mechanical structures are visible.
        ''',
        'name': 'Engine Block',
        'modality': 'CT'
    },
    'foot': {
        'url': 'https://klacansky.com/open-scivis-datasets/foot/foot_256x256x256_uint8.raw.gz',
        'dimensions': [256, 256, 256],
        'dtype': 'uint8',
        'description': '''
        Medical CT scan of a human foot.
        Contains bone structures, soft tissue, muscle, and air.
        Shows anatomical features including bones, joints, and tissue layers.
        ''',
        'name': 'Human Foot',
        'modality': 'Medical CT'
    },
    'skull': {
        'url': 'https://klacansky.com/open-scivis-datasets/skull/skull_256x256x256_uint8.raw.gz',
        'dimensions': [256, 256, 256],
        'dtype': 'uint8',
        'description': '''
        Medical CT scan of a human skull.
        Shows bone density variations, cranial structures, teeth, and sinuses.
        Demonstrates multiple tissue types including dense bone, spongy bone, and air cavities.
        ''',
        'name': 'Human Skull',
        'modality': 'Medical CT'
    },
    'backpack': {
        'url': 'https://klacansky.com/open-scivis-datasets/backpack/backpack_512x512x373_uint16.raw.gz',
        'dimensions': [512, 512, 373],
        'dtype': 'uint16',
        'description': '''
        Security CT scan of a backpack with various items.
        Contains different materials: fabric, metal, plastic, liquids, and electronics.
        Useful for studying material discrimination in security screening.
        ''',
        'name': 'Backpack Scan',
        'modality': 'Security CT'
    }
}


def download_dataset(dataset_key: str, output_dir: str = 'datasets') -> str:
    """Download and extract a dataset.

    Args:
        dataset_key: Key from DATASETS dictionary
        output_dir: Directory to save the dataset

    Returns:
        str: Path to the extracted .raw file
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")

    dataset = DATASETS[dataset_key]
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Determine filenames
    gz_filename = output_path / f"{dataset_key}.raw.gz"
    raw_filename = output_path / f"{dataset_key}.raw"

    # Download if not already present
    if not raw_filename.exists():
        print(f"Downloading {dataset_key} dataset...")
        print(f"URL: {dataset['url']}")

        # Create SSL context that's more permissive for older servers
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Retry logic for flaky connections
        max_retries = 3
        download_success = False

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}...")
                # Use custom SSL context
                with urllib.request.urlopen(dataset['url'], context=ssl_context, timeout=30) as response:
                    with open(gz_filename, 'wb') as out_file:
                        # Download in chunks with progress
                        chunk_size = 8192
                        downloaded = 0
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (1024 * 1024) == 0:  # Every MB
                                print(f"  Downloaded {downloaded // (1024 * 1024)} MB...", end='\r')

                print(f"\nDownloaded to {gz_filename}")
                download_success = True
                break

            except (urllib.error.URLError, ssl.SSLError, TimeoutError) as e:
                print(f"\nAttempt {attempt + 1} failed: {e}")
                if gz_filename.exists():
                    gz_filename.unlink()

                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)

        if not download_success:
            # Try with curl or wget as fallback
            print("\nTrying alternative download methods...")

            # Try curl
            try:
                print("Attempting with curl...")
                result = subprocess.run(
                    ['curl', '-L', '-k', '-o', str(gz_filename), dataset['url']],
                    capture_output=True,
                    timeout=300
                )
                if result.returncode == 0 and gz_filename.exists():
                    print(f"Successfully downloaded with curl to {gz_filename}")
                    download_success = True
            except (FileNotFoundError, subprocess.SubprocessError) as e:
                print(f"curl failed: {e}")

            # Try wget if curl failed
            if not download_success:
                try:
                    print("Attempting with wget...")
                    result = subprocess.run(
                        ['wget', '--no-check-certificate', '-O', str(gz_filename), dataset['url']],
                        capture_output=True,
                        timeout=300
                    )
                    if result.returncode == 0 and gz_filename.exists():
                        print(f"Successfully downloaded with wget to {gz_filename}")
                        download_success = True
                except (FileNotFoundError, subprocess.SubprocessError) as e:
                    print(f"wget failed: {e}")

        if not download_success:
            print(f"\n{'='*70}")
            print("DOWNLOAD FAILED")
            print(f"{'='*70}")
            print("\nAll download methods failed. Manual download required:")
            print(f"1. Download manually from: {dataset['url']}")
            print(f"2. Extract the .gz file")
            print(f"3. Place the .raw file in: {output_path}/")
            print(f"4. Then run: python examples/analyze_foot.py {raw_filename}")
            print("\nOr use curl/wget directly:")
            print(f"  curl -L -k -o {gz_filename} {dataset['url']}")
            print(f"  gunzip {gz_filename}")
            raise RuntimeError(f"Failed to download after trying all methods")

        try:

            # Extract
            print(f"Extracting...")
            with gzip.open(gz_filename, 'rb') as f_in:
                with open(raw_filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Clean up compressed file
            gz_filename.unlink()
            print(f"Extracted to {raw_filename}")

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if gz_filename.exists():
                gz_filename.unlink()
            raise

    else:
        print(f"Dataset already exists at {raw_filename}")

    return str(raw_filename)


def analyze_dataset(dataset_key: str, download_dir: str = 'datasets'):
    """Download (if needed) and analyze a dataset.

    Args:
        dataset_key: Key from DATASETS dictionary
        download_dir: Directory for dataset storage
    """
    if dataset_key not in DATASETS:
        print(f"Unknown dataset: {dataset_key}")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return

    dataset_info = DATASETS[dataset_key]

    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_info['name']}")
    print(f"{'='*70}\n")

    # Download dataset
    volume_path = download_dataset(dataset_key, download_dir)

    # Initialize agent
    print("\nInitializing VolumeAnalysisAgent...")
    agent = VolumeAnalysisAgent()

    # Prepare metadata
    metadata = {
        'name': dataset_info['name'],
        'dimensions': dataset_info['dimensions'],
        'dtype': dataset_info['dtype'],
        'modality': dataset_info['modality']
    }

    # Analyze
    print(f"\nAnalyzing {dataset_info['name']}...")
    output_histogram = f"{dataset_key}_histogram.png"

    results = agent.analyze_volume(
        volume_path=volume_path,
        user_description=dataset_info['description'],
        metadata=metadata,
        save_histogram=output_histogram
    )

    # Display results
    print(f"\n{'='*70}")
    print("VOLUME STATISTICS")
    print(f"{'='*70}")
    stats = results['volume_stats']
    print(f"Shape:        {stats['shape']}")
    print(f"Data type:    {stats['dtype']}")
    print(f"Total voxels: {stats['total_voxels']:,}")
    print(f"Value range:  {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"Mean:         {stats['mean']:.2f}")
    print(f"Median:       {stats['median']:.2f}")
    print(f"Std dev:      {stats['std']:.2f}")

    print(f"\n{'='*70}")
    print("FEATURE ANALYSIS")
    print(f"{'='*70}")
    print(results['feature_analysis'])

    print(f"\n{'='*70}")
    print(f"✓ Histogram saved to: {output_histogram}")
    print(f"{'='*70}\n")


def list_datasets():
    """List all available datasets."""
    print("\nAvailable Datasets from open-scivis-datasets:")
    print("=" * 70)

    for key, info in DATASETS.items():
        dims = 'x'.join(map(str, info['dimensions']))
        print(f"\n{key:12} - {info['name']}")
        print(f"             {dims}, {info['dtype']}, {info['modality']}")
        print(f"             {info['description'].strip()[:100]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python download_and_analyze.py <dataset_key>")
        print("       python download_and_analyze.py list")
        print("\nExample: python download_and_analyze.py bonsai")
        list_datasets()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'list':
        list_datasets()
    elif command in DATASETS:
        analyze_dataset(command)
    else:
        print(f"Unknown dataset or command: {command}")
        list_datasets()
