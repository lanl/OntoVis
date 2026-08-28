# Volume Analysis Agent

The VolumeAnalysisAgent uses AI to analyze 3D volume datasets, automatically identifying features and materials based on their intensity distributions.

## Overview

The agent performs the following workflow:
1. **Load Volume Data** - Reads volume data from various formats (.raw, .npy, .dat)
2. **Compute Histogram** - Calculates intensity distribution with peak detection
3. **Plot Histogram** - Creates publication-quality visualizations following data visualization best practices
4. **Analyze Features** - Uses Claude's vision capabilities to identify what each intensity range represents

## Quick Start

```python
from ontovis import VolumeAnalysisAgent

# Initialize agent
agent = VolumeAnalysisAgent()

# Analyze a volume dataset
results = agent.analyze_volume(
    volume_path='skull.raw',
    user_description='CT scan of human skull with bone and air spaces',
    metadata={
        'name': 'Skull CT',
        'dimensions': [256, 256, 256],  # Required for .raw files
        'dtype': 'uint8',               # Required for .raw files
        'modality': 'Medical CT'
    },
    save_histogram='skull_analysis.png'
)

# View AI analysis
print(results['feature_analysis'])
```

## Supported Formats

### .npy (NumPy Binary)
Easiest format - no metadata required:
```python
results = agent.analyze_volume(
    volume_path='volume.npy',
    user_description='Description of dataset'
)
```

### .raw (Raw Binary)
Requires dimensions and dtype in metadata:
```python
results = agent.analyze_volume(
    volume_path='volume.raw',
    user_description='Description',
    metadata={
        'dimensions': [256, 256, 256],  # [x, y, z]
        'dtype': 'uint8'                # numpy dtype string
    }
)
```

Supported dtypes: `'uint8'`, `'uint16'`, `'float32'`, `'float64'`

### .dat (Structured DAT)
Custom format with embedded dimensions:
```python
results = agent.analyze_volume(
    volume_path='volume.dat',
    user_description='Description'
)
```

## Working with Real Datasets

The easiest way to get started is with datasets from [open-scivis-datasets](http://klacansky.com/open-scivis-datasets/):

```bash
# Download and analyze in one command
uv run python examples/download_and_analyze.py bonsai
uv run python examples/download_and_analyze.py skull
uv run python examples/download_and_analyze.py engine
uv run python examples/download_and_analyze.py foot

# List all available datasets
uv run python examples/download_and_analyze.py list
```

Available datasets:
- **bonsai** - Micro-CT of bonsai tree (wood structure)
- **engine** - CT of engine block (metal, air)
- **foot** - Medical CT of human foot (bone, tissue)
- **skull** - Medical CT of human skull (bone, teeth, sinuses)
- **backpack** - Security scan (multiple materials)

## Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `dimensions` | Yes (for .raw) | `[x, y, z]` dimensions |
| `dtype` | Yes (for .raw) | NumPy dtype: `'uint8'`, `'uint16'`, `'float32'` |
| `name` | No | Dataset name for plot title |
| `modality` | No | Imaging type: 'CT', 'MRI', 'micro-CT', etc. |
| `resolution` | No | Spatial resolution (e.g., '0.5mm per voxel') |
| `units` | No | Intensity units (e.g., 'Hounsfield Units') |

## Return Values

The `analyze_volume()` method returns a dictionary:

```python
{
    'feature_analysis': str,          # AI-generated analysis
    'volume_stats': {
        'min': float,                 # Minimum intensity
        'max': float,                 # Maximum intensity
        'mean': float,                # Mean intensity
        'median': float,              # Median intensity
        'std': float,                 # Standard deviation
        'shape': tuple,               # Volume dimensions
        'dtype': str,                 # Data type
        'total_voxels': int          # Total number of voxels
    },
    'histogram_data': {
        'hist': ndarray,              # Histogram counts
        'bin_edges': ndarray,         # Bin edges
        'bin_centers': ndarray,       # Bin centers
        'peaks': list,                # Detected peak intensities
        'stats': dict                 # Statistics dict
    },
    'histogram_image_base64': str    # Base64-encoded PNG
}
```

## Custom Annotations

Create annotated histograms with your own feature labels:

```python
# First, analyze the volume
results = agent.analyze_volume(
    volume_path='data.npy',
    user_description='Multi-phase material'
)

# Define feature boundaries based on AI analysis
feature_annotations = {
    15.0: 'Background',
    80.0: 'Phase 1',
    150.0: 'Phase 2',
    220.0: 'Dense Core'
}

# Create annotated plot
agent.save_annotated_histogram(
    histogram_data=results['histogram_data'],
    feature_annotations=feature_annotations,
    output_path='annotated.png',
    metadata={'name': 'My Dataset'}
)
```

## Examples

### Example 1: Synthetic Data
```python
import numpy as np
from ontovis import VolumeAnalysisAgent

# Create synthetic volume
volume = np.zeros((128, 128, 128), dtype=np.uint8)
volume[:] = 10  # Background
# Add features...
np.save('synthetic.npy', volume)

# Analyze
agent = VolumeAnalysisAgent()
results = agent.analyze_volume(
    volume_path='synthetic.npy',
    user_description='Synthetic multi-material volume',
    save_histogram='synthetic_hist.png'
)
```

### Example 2: Medical CT
```python
results = agent.analyze_volume(
    volume_path='patient_scan.raw',
    user_description='''
    Chest CT scan showing lungs, heart, ribs, and soft tissue.
    Looking to identify bone, soft tissue, and air regions.
    ''',
    metadata={
        'name': 'Chest CT',
        'dimensions': [512, 512, 400],
        'dtype': 'uint16',
        'modality': 'Medical CT',
        'resolution': '0.5mm per voxel'
    },
    save_histogram='chest_analysis.png'
)

print(results['feature_analysis'])
```

### Example 3: Materials Science
```python
results = agent.analyze_volume(
    volume_path='composite_material.npy',
    user_description='''
    X-ray CT of composite material with polymer matrix and ceramic fibers.
    Three phases expected: matrix, fibers, and voids/defects.
    ''',
    metadata={
        'name': 'Composite Material',
        'modality': 'X-ray CT',
        'resolution': '5 micron per voxel'
    }
)

# AI will identify intensity ranges for:
# - Voids (low intensity)
# - Polymer matrix (medium intensity)  
# - Ceramic fibers (high intensity)
```

## Tips for Best Results

### 1. Provide Detailed Descriptions
The AI analysis is powered by your description. Be specific:

✗ Bad: "A CT scan"
✓ Good: "Medical CT scan of femur bone showing cortical bone, trabecular bone, and bone marrow"

### 2. Include Context
Help the AI understand what to look for:

```python
user_description='''
This is a micro-CT scan of a rock sample containing three minerals:
- Quartz (high density, bright)
- Feldspar (medium density)
- Mica (low density, dark)
We expect to see three distinct peaks in the histogram.
'''
```

### 3. Use Relevant Metadata
Include information that helps interpret intensities:

```python
metadata={
    'name': 'Rock Sample A',
    'modality': 'micro-CT',
    'resolution': '10 micron/voxel',
    'voltage': '80 kV',
    'expected_phases': 'quartz, feldspar, mica'
}
```

### 4. Iterate Based on Results
The AI might suggest thresholds - use them to create annotated versions:

```python
# Step 1: Get initial analysis
results = agent.analyze_volume(...)
print(results['feature_analysis'])

# Step 2: Extract suggested thresholds from analysis
# (AI might say "threshold at intensity 85 separates tissue from bone")

# Step 3: Create annotated histogram
annotations = {85.0: 'Tissue/Bone Boundary', 150.0: 'Dense Bone'}
agent.save_annotated_histogram(...)
```

## Visualization Style

Histograms follow professional data visualization guidelines:
- Sequential blue color for data bars
- Red dashed lines for detected peaks
- Recessive grid lines
- Clean, minimal styling
- Statistics box in corner
- Accessible color choices

## Requirements

- Python ≥3.12
- numpy ≥2.0.0
- scipy ≥1.15.1
- matplotlib ≥3.11.1
- langchain-anthropic ≥1.7.0
- langgraph ≥1.2.11

## Troubleshooting

### "Config file not found"
```bash
cp .config.example .config
# Edit .config with your API credentials
```

### "For .raw files, metadata must include 'dimensions'"
```python
# Add dimensions to metadata:
metadata={'dimensions': [256, 256, 256], 'dtype': 'uint8'}
```

### "Unsupported file format"
Convert to .npy:
```python
import numpy as np
volume = ...  # load your data
np.save('volume.npy', volume)
```

## See Also

- [README.md](README.md) - Main project documentation
- [example_volume_analysis.py](example_volume_analysis.py) - Example usage
- [download_and_analyze.py](download_and_analyze.py) - Real dataset examples
- [test_volume_agent.py](test_volume_agent.py) - Testing script
