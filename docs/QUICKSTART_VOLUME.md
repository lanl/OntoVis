# Volume Analysis Agent - Quick Start

## 30-Second Start

```bash
# Install dependencies
uv pip install -e .

# Test with synthetic data
uv run python examples/test_volume_agent.py

# Analyze a real dataset
uv run python examples/download_and_analyze.py bonsai
```

## 2-Minute Start

```python
from ontovis import VolumeAnalysisAgent
import numpy as np

# 1. Create or load volume data
volume = np.load('mydata.npy')  # or use .raw with metadata

# 2. Initialize agent
agent = VolumeAnalysisAgent()

# 3. Analyze
results = agent.analyze_volume(
    volume_path='mydata.npy',
    user_description='Description of what this data contains',
    save_histogram='analysis.png'
)

# 4. View results
print(results['feature_analysis'])
```

## Format Cheat Sheet

| Format | Extension | Metadata Needed | Example |
|--------|-----------|-----------------|---------|
| NumPy | `.npy` | None | `np.load('data.npy')` |
| Raw Binary | `.raw` | dimensions, dtype | `{'dimensions': [256,256,256], 'dtype': 'uint8'}` |
| DAT | `.dat` | None | Auto-detected |

## Real Dataset Examples

```bash
# Download & analyze in one command:
uv run python examples/download_and_analyze.py bonsai    # Tree structure
uv run python examples/download_and_analyze.py skull     # Medical CT
uv run python examples/download_and_analyze.py engine    # Industrial CT
uv run python examples/download_and_analyze.py foot      # Medical CT
uv run python examples/download_and_analyze.py backpack  # Security scan

# List all available
uv run python examples/download_and_analyze.py list
```

## Common Use Cases

### Medical Imaging
```python
results = agent.analyze_volume(
    volume_path='ct_scan.raw',
    user_description='CT scan showing bone, soft tissue, and air',
    metadata={
        'dimensions': [512, 512, 300],
        'dtype': 'uint16',
        'modality': 'CT'
    }
)
```

### Materials Science
```python
results = agent.analyze_volume(
    volume_path='composite.npy',
    user_description='X-ray CT of composite with matrix and fibers',
    metadata={'modality': 'X-ray CT'}
)
```

### Scientific Simulation
```python
results = agent.analyze_volume(
    volume_path='simulation.raw',
    user_description='Fluid dynamics simulation showing density field',
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'float32'
    }
)
```

## What You Get

### 1. AI Analysis
Identifies features like:
- "Background/air (0-20)"
- "Soft tissue (50-100)"
- "Bone (150-255)"
- Suggested thresholds for segmentation
- Visualization recommendations

### 2. Statistics
```python
results['volume_stats'] = {
    'min': 0.0,
    'max': 255.0,
    'mean': 87.3,
    'median': 82.0,
    'std': 45.2,
    'shape': (256, 256, 256),
    'total_voxels': 16777216
}
```

### 3. Histogram Plot
Professional visualization saved as PNG

### 4. Peak Detection
```python
results['histogram_data']['peaks'] = [15.3, 85.7, 180.2]
```

## Tips for Best Results

✓ **Be specific in descriptions**
```python
# Good
"CT scan of femur bone showing cortical bone, trabecular bone, and marrow"

# Not as good
"A bone scan"
```

✓ **Include relevant metadata**
```python
metadata={
    'name': 'Sample A',
    'modality': 'CT',
    'resolution': '0.5mm/voxel',
    'expected_materials': 'aluminum, steel, plastic'
}
```

✓ **Use .npy for simplicity**
```python
np.save('mydata.npy', volume_array)  # No metadata needed
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Config file not found" | `cp .config.example .config` and edit |
| ".raw needs dimensions" | Add `{'dimensions': [x,y,z], 'dtype': '...'}` |
| "Unsupported format" | Convert to .npy: `np.save('data.npy', array)` |
| Import error | `uv pip install -e .` |

## Next Steps

- Read [VOLUME_ANALYSIS.md](VOLUME_ANALYSIS.md) for full documentation
- Check [example_volume_analysis.py](example_volume_analysis.py) for detailed examples
- See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details

## One-Liners

```bash
# Test everything works
uv run python -c "from ontovis import VolumeAnalysisAgent; print('✓ Ready!')"

# Quick synthetic test
uv run python examples/test_volume_agent.py

# Download bonsai dataset
uv run python examples/download_and_analyze.py bonsai

# List what's available
uv run python examples/download_and_analyze.py list
```

---

**That's it!** The agent handles the rest - loading, analyzing, visualizing, and interpreting your volume data.
