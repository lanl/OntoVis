# Volume File Format Guide

## Quick Comparison

| Format | Extension | Metadata Required | Best For | Example |
|--------|-----------|-------------------|----------|---------|
| **NumPy** | `.npy` | None | Python workflows | `np.save('vol.npy', array)` |
| **Raw Binary** | `.raw` | dimensions, dtype | Scientific datasets | `array.tofile('vol.raw')` |
| **DAT** | `.dat` | None | Custom formats | Structured binary |

## Format Details

### 1. NumPy Binary (.npy)

**Easiest to use** - no metadata required.

#### Creating:
```python
import numpy as np
volume = np.zeros((256, 256, 256), dtype=np.uint8)
np.save('volume.npy', volume)
```

#### Analyzing:
```python
from ontovis import VolumeAnalysisAgent

agent = VolumeAnalysisAgent()
results = agent.analyze_volume(
    volume_path='volume.npy',
    user_description='Your dataset description'
)
```

**Pros:**
- No metadata needed
- Includes shape and dtype automatically
- Fast to load
- Python-native format

**Cons:**
- Not readable by non-Python tools
- Larger file size than raw

---

### 2. Raw Binary (.raw)

**Most common** for scientific volume datasets.

#### Creating:
```python
import numpy as np
volume = np.zeros((256, 256, 256), dtype=np.uint8)
volume.tofile('volume.raw')  # Binary dump
```

#### Analyzing:
```python
from ontovis import VolumeAnalysisAgent

agent = VolumeAnalysisAgent()
results = agent.analyze_volume(
    volume_path='volume.raw',
    user_description='Your dataset description',
    metadata={
        'dimensions': [256, 256, 256],  # REQUIRED
        'dtype': 'uint8'                # REQUIRED
    }
)
```

**Pros:**
- Universal format (works with ParaView, ImageJ, etc.)
- Smallest file size
- No header overhead
- Standard for scientific datasets

**Cons:**
- Requires external metadata (dimensions, dtype)
- No built-in validation

#### Supported dtypes:
- `'uint8'` - 8-bit unsigned integer (0-255)
- `'uint16'` - 16-bit unsigned integer (0-65535)
- `'int16'` - 16-bit signed integer
- `'float32'` - 32-bit floating point
- `'float64'` - 64-bit floating point

---

### 3. DAT Format (.dat)

**Custom structured format** with embedded dimensions.

#### Structure:
```
[3 x uint32: dimensions] [volume data]
```

#### Creating:
```python
import struct
import numpy as np

volume = np.zeros((256, 256, 256), dtype=np.uint8)

with open('volume.dat', 'wb') as f:
    # Write dimensions as 3 uint32 values
    f.write(struct.pack('III', *volume.shape))
    # Write volume data
    volume.tofile(f)
```

#### Analyzing:
```python
agent = VolumeAnalysisAgent()
results = agent.analyze_volume(
    volume_path='volume.dat',
    user_description='Your dataset description'
    # No metadata needed - dimensions are in the file
)
```

---

## Converting Between Formats

### Raw → NumPy
```python
import numpy as np

# Load raw file with known dimensions
volume = np.fromfile('data.raw', dtype=np.uint8)
volume = volume.reshape((256, 256, 256))

# Save as .npy
np.save('data.npy', volume)
```

### NumPy → Raw
```python
import numpy as np

# Load .npy file
volume = np.load('data.npy')

# Save as raw
volume.tofile('data.raw')
# Remember to document dimensions and dtype!
```

### Any Format → DAT
```python
import numpy as np
import struct

# Load volume (from any source)
volume = np.load('data.npy')  # or np.fromfile(...).reshape(...)

# Write as DAT
with open('data.dat', 'wb') as f:
    f.write(struct.pack('III', *volume.shape))
    volume.tofile(f)
```

---

## Real Dataset Examples

### Open SciVis Datasets (Raw Format)

All datasets from http://klacansky.com/open-scivis-datasets/ are in raw format:

```python
# Download and analyze in one command:
# python download_and_analyze.py bonsai

# Or manually:
agent = VolumeAnalysisAgent()

results = agent.analyze_volume(
    volume_path='bonsai_256x256x256_uint8.raw',
    user_description='''
    Micro-CT scan of a bonsai tree showing wood structure,
    bark, and air spaces.
    ''',
    metadata={
        'name': 'Bonsai Tree',
        'dimensions': [256, 256, 256],
        'dtype': 'uint8',
        'modality': 'micro-CT'
    }
)
```

Available datasets:
- **Bonsai**: 256³, uint8 - Tree structure
- **Engine**: 256×256×128, uint8 - Engine block
- **Foot**: 256³, uint8 - Human foot
- **Skull**: 256³, uint8 - Human skull
- **Backpack**: 512×512×373, uint16 - Security scan

---

## Format Selection Guide

### Choose NumPy (.npy) when:
- ✓ Working primarily in Python
- ✓ You want the simplest workflow
- ✓ File size is not critical
- ✓ Converting from other Python arrays

### Choose Raw Binary (.raw) when:
- ✓ Working with standard scientific datasets
- ✓ Need compatibility with multiple tools
- ✓ Minimizing file size is important
- ✓ Following field conventions (medical imaging, etc.)

### Choose DAT (.dat) when:
- ✓ You need embedded metadata
- ✓ Raw format but don't want separate dimension files
- ✓ Custom pipeline requirements

---

## Common Issues

### "For .raw files, metadata must include 'dimensions'"
```python
# Fix: Add dimensions and dtype to metadata
metadata = {
    'dimensions': [256, 256, 256],  # [x, y, z]
    'dtype': 'uint8'
}
```

### "Cannot reshape array"
```python
# Problem: Wrong dimensions for raw file
# Check file size matches expected dimensions:
import os
filesize = os.path.getsize('volume.raw')
expected = np.prod([256, 256, 256]) * np.dtype('uint8').itemsize
print(f"File: {filesize}, Expected: {expected}")
```

### "Data type not understood"
```python
# Fix: Use numpy dtype strings
dtype='uint8'    # ✓ Correct
dtype='unsigned char'  # ✗ Wrong
```

---

## Testing

Test all formats with the example scripts:

```bash
# NumPy format
uv run python example_volume_analysis.py

# Raw format  
uv run python test_volume_agent.py

# Real datasets (raw format with metadata)
uv run python download_and_analyze.py bonsai
```
