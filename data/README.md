# Data Directory

Place your volume datasets here for analysis.

## Analyze Existing Datasets

Use the universal analysis script:

```bash
# From project root
python examples/analyze_foot.py data/your_volume.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "Description of your dataset" \
    -n "Dataset Name" \
    --no-display
```

## Current Datasets

- `foot_256x256x256_uint8.raw` - Human foot CT scan

### Analyze the foot dataset:
```bash
python examples/analyze_foot.py data/foot_256x256x256_uint8.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "Medical CT scan of human foot showing bones, soft tissue, and air spaces" \
    -n "Human Foot CT" \
    -m "Medical CT"
```

## Adding New Datasets

### Option 1: Download Manually

If automatic download fails (SSL issues), download manually:

1. **Bonsai Tree** (256³, uint8, 16 MB)
   ```bash
   curl -L -k -o bonsai.raw.gz https://klacansky.com/open-scivis-datasets/bonsai/bonsai_256x256x256_uint8.raw.gz
   gunzip bonsai.raw.gz
   mv bonsai.raw data/
   ```

2. **Skull** (256³, uint8, 16 MB)
   ```bash
   curl -L -k -o skull.raw.gz https://klacansky.com/open-scivis-datasets/skull/skull_256x256x256_uint8.raw.gz
   gunzip skull.raw.gz
   mv skull.raw data/
   ```

3. **Engine** (256×256×128, uint8, 8 MB)
   ```bash
   curl -L -k -o engine.raw.gz https://klacansky.com/open-scivis-datasets/engine/engine_256x256x128_uint8.raw.gz
   gunzip engine.raw.gz
   mv engine.raw data/
   ```

Then analyze:
```bash
python examples/analyze_foot.py data/skull.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "Medical CT of human skull with bone and sinuses"
```

### Option 2: From Your Own Data

If you have your own volume data:

**NumPy format (.npy):**
```python
import numpy as np
volume = your_data_array  # shape: (x, y, z)
np.save('data/my_volume.npy', volume)
```

```bash
# Analyze (no dimensions needed for .npy)
python examples/analyze_foot.py data/my_volume.npy \
    -d "Your dataset description"
```

**Raw binary format (.raw):**
```python
import numpy as np
volume = your_data_array
volume.tofile('data/my_volume.raw')
# Remember the dimensions and dtype!
```

```bash
# Analyze (dimensions required for .raw)
python examples/analyze_foot.py data/my_volume.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "Your dataset description"
```

## Dataset Sources

- **Open SciVis Datasets**: http://klacansky.com/open-scivis-datasets/
- **Volume Library**: https://www.volvis.org/
- **NIST Scientific Datasets**: https://www.nist.gov/itl/math/mcsd/visualization-and-usability-group/datasets

## File Formats

| Format | Extension | Metadata Required |
|--------|-----------|-------------------|
| NumPy | `.npy` | None (embedded) |
| Raw Binary | `.raw` | dimensions, dtype |
| DAT | `.dat` | None (embedded) |

See [../FORMATS.md](../FORMATS.md) for details.
