# Troubleshooting Guide

## Common Issues and Solutions

### Font Warning

**Issue:**
```
findfont: Failed to find font weight 600, now using 700.
```

**What it means:**
Matplotlib couldn't find a semibold font (weight 600) and fell back to bold (700).

**Solution:**
✅ Fixed in latest version - code now uses `fontweight='bold'` instead of `fontweight='600'`

---

### SSL Certificate Errors

**Issue:**
```
URLError: <urlopen error [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol>
```

**What it means:**
Python's urllib has trouble with the SSL certificates on some servers.

**Solution:**
Use manual download:
```bash
cd data
curl -L -k -o dataset.raw.gz https://url/to/dataset.raw.gz
gunzip dataset.raw.gz
cd ..
python examples/analyze_foot.py data/dataset.raw -x 256 -y 256 -z 256 -d "description"
```

Or use `wget`:
```bash
wget --no-check-certificate -O dataset.raw.gz https://url/to/dataset.raw.gz
```

---

### Config File Not Found

**Issue:**
```
FileNotFoundError: Config file not found at .config
```

**Solution:**
```bash
cp .config.example .config
# Edit .config with your API credentials
```

---

### Raw File Dimensions Error

**Issue:**
```
ValueError: For .raw files, metadata must include 'dimensions'
```

**Solution:**
Always provide dimensions and dtype for .raw files:
```python
metadata = {
    'dimensions': [256, 256, 256],  # [x, y, z]
    'dtype': 'uint8'
}
```

Or use command line:
```bash
python examples/analyze_foot.py data/volume.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "description"
```

---

### Cannot Reshape Array

**Issue:**
```
ValueError: cannot reshape array of size X into shape (Y, Z, W)
```

**What it means:**
The file size doesn't match the specified dimensions.

**Solution:**
Check the actual file size and calculate correct dimensions:
```bash
ls -l data/volume.raw  # Check file size in bytes

# For uint8: size = x * y * z * 1
# For uint16: size = x * y * z * 2
# For float32: size = x * y * z * 4
```

Example:
```python
import os
filesize = os.path.getsize('data/volume.raw')
# If filesize is 16777216 bytes and dtype is uint8:
# 16777216 = 256 * 256 * 256 * 1
dimensions = [256, 256, 256]
```

---

### Import Error

**Issue:**
```
ModuleNotFoundError: No module named 'ontovis'
```

**Solution:**
Install the package in editable mode:
```bash
uv pip install -e .
```

Or if using regular pip:
```bash
pip install -e .
```

---

### Missing Dependencies

**Issue:**
```
ModuleNotFoundError: No module named 'scipy' (or numpy, matplotlib, etc.)
```

**Solution:**
Reinstall dependencies:
```bash
uv pip install -e .
```

This will install all required packages from `pyproject.toml`.

---

### Display Issues (No Display)

**Issue:**
```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

**What it means:**
Running on a system without a display (e.g., SSH session, headless server).

**Solution:**
Use `--no-display` flag:
```bash
python examples/analyze_foot.py data/volume.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "description" \
    --no-display
```

The histogram will still be saved to a PNG file, just not displayed interactively.

---

### API Connection Issues

**Issue:**
```
Connection refused / Timeout / Authentication error
```

**Solution:**
1. Check your `.config` file has correct credentials
2. Test the connection:
   ```bash
   uv run python examples/test_setup.py
   ```

3. Debug API connection:
   ```bash
   uv run python examples/debug_api.py
   ```

---

### Memory Issues with Large Datasets

**Issue:**
```
MemoryError: Unable to allocate array
```

**What it means:**
Dataset is too large to fit in RAM.

**Solution:**
- Use a machine with more RAM
- Process dataset in chunks (requires custom code)
- Downsample the volume:
  ```python
  import numpy as np
  volume = np.fromfile('large.raw', dtype='uint8').reshape(512, 512, 512)
  # Downsample by factor of 2
  downsampled = volume[::2, ::2, ::2]
  downsampled.tofile('smaller.raw')
  # New dimensions: [256, 256, 256]
  ```

---

## Getting Help

1. **Check documentation:**
   - [README.md](README.md) - Main documentation
   - [VOLUME_ANALYSIS.md](VOLUME_ANALYSIS.md) - Volume analysis guide
   - [FORMATS.md](FORMATS.md) - File formats
   - [examples/README.md](examples/README.md) - Example scripts

2. **Test your setup:**
   ```bash
   uv run python examples/test_setup.py
   uv run python examples/test_volume_agent.py
   ```

3. **Check file paths:**
   - Use absolute paths if relative paths don't work
   - Ensure files exist: `ls -l data/`

4. **Verify dependencies:**
   ```bash
   uv pip list | grep -E "numpy|scipy|matplotlib|langchain"
   ```

5. **Try a simple test:**
   ```bash
   uv run python -c "from ontovis import VolumeAnalysisAgent; print('OK')"
   ```
