# OntoVis Examples

This directory contains examples and test scripts for OntoVis functionality.

## Quick Start

### Test Setup
```bash
# Verify your configuration
uv run python test_setup.py
```

## Volume Analysis Examples

### 1. Simple Test (Synthetic Data)
```bash
# Quick test with synthetic volume
uv run python test_volume_agent.py
```

### 2. Analyze Any Dataset
```bash
# Generic analysis script - works with any volume file
uv run python analyze_foot.py <path> [options]

# Example with .npy file (no metadata needed)
uv run python analyze_foot.py ../data/volume.npy -d "CT scan description"

# Example with .raw file (requires dimensions)
uv run python analyze_foot.py ../data/foot_256x256x256_uint8.raw \
    -x 256 -y 256 -z 256 --dtype uint8 \
    -d "Medical CT scan of human foot" \
    -n "Foot CT" -m "Medical CT"
```

**Options:**
- `-d, --description`: Description of dataset (required)
- `-x, -y, -z`: Dimensions for .raw files
- `--dtype`: Data type (default: uint8)
- `-n, --name`: Dataset name for plot
- `-m, --modality`: Imaging modality
- `-o, --output`: Output histogram path
- `--no-display`: Don't show histogram window

### 3. Download & Analyze Real Datasets
```bash
# Download and analyze datasets from open-scivis-datasets
uv run python download_and_analyze.py bonsai
uv run python download_and_analyze.py skull
uv run python download_and_analyze.py engine
uv run python download_and_analyze.py foot
uv run python download_and_analyze.py backpack

# List all available datasets
uv run python download_and_analyze.py list
```

### 4. Detailed Examples
```bash
# Run various example scenarios
uv run python example_volume_analysis.py
```

## Vision Agent Examples

### Basic Image Analysis
```bash
uv run python example_vision.py
```

## Knowledge Graph Example

### Simple Knowledge Graph
```bash
uv run python main.py
```

## Debug Tools

### API Connection Testing
```bash
# Test Anthropic API connection
uv run python debug_api.py

# Test Bedrock API connection
uv run python debug_api_bedrock.py

# Diagnose endpoint issues
uv run python diagnose_endpoint.py
```

## Files Overview

| File | Purpose |
|------|---------|
| `analyze_foot.py` | Universal volume analysis script with CLI |
| `download_and_analyze.py` | Download and analyze scientific datasets |
| `example_volume_analysis.py` | Volume analysis examples (synthetic data) |
| `test_volume_agent.py` | Volume agent test suite |
| `example_vision.py` | Vision agent examples |
| `main.py` | Basic knowledge graph example |
| `test_setup.py` | Configuration verification |
| `debug_api.py` | API connection debugging |
| `debug_api_bedrock.py` | Bedrock API debugging |
| `diagnose_endpoint.py` | Endpoint diagnostics |

## Typical Workflow

1. **Verify setup:**
   ```bash
   uv run python test_setup.py
   ```

2. **Quick test:**
   ```bash
   uv run python test_volume_agent.py
   ```

3. **Analyze your data:**
   ```bash
   uv run python analyze_foot.py your_data.raw -x 256 -y 256 -z 256 -d "Your description"
   ```

4. **Try real datasets:**
   ```bash
   uv run python download_and_analyze.py bonsai
   ```
