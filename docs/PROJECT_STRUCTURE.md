# OntoVis Project Structure

## Overview

```
OntoVis/
├── src/ontovis/              # Core library package
├── examples/                 # Examples and test scripts
├── data/                     # Volume datasets (not in git)
├── *.md                      # Documentation
├── .config                   # API configuration (not in git)
└── pyproject.toml           # Project dependencies
```

## Core Library (`src/ontovis/`)

The main package with reusable components:

| File | Description |
|------|-------------|
| `__init__.py` | Package exports: KnowledgeGraph, VisionAgent, VolumeAnalysisAgent, Config |
| `config.py` | Configuration manager (reads `.config` file) |
| `graph.py` | KnowledgeGraph class for entity-relationship graphs |
| `visualizer.py` | GraphVisualizer for rendering knowledge graphs |
| `vision_agent.py` | VisionAgent for analyzing images with Claude |
| `volume_agent.py` | VolumeAnalysisAgent for 3D volume dataset analysis |

## Examples Directory (`examples/`)

All example scripts, tests, and utilities:

### Volume Analysis
| File | Purpose |
|------|---------|
| `analyze_foot.py` | **Universal CLI** - Analyze any volume dataset |
| `download_and_analyze.py` | Download and analyze scientific datasets |
| `example_volume_analysis.py` | Various volume analysis examples |
| `test_volume_agent.py` | Volume agent test suite |

### Other Examples
| File | Purpose |
|------|---------|
| `example_vision.py` | Vision agent examples |
| `main.py` | Basic knowledge graph example |
| `test_setup.py` | Verify configuration |

### Debug Tools
| File | Purpose |
|------|---------|
| `debug_api.py` | Test Anthropic API connection |
| `debug_api_bedrock.py` | Test Bedrock API connection |
| `diagnose_endpoint.py` | Diagnose endpoint issues |

## Documentation Files (Root)

| File | Description |
|------|-------------|
| `README.md` | Main project documentation |
| `VOLUME_ANALYSIS.md` | Complete volume analysis guide |
| `QUICKSTART_VOLUME.md` | Quick reference for volume analysis |
| `FORMATS.md` | Volume file format guide (.raw, .npy, .dat) |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `PROJECT_STRUCTURE.md` | This file |

## Configuration

| File | Purpose |
|------|---------|
| `.config.example` | Template configuration file |
| `.config` | Your actual API configuration (git-ignored) |
| `pyproject.toml` | Python dependencies and build config |
| `.gitignore` | Excludes config, datasets, generated files |

## Data Directory (`data/`)

Store volume datasets here (git-ignored):
- `.raw` files from scientific datasets
- `.npy` files from NumPy arrays
- `.dat` files with custom formats

## Usage Patterns

### Import the Library
```python
from ontovis import VolumeAnalysisAgent, VisionAgent, KnowledgeGraph
```

### Run Examples
```bash
# From project root
uv run python examples/test_volume_agent.py
uv run python examples/analyze_foot.py data/volume.raw -x 256 -y 256 -z 256 -d "description"
uv run python examples/download_and_analyze.py bonsai
```

### Test Setup
```bash
uv run python examples/test_setup.py
```

## Development Workflow

1. **Library code** goes in `src/ontovis/`
2. **Example scripts** go in `examples/`
3. **Documentation** goes in root `.md` files
4. **Data files** go in `data/` (git-ignored)
5. **Generated outputs** (histograms, etc.) stay in root or examples/ (git-ignored)

## File Organization Principles

- ✅ **Core library**: Reusable, well-tested components in `src/ontovis/`
- ✅ **Examples**: Self-contained scripts in `examples/`
- ✅ **Documentation**: Comprehensive guides in root directory
- ✅ **Clean root**: Minimal clutter, only essential config/docs
- ✅ **Git-ignored**: Config files, datasets, generated outputs

## Import Paths

From anywhere in the project:
```python
# Works because package is installed with `uv pip install -e .`
from ontovis import VolumeAnalysisAgent
from ontovis.config import Config
```

## Adding New Features

1. **New agent/component**: Add to `src/ontovis/`, export in `__init__.py`
2. **New example**: Add to `examples/`, document in `examples/README.md`
3. **New documentation**: Add markdown file in root directory
4. **New test**: Add test script to `examples/`
