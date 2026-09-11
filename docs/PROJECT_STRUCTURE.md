# OntoVis Project Structure

## Overview

OntoVis is an AI-powered 3D volume analysis and visualization library with professional software engineering practices including run management, structured logging, and organized artifact storage.

## Entry Point

**`main.py`** - Main entry point for the application

```bash
# Start interactive CLI
python main.py

# With custom run name
python main.py --run-name my_analysis

# Management commands
python main.py --list-runs
python main.py --latest
python main.py --cleanup 10
```

## Directory Structure

```
OntoVis/
├── main.py                       # Entry point
├── README.md                     # Quick start guide
├── PROJECT_STRUCTURE.md          # This file
├── pyproject.toml                # Project metadata and dependencies
├── .config.example               # Example configuration
├── .config                       # Your API keys (gitignored)
│
├── src/ontovis/                  # Main package
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Configuration loader
│   ├── run_manager.py            # Run management system
│   ├── interactive_cli.py        # Interactive CLI (promoted from examples/)
│   │
│   ├── volume_agent.py           # Volume analysis agent
│   ├── render_agent_v2.py        # Volume rendering agent
│   ├── smart_render_agent.py    # Smart render with vision feedback
│   ├── vision_agent.py           # Vision analysis agent
│   ├── multimodal_kg.py          # Knowledge graph
│   └── graph.py                  # Original KG (deprecated)
│
├── examples/                     # Example scripts
│   ├── example_kg_setup.py       # Setup KG with conventions
│   ├── example_kg_add_references.py
│   ├── example_smart_render.py
│   ├── example_smart_render_with_intermediates.py
│   └── example_vision.py
│
├── docs/                         # Documentation
│   ├── INDEX.md                  # Documentation index
│   ├── README.md                 # Full project documentation
│   ├── QUICKSTART_VOLUME.md      # Quick reference
│   ├── SMART_RENDER.md           # Smart rendering guide
│   ├── VIEWING_INTERMEDIATES.md  # Intermediate renders
│   ├── KNOWLEDGE_GRAPH.md        # KG documentation
│   └── RUN_MANAGEMENT.md         # Run management guide
│
├── runs/                         # Run outputs (auto-created)
│   ├── 20260909_143022/          # Individual run
│   │   ├── metadata.json
│   │   ├── SUMMARY.txt
│   │   ├── logs/
│   │   │   ├── run.log
│   │   │   └── operations.jsonl
│   │   ├── renders/
│   │   ├── histograms/
│   │   └── artifacts/
│   └── ...
│
├── .kg/                          # Knowledge graph storage
│   ├── graph.json
│   ├── images/
│   │   ├── skeleton/
│   │   ├── skull/
│   │   └── learned_renders/
│   └── ...
│
├── 3d_datasets/                  # Volume datasets
│   ├── vis_male_128x256x256_uint8.raw
│   ├── foot_256x256x256_uint8.raw
│   └── skull_256x256x256_uint8.raw
│
└── images/                       # Documentation images
```

## Core Components

### 1. Run Management (`src/ontovis/run_manager.py`)

Manages execution runs with logging and artifact organization.

**Classes:**
- `RunManager` - Manages a single run
- `RunRegistry` - Query and manage all runs

**Features:**
- Unique run IDs
- Structured logging (file + console)
- Organized artifact storage
- Metadata tracking
- Operations log (JSONL format)

### 2. Interactive CLI (`src/ontovis/interactive_cli.py`)

Chat-based interface with AI agents as tools.

**Tools:**
- `analyze_volume` - Volume analysis
- `render_volume` - Standard rendering
- `smart_render_volume` - Vision-guided rendering
- `analyze_image` - Vision AI analysis
- `list_files` - File browsing

### 3. Agents

#### Volume Analysis Agent (`volume_agent.py`)
- Histogram analysis
- Intensity statistics
- Feature identification
- AI-powered interpretation

#### Volume Render Agent (`render_agent_v2.py`)
- Natural language controlled rendering
- PyVista-based visualization
- Transfer function generation

#### Smart Render Agent (`smart_render_agent.py`)
- Iterative rendering with vision feedback
- Automatic framing optimization
- Saves intermediate renders (optional)

#### Vision Agent (`vision_agent.py`)
- Image analysis
- Entity extraction
- Vision-based quality assessment

### 4. Knowledge Graph (`multimodal_kg.py`)

Multimodal knowledge storage for rendering conventions.

**Stores:**
- Anatomical conventions (colors, intensities)
- Colormaps (bone_window, heat_map, etc.)
- Reference images (good/bad examples)
- Dataset knowledge
- Learned parameters

### 5. Configuration (`config.py`)

Loads API keys and settings from `.config` file.

## Workflow

### Typical Usage Flow

```
1. User runs: python main.py
   ↓
2. RunManager creates unique run directory
   ↓
3. Interactive CLI starts with logging
   ↓
4. User interacts with AI agents
   ↓
5. All renders → runs/RUN_ID/renders/
   All histograms → runs/RUN_ID/histograms/
   All logs → runs/RUN_ID/logs/
   ↓
6. User exits
   ↓
7. Run finalized with summary
```

### Example Session

```bash
$ python main.py --run-name skull_analysis

OntoVis v0.1.0
====================
Run initialized: 20260909_143022_skull_analysis
Run directory: /path/to/runs/20260909_143022_skull_analysis