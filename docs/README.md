# OntoVis

A Python library for analyzing and visualizing 3D volume datasets with AI-powered agents.

## Features

- 🔬 **Volume Analysis Agent** - AI-powered feature identification from intensity distributions
- 🎨 **Volume Rendering Agent** - Natural language controlled 3D visualization  
- 💬 **Interactive CLI** - Chat with AI that uses agents as tools
- 📊 **Automatic Histograms** - Statistical analysis and peak detection
- 🤖 **LangGraph Workflows** - Structured agent orchestration

## Quick Start

```bash
# Install
uv pip install -e .

# Configure
cp .config.example .config
# Edit .config with your API credentials

# Test
python examples/interactive_cli.py
```

## Usage Examples

### Analyze a Volume Dataset

```python
from ontovis import VolumeAnalysisAgent

agent = VolumeAnalysisAgent()
results = agent.analyze_volume(
    volume_path='data/foot.raw',
    user_description='CT scan of foot',
    metadata={'dimensions': [256, 256, 256], 'dtype': 'uint8'},
    save_histogram='analysis.png'
)

print(results['feature_analysis'])
```

### Render 3D Visualization

```python
from ontovis import VolumeRenderAgent

agent = VolumeRenderAgent()
agent.render(
    volume_path='data/foot.raw',
    prompt='Show the bones',
    metadata={'dimensions': [256, 256, 256], 'dtype': 'uint8'},
    save_image='bones.png'
)
```

### Interactive Chat

```bash
python examples/interactive_cli.py
```

Then chat naturally:
- "What files do I have?"
- "Analyze the foot dataset"
- "Show me the bones"

## Documentation

All documentation is in the **[docs/](docs/)** folder:

- **[docs/INDEX.md](docs/INDEX.md)** - Documentation index
- **[docs/QUICKSTART_VOLUME.md](docs/QUICKSTART_VOLUME.md)** - Quick reference
- **[docs/VOLUME_ANALYSIS.md](docs/VOLUME_ANALYSIS.md)** - Analysis guide
- **[docs/VOLUME_RENDERING.md](docs/VOLUME_RENDERING.md)** - Rendering guide
- **[docs/INTERACTIVE_CLI.md](docs/INTERACTIVE_CLI.md)** - CLI guide
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues

## Project Structure

```
OntoVis/
├── src/ontovis/          # Core library
│   ├── volume_agent.py      # Analysis agent
│   ├── render_agent_v2.py   # Rendering agent
│   └── ...
├── examples/             # Example scripts
│   ├── interactive_cli.py   # Chat interface
│   ├── analyze_foot.py      # Analysis CLI
│   └── ...
├── docs/                 # Documentation
├── data/                 # Volume datasets
└── pyproject.toml
```

## Requirements

- Python ≥3.12
- Claude API access (via Anthropic or AWS Bedrock)

## License

See LICENSE file for details.

## Learn More

👉 **Start with [docs/INDEX.md](docs/INDEX.md)** for complete documentation.
