# OntoVis

AI-powered 3D volume analysis and visualization library.

## Quick Start

```bash
uv pip install -e .
cp .config.example .config  # Add your API key
python main.py
```

## Features

- 🔬 **Volume Analysis** - AI identifies features from intensity distributions
- 🎨 **3D Rendering** - Natural language controlled visualization
- 🤖 **Smart Rendering** - Iterative vision-guided framing for optimal results
- 💬 **Interactive CLI** - Chat with AI that uses agents as tools

## Documentation

📚 **All documentation is in [docs/](docs/)**

- [docs/INDEX.md](docs/INDEX.md) - Documentation index
- [docs/README.md](docs/README.md) - Full project documentation  
- [docs/QUICKSTART_VOLUME.md](docs/QUICKSTART_VOLUME.md) - Quick reference
- [docs/RENDER_TOOL.md](docs/RENDER_TOOL.md) - Command-line rendering tool

## Command-Line Tools

```bash
# Render volume with specific camera angle (azimuth, elevation, distance)
python tools/render_volume.py data/volume.raw --azimuth 45 --elevation 30 --distance 2.0

# Find optimal camera angle by matching reference image
python tools/find_camera_angle.py data/volume.raw skull
```

## Examples

```python
# Analyze
from ontovis import VolumeAnalysisAgent
agent = VolumeAnalysisAgent()
agent.analyze_volume('data/volume.raw', 'CT scan of foot', 
                     metadata={'dimensions': [256,256,256], 'dtype': 'uint8'})

# Render
from ontovis import VolumeRenderAgent  
agent = VolumeRenderAgent()
agent.render('data/volume.raw', 'Show the bones',
             metadata={'dimensions': [256,256,256], 'dtype': 'uint8'})

# Smart Render (with automatic framing)
from ontovis import SmartVolumeRenderAgent
agent = SmartVolumeRenderAgent()
agent.render('data/volume.raw', 'Show the bones',
             metadata={'dimensions': [256,256,256], 'dtype': 'uint8'})
```

---

👉 **Start here: [docs/INDEX.md](docs/INDEX.md)**
