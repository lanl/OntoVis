# Volume Rendering Agent

The VolumeRenderAgent uses AI to interpret natural language prompts and generate 3D volume visualizations with appropriate rendering parameters.

## Overview

Traditional volume rendering requires manually tuning complex parameters (transfer functions, opacity curves, color maps). The VolumeRenderAgent automates this by:

1. Loading your volume data
2. Analyzing the intensity distribution
3. Using AI to interpret your prompt
4. Generating optimal rendering parameters
5. Creating a 3D visualization

## Quick Start

```python
from ontovis import VolumeRenderAgent

agent = VolumeRenderAgent()

results = agent.render(
    volume_path='data/foot.raw',
    prompt="Show the bones in white, make everything else transparent",
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8'
    },
    save_image='bones.png'
)
```

## CLI Usage

```bash
python examples/render_volume.py data/foot.raw \
    -x 256 -y 256 -z 256 \
    -p "Your rendering prompt here" \
    -o output.png
```

## Example Prompts

### Medical Imaging

**Show skeletal structure:**
```
"Show the bones in white and make everything else transparent"
```

**Bones and soft tissue:**
```
"Show bones in white and soft tissue in semi-transparent pink. Background should be black."
```

**Isolate dense structures:**
```
"Show only high-density structures like bones. Use grayscale."
```

### Scientific Visualization

**Density heatmap:**
```
"Create a heat map visualization showing density variations. Use blue-to-red color scale."
```

**Phase separation:**
```
"Highlight different material phases. Use different colors for low, medium, and high density regions."
```

**Structural analysis:**
```
"Show internal structures. Make outer layers semi-transparent and inner core opaque."
```

### Camera Views

**Top-down view:**
```
"Show bones from above. White bones on black background."
```

**Specific angle:**
```
"View from the side showing the profile. Transparent background."
```

## API Reference

### `VolumeRenderAgent.render()`

```python
results = agent.render(
    volume_path: str,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_image: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `volume_path` | str | Path to volume file (.raw, .npy, .dat) |
| `prompt` | str | Natural language rendering description |
| `metadata` | dict | Optional metadata (dimensions, dtype for .raw) |
| `save_image` | str | Optional output path for PNG |

**Returns:**

```python
{
    'rendered_image_base64': str,  # Base64-encoded PNG
    'render_params': {
        'camera_position': [x, y, z],
        'opacity_mapping': [...],
        'color_mapping': [...],
        'background_color': [r, g, b],
        'lighting': {...},
        'volume_scale': [sx, sy, sz]
    },
    'explanation': str  # Why these parameters were chosen
}
```

## Metadata Requirements

### .raw files (requires metadata):
```python
metadata = {
    'dimensions': [256, 256, 256],  # [x, y, z]
    'dtype': 'uint8'                # numpy dtype string
}
```

### .npy files (no metadata needed):
```python
# Dimensions and dtype are embedded
results = agent.render(
    volume_path='volume.npy',
    prompt="Your prompt"
)
```

## How It Works

### 1. Volume Analysis
The agent first analyzes your volume:
- Computes histogram and statistics
- Finds intensity peaks
- Calculates percentiles for dynamic range

### 2. AI Parameter Selection
Claude analyzes:
- Your prompt (what you want to see)
- Volume statistics (what's in the data)
- Best practices for the visualization type

And determines:
- Opacity transfer function (what's visible)
- Color transfer function (what colors to use)
- Camera position (viewing angle)
- Lighting parameters (shading)

### 3. Rendering
Uses PyVista (VTK) to:
- Apply transfer functions
- Position camera
- Render the volume
- Capture as PNG image

## Rendering Parameters

The AI generates these parameters automatically:

### Opacity Mapping
Controls visibility at each intensity:
```python
opacity_mapping = [
    {"value": 0, "opacity": 0.0},      # Transparent at low intensity
    {"value": 50, "opacity": 0.1},     # Slightly visible
    {"value": 100, "opacity": 0.5},    # Semi-transparent
    {"value": 200, "opacity": 1.0}     # Fully opaque at high intensity
]
```

### Color Mapping
Assigns colors to intensities:
```python
color_mapping = [
    {"value": 0, "color": [0, 0, 0]},      # Black (low)
    {"value": 128, "color": [128, 128, 128]},  # Gray (mid)
    {"value": 255, "color": [255, 255, 255]}   # White (high)
]
```

### Camera Position
Relative coordinates for viewing angle:
```python
camera_position = [1.5, 1.5, 1.5]  # Diagonal view
# [0, 0, 2.0]  # Front view
# [2.0, 0, 0]  # Side view
# [0, 2.0, 0]  # Top view
```

### Lighting
Surface shading parameters:
```python
lighting = {
    "ambient": 0.3,   # Base illumination
    "diffuse": 0.6,   # Directional light
    "specular": 0.3   # Shininess
}
```

## Examples

### Example 1: Medical - Show Bones

```python
from ontovis import VolumeRenderAgent

agent = VolumeRenderAgent()

results = agent.render(
    volume_path='data/foot_256x256x256_uint8.raw',
    prompt="Show the bones in bright white. Make soft tissue and background completely transparent so only the skeletal structure is visible.",
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8',
        'name': 'Foot CT',
        'modality': 'Medical CT'
    },
    save_image='foot_skeleton.png'
)

print(f"Why these parameters? {results['explanation']}")
```

### Example 2: Scientific - Heatmap

```python
results = agent.render(
    volume_path='composite_material.npy',
    prompt="Create a heat map visualization showing density distribution. Use a blue (low) to red (high) color gradient. Show the full volume with semi-transparency.",
    save_image='density_heatmap.png'
)
```

### Example 3: Multi-Material

```python
results = agent.render(
    volume_path='engine_scan.raw',
    prompt="Show different materials in different colors. Metal should be silver/gray and high opacity. Air channels should be transparent. Cooling fluid should be semi-transparent blue.",
    metadata={'dimensions': [256, 256, 128], 'dtype': 'uint8'},
    save_image='engine_materials.png'
)
```

### Example 4: Custom View

```python
results = agent.render(
    volume_path='rock_sample.raw',
    prompt="View from directly above. Show high-density minerals in yellow, medium density in green, low density transparent. Black background.",
    metadata={'dimensions': [512, 512, 512], 'dtype': 'uint16'},
    save_image='minerals_topview.png'
)
```

## Prompt Writing Tips

### Be Specific
✓ Good: "Show bones in white with 80% opacity, soft tissue in pink with 30% opacity"
✗ Vague: "Show the structure"

### Describe What You Want to See
✓ "Highlight high-density structures"
✓ "Make low-intensity values transparent"
✓ "Show only features above intensity 100"

### Specify Colors and Opacity
✓ "White bones on black background"
✓ "Semi-transparent red for tissue"
✓ "Grayscale with transparent background"

### Mention Viewing Angle
✓ "View from above"
✓ "Side profile view"
✓ "Diagonal perspective"

### Reference Intensity Ranges
✓ "Low intensity (<50) should be transparent"
✓ "High density regions (>150) in bright white"

## Advanced Usage

### Access Full Parameters

```python
results = agent.render(...)

# Inspect what the AI chose
import json
print(json.dumps(results['render_params'], indent=2))

# See the explanation
print(results['explanation'])
```

### Save Parameters for Reuse

```python
import json

results = agent.render(...)

# Save parameters
with open('render_params.json', 'w') as f:
    json.dump(results['render_params'], f, indent=2)

# Later: Load and modify
with open('render_params.json') as f:
    params = json.load(f)
    
# Manually adjust and re-render
# (requires direct PyVista usage)
```

## Troubleshooting

### "No module named 'pyvista'"
```bash
uv pip install -e .  # Reinstall with all dependencies
```

### Off-screen Rendering Issues
PyVista uses off-screen rendering. On some systems you may need:
```bash
# Linux: Install OSMesa or Xvfb
sudo apt-get install libgl1-mesa-glx xvfb

# Or use Xvfb wrapper
xvfb-run -a python examples/render_volume.py ...
```

### Rendering is Black
- Check your volume data range
- Try: "Show all structures with full opacity for debugging"
- Verify volume loaded correctly (check shape in logs)

### Slow Rendering
- Large volumes (>512³) take time
- Consider downsampling first
- Use simpler prompts initially

## Requirements

- Python ≥3.12
- pyvista ≥0.45.0 (volume rendering)
- numpy, scipy, pillow
- langchain-anthropic, langgraph (AI control)

## See Also

- [VOLUME_ANALYSIS.md](VOLUME_ANALYSIS.md) - Analyze before rendering
- [examples/example_volume_render.py](examples/example_volume_render.py) - More examples
- [examples/render_volume.py](examples/render_volume.py) - CLI tool
