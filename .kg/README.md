# Knowledge Graph Directory

This directory contains all the knowledge that the agentic rendering system uses.

## Structure

```
.kg/
├── README.md                       # This file
├── graph.json                      # Structured knowledge (conventions, references, learned params)
├── images/                         # Reference images
│   ├── skull_front.jpg
│   └── skull_side.jpg
├── documentation/                  # Strategic knowledge (workflows, strategies)
│   └── rendering_strategies.md     # Angle-matching workflow, rendering rules
├── datasets/                       # Dataset-specific metadata
└── conventions/                    # Additional convention files
```

## What's Inside

### graph.json (Structured Data)
- **Conventions**: 1 rendering convention (bones → white)
- **Reference Renders**: 2 reference images (skull front/side views)
- **Rules**: 19 parsed rules (preferences, framing, camera positioning)
- **Learned Params**: Empty (will populate as you use the system)
- **Relationships**: 8 relationships between entities

### documentation/ (Strategic Knowledge)
- **rendering_strategies.md**: Complete rendering workflow including:
  - Rendering conventions
  - Camera positioning strategies
  - Framing and cropping rules
  - **Iterative angle-matching workflow** (coarse → fine search)
  - Vision model integration
  - Search termination conditions

### images/
- **skull_front.jpg** (400×320): Reference image for skull front view
- **skull_side.jpg** (791×629): Reference image for skull side view

## How Agents Use This

When you run:
```python
from ontovis import VolumeRenderAgent

agent = VolumeRenderAgent(kg_path='.kg')
result = agent.render(
    volume_path='data/volume.raw',
    prompt='Show me a skull from the front',
    metadata={'dimensions': [256, 256, 128], 'dtype': 'uint8'}
)
```

The agent:
1. ✅ Loads **structured data** from `graph.json` (conventions, references)
2. ✅ Loads **strategic documentation** from `documentation/rendering_strategies.md`
3. ✅ Includes both in the LLM prompt for intelligent parameter selection
4. ✅ Stores successful results back to `learned_params` for future use

## Growth Over Time

### Initially (now)
- 1 convention
- 2 reference images
- 19 rules
- 0 learned parameters

### After Use
The KG learns from successful renders:
```json
{
  "learned_params": {
    "vis_male": [{
      "reference_id": "skull_front_view",
      "final_angles": {"azimuth": 270, "elevation": 0, "roll": -90, "distance": 2.0},
      "match_score": 9.7,
      "iterations_taken": 15,
      "timestamp": "2026-09-11T13:45:00"
    }]
  }
}
```

Next time you ask for "skull front view", the agent starts with these known-good angles!

## Adding Knowledge

### Structured Data
```python
kg = MultimodalKnowledgeGraph('.kg')

# Add convention
kg.add_anatomical_convention(
    name="vessels",
    description="Blood vessels",
    color=[200, 50, 50],
    color_name="red"
)

# Add reference image
kg.add_reference_render(
    image_path="images/heart_view.jpg",
    label="heart anterior view",
    category="heart",
    quality="good"
)
```

### Strategic Documentation
```python
kg.add_documentation("advanced_lighting", """
## Advanced Lighting Techniques
- Use three-point lighting for complex structures
- Ambient: 0.2-0.3 for base illumination
- Key light: diffuse 0.7-0.8 from camera direction
- Fill light: specular 0.1-0.2 for highlights
""")
```

Or create files directly:
```bash
cat > .kg/documentation/my_strategies.md << 'EOF'
## My Custom Strategies
...
EOF
```

## Source Control

The `.kg/` directory can be:
- **Git-ignored**: If you treat it as derived/runtime data
- **Git-tracked**: If you want to preserve learned parameters across sessions

Recommendation:
```gitignore
# Ignore runtime KG (regenerate from source)
.kg/

# Except documentation (or track source in data/KG/ instead)
!.kg/documentation/
```

Keep source files in `data/KG/` and copy to `.kg/documentation/` for runtime use.

## Documentation

For more details, see:
- [docs/KG_INTEGRATION.md](../docs/KG_INTEGRATION.md) - Full integration guide
- [docs/ANGLE_MATCHING_KG.md](../docs/ANGLE_MATCHING_KG.md) - Angle-matching workflow details

## Summary

This KG is **self-contained** and **grows with use**. All rendering knowledge - both structured and strategic - lives here. Copy this folder to another machine, and all knowledge comes with it! 🎯
