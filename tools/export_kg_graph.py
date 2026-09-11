#!/usr/bin/env python3
"""
Export Knowledge Graph as an actual graph visualization with nodes and edges.

Creates an interactive network graph showing:
- Nodes: Conventions, References, Datasets, Rules
- Edges: Relationships between them

Usage:
    python tools/export_kg_graph.py
    python tools/export_kg_graph.py --output kg_graph.html
"""

import sys
import argparse
from pathlib import Path
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.multimodal_kg import MultimodalKnowledgeGraph


def infer_relationships(kg):
    """Infer relationships from the data.

    Returns:
        List of (source, target, relationship_type) tuples
    """
    relationships = []

    conventions = kg.graph.get("conventions", {})
    references = kg.graph.get("reference_renders", {})
    datasets = kg.graph.get("dataset_knowledge", {})
    rules = kg.graph.get("rules", {})

    # Link references to conventions by category
    for ref_id, ref in references.items():
        category = ref.get('category', '')

        # Try to match category to conventions
        for conv_name in conventions.keys():
            if conv_name.lower() in category.lower() or category.lower() in conv_name.lower():
                relationships.append((ref_id, conv_name, "shows"))

        # If category contains structure names, link those
        for struct in ['bone', 'skull', 'teeth', 'tissue', 'artery', 'vein']:
            if struct in category.lower():
                for conv_name in conventions.keys():
                    if struct in conv_name.lower():
                        relationships.append((ref_id, conv_name, "depicts"))

    # Link datasets to conventions based on intensity ranges
    for ds_name, ds in datasets.items():
        intensity_ranges = ds.get('intensity_ranges', {})
        for struct in intensity_ranges.keys():
            for conv_name in conventions.keys():
                if struct.lower() in conv_name.lower() or conv_name.lower() in struct.lower():
                    relationships.append((ds_name, conv_name, "contains"))

    # Link rules to conventions based on content
    for rule_type, rule_list in rules.items():
        for rule in rule_list:
            text = rule.get('text', '').lower()
            for conv_name in conventions.keys():
                if conv_name.lower() in text:
                    rule_id = f"rule_{rule_type}_{hash(rule['text']) % 1000}"
                    relationships.append((rule_id, conv_name, "applies_to"))

    return relationships


def create_graph_html(kg_path: str, output_path: str):
    """Create an interactive network graph using vis.js.

    Args:
        kg_path: Path to KG directory
        output_path: Where to save HTML file
    """
    kg = MultimodalKnowledgeGraph(kg_path=kg_path)

    # Build nodes
    nodes = []
    edges = []

    # Convention nodes (center)
    conventions = kg.graph.get("conventions", {})
    for name, conv in conventions.items():
        color = conv.get('color', {})
        rgb = color.get('rgb', [150, 150, 150])
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

        nodes.append({
            "id": name,
            "label": name.upper(),
            "title": f"{name}<br>Color: {color.get('name', 'N/A')}<br>Opacity: {conv.get('opacity_range')}",
            "group": "convention",
            "color": hex_color,
            "shape": "box",
            "size": 30
        })

    # Reference nodes
    references = kg.graph.get("reference_renders", {})
    for ref_id, ref in references.items():
        exists = ref.get('image_exists', False)
        color = "#27ae60" if exists else "#e74c3c"

        nodes.append({
            "id": ref_id,
            "label": ref_id.replace('_', ' ').title(),
            "title": f"{ref.get('description', '')}<br>Category: {ref.get('category')}<br>Exists: {'Yes' if exists else 'No'}",
            "group": "reference",
            "color": color,
            "shape": "image",
            "image": f"../{ref['image_path']}" if exists else None,
            "size": 40
        })

    # Dataset nodes
    datasets = kg.graph.get("dataset_knowledge", {})
    for ds_name, ds in datasets.items():
        nodes.append({
            "id": ds_name,
            "label": ds_name,
            "title": f"Anatomy: {ds.get('anatomy')}<br>Modality: {ds.get('modality')}<br>Dims: {ds.get('dimensions')}",
            "group": "dataset",
            "color": "#9b59b6",
            "shape": "database",
            "size": 30
        })

    # Rule nodes
    rules = kg.graph.get("rules", {})
    for rule_type, rule_list in rules.items():
        for i, rule in enumerate(rule_list):
            rule_id = f"rule_{rule_type}_{i}"
            nodes.append({
                "id": rule_id,
                "label": f"{rule_type}",
                "title": rule.get('text', ''),
                "group": "rule",
                "color": "#f39c12",
                "shape": "diamond",
                "size": 20
            })

    # Infer relationships
    relationships = infer_relationships(kg)

    # Explicit relationships from KG
    for rel in kg.graph.get("relationships", []):
        relationships.append((rel['source'], rel['target'], rel['relationship']))

    # Add edges
    edge_id = 0
    seen_edges = set()
    for source, target, rel_type in relationships:
        edge_key = (source, target)
        if edge_key not in seen_edges:
            edges.append({
                "id": edge_id,
                "from": source,
                "to": target,
                "label": rel_type,
                "arrows": "to",
                "color": {"color": "#95a5a6", "highlight": "#3498db"},
                "font": {"size": 10, "align": "middle"}
            })
            seen_edges.add(edge_key)
            edge_id += 1

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph - Network View</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        #graph {{
            width: 100%;
            height: 800px;
            border: 1px solid #ddd;
        }}
        #info {{
            padding: 20px;
            background: #f8f9fa;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            padding: 15px;
            background: white;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .stats {{
            text-align: center;
            padding: 10px;
            background: white;
            margin: 10px;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🕸️ Knowledge Graph - Network Visualization</h1>
        <p>Interactive graph showing relationships between conventions, references, datasets, and rules</p>
    </div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-box" style="background: #3498db;"></div>
            <span>Conventions</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #27ae60;"></div>
            <span>References (exists)</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #e74c3c;"></div>
            <span>References (missing)</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #9b59b6;"></div>
            <span>Datasets</span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background: #f39c12;"></div>
            <span>Rules</span>
        </div>
    </div>

    <div class="stats">
        <strong>Nodes:</strong> {len(nodes)} |
        <strong>Edges:</strong> {len(edges)} |
        <strong>Conventions:</strong> {len(conventions)} |
        <strong>References:</strong> {len(references)} |
        <strong>Datasets:</strong> {len(datasets)}
    </div>

    <div id="graph"></div>

    <div id="info">
        <h3>💡 Tips</h3>
        <ul>
            <li><strong>Click & drag</strong> nodes to rearrange</li>
            <li><strong>Scroll</strong> to zoom in/out</li>
            <li><strong>Hover</strong> over nodes for details</li>
            <li><strong>Click</strong> nodes to highlight connections</li>
        </ul>
    </div>

    <script type="text/javascript">
        // Create nodes and edges
        var nodes = new vis.DataSet({json.dumps(nodes, indent=2)});

        var edges = new vis.DataSet({json.dumps(edges, indent=2)});

        // Create network
        var container = document.getElementById('graph');
        var data = {{
            nodes: nodes,
            edges: edges
        }};

        var options = {{
            nodes: {{
                font: {{
                    size: 14,
                    color: '#333'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                width: 2,
                smooth: {{
                    type: 'continuous'
                }},
                shadow: true
            }},
            physics: {{
                stabilization: {{
                    iterations: 150
                }},
                barnesHut: {{
                    gravitationalConstant: -2000,
                    springLength: 200,
                    springConstant: 0.04
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                navigationButtons: true,
                keyboard: true
            }}
        }};

        var network = new vis.Network(container, data, options);

        // Highlight connections on click
        network.on("selectNode", function(params) {{
            var nodeId = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(nodeId);
            var connectedEdges = network.getConnectedEdges(nodeId);

            // Highlight connected
            network.selectNodes(connectedNodes);
            network.selectEdges(connectedEdges);
        }});
    </script>
</body>
</html>
"""

    # Write file
    output_file = Path(output_path)
    output_file.write_text(html)
    print(f"✓ Created network graph: {output_file.absolute()}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Open: file://{output_file.absolute()}")

    return str(output_file.absolute())


def main():
    parser = argparse.ArgumentParser(
        description="Export Knowledge Graph as interactive network visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Export to kg_graph.html
  %(prog)s --output my_graph.html       # Custom output
        """
    )

    parser.add_argument(
        '--kg-path',
        default='.kg',
        help='Path to knowledge graph directory (default: .kg)'
    )

    parser.add_argument(
        '--output',
        '-o',
        default='kg_graph.html',
        help='Output file path (default: kg_graph.html)'
    )

    args = parser.parse_args()

    kg_path = Path(args.kg_path)
    if not kg_path.exists():
        print(f"Error: KG directory not found: {kg_path}")
        sys.exit(1)

    graph_file = kg_path / "graph.json"
    if not graph_file.exists():
        print(f"Error: No graph.json found in {kg_path}")
        sys.exit(1)

    create_graph_html(str(kg_path), args.output)


if __name__ == "__main__":
    main()
