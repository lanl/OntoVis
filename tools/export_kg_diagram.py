#!/usr/bin/env python3
"""
Export Knowledge Graph as a visual diagram.

Creates a visual representation showing:
- Conventions (nodes with colors)
- References (image thumbnails)
- Datasets
- Rules
- Relationships

Usage:
    python tools/export_kg_diagram.py
    python tools/export_kg_diagram.py --output kg_diagram.png
    python tools/export_kg_diagram.py --format svg
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.multimodal_kg import MultimodalKnowledgeGraph


def create_html_diagram(kg_path: str, output_path: str):
    """Create an HTML visualization of the KG.

    Args:
        kg_path: Path to KG directory
        output_path: Where to save HTML file
    """
    kg = MultimodalKnowledgeGraph(kg_path=kg_path)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .section {{
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .color-box {{
            display: inline-block;
            width: 40px;
            height: 40px;
            border-radius: 5px;
            border: 2px solid #333;
            vertical-align: middle;
            margin-right: 10px;
        }}
        .tag {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 12px;
            margin: 2px;
        }}
        .exists {{
            color: #27ae60;
            font-weight: bold;
        }}
        .missing {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .reference-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .reference-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .reference-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .rule {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card .number {{
            font-size: 36px;
            font-weight: bold;
        }}
        .stat-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Knowledge Graph Visualization</h1>
        <p><strong>Location:</strong> {Path(kg_path).absolute()}</p>
"""

    # Statistics
    stats = kg.get_statistics()
    html += """
        <h2>📈 Statistics</h2>
        <div class="stats">
"""

    stat_items = [
        ("Conventions", stats['total_conventions'], "📚"),
        ("Reference Images", stats['total_reference_renders'], "🖼️"),
        ("Datasets", stats['total_datasets'], "💾"),
        ("Learned Renders", stats['total_learned_renders'], "🎯"),
    ]

    for label, value, icon in stat_items:
        html += f"""
            <div class="stat-card">
                <div class="number">{icon} {value}</div>
                <div class="label">{label}</div>
            </div>
"""

    html += """
        </div>
"""

    # Conventions
    conventions = kg.graph.get("conventions", {})
    if conventions:
        html += """
        <h2>📚 Anatomical Conventions</h2>
        <div class="section">
"""
        for name, conv in conventions.items():
            color = conv.get('color', {})
            rgb = color.get('rgb', [128, 128, 128])
            color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

            html += f"""
            <div class="card">
                <h3>
                    <span class="color-box" style="background-color: {color_hex};"></span>
                    {name.upper()}
                </h3>
                <p><strong>Description:</strong> {conv.get('description', 'N/A')}</p>
                <p><strong>Color:</strong> {color.get('name', 'N/A')} RGB{rgb}</p>
                <p><strong>Opacity:</strong> {conv.get('opacity_range', 'N/A')}</p>
"""
            if conv.get('intensity_range'):
                html += f"                <p><strong>Intensity Range:</strong> {conv['intensity_range']}</p>\n"

            if conv.get('medical_rationale'):
                html += f"                <p><strong>Rationale:</strong> {conv['medical_rationale']}</p>\n"

            html += """
            </div>
"""

        html += """
        </div>
"""

    # References
    references = kg.graph.get("reference_renders", {})
    if references:
        html += """
        <h2>🖼️ Reference Images</h2>
        <div class="reference-grid">
"""
        for ref_id, ref in references.items():
            exists = ref.get('image_exists', False)
            status_class = "exists" if exists else "missing"
            status_text = "✓ Available" if exists else "✗ Missing"

            html += f"""
            <div class="reference-card">
                <h3>{ref['id'].replace('_', ' ').title()}</h3>
"""

            # Show image if exists
            if exists:
                img_path = Path(kg_path) / ref['image_path']
                if img_path.exists():
                    # Copy image to output directory or use relative path
                    html += f"""
                <img src="../{ref['image_path']}" alt="{ref['id']}">
"""

            html += f"""
                <p><strong>Status:</strong> <span class="{status_class}">{status_text}</span></p>
                <p><strong>Category:</strong> {ref.get('category', 'N/A')}</p>
                <p><strong>Quality:</strong> {ref.get('quality', 'N/A')}</p>
                <div>
"""

            for tag in ref.get('tags', []):
                html += f'                    <span class="tag">{tag}</span>\n'

            html += f"""
                </div>
                <p><small>{ref.get('description', '')}</small></p>
            </div>
"""

        html += """
        </div>
"""

    # Rules
    rules = kg.graph.get("rules", {})
    if rules:
        html += """
        <h2>📋 Rendering Rules</h2>
        <div class="section">
"""
        for rule_type, rule_list in rules.items():
            html += f"""
            <h3>{rule_type.replace('_', ' ').title()}</h3>
"""
            for i, rule in enumerate(rule_list, 1):
                html += f"""
            <div class="rule">
                <strong>{i}.</strong> {rule['text']}
            </div>
"""

        html += """
        </div>
"""

    # Datasets
    datasets = kg.graph.get("dataset_knowledge", {})
    if datasets:
        html += """
        <h2>💾 Datasets</h2>
        <div class="section">
"""
        for ds_name, ds in datasets.items():
            html += f"""
            <div class="card">
                <h3>{ds_name}</h3>
                <p><strong>Path:</strong> {ds.get('dataset_path', 'N/A')}</p>
                <p><strong>Dimensions:</strong> {ds.get('dimensions', 'N/A')}</p>
                <p><strong>Anatomy:</strong> {ds.get('anatomy', 'N/A')}</p>
                <p><strong>Modality:</strong> {ds.get('modality', 'N/A')}</p>
"""
            if ds.get('intensity_ranges'):
                html += "                <p><strong>Intensity Ranges:</strong></p>\n                <ul>\n"
                for struct, range_val in ds['intensity_ranges'].items():
                    html += f"                    <li>{struct}: {range_val}</li>\n"
                html += "                </ul>\n"

            html += """
            </div>
"""

        html += """
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    # Write HTML file
    output_file = Path(output_path)
    output_file.write_text(html)
    print(f"✓ Created HTML visualization: {output_file.absolute()}")
    print(f"  Open in browser: file://{output_file.absolute()}")

    return str(output_file.absolute())


def main():
    parser = argparse.ArgumentParser(
        description="Export Knowledge Graph as visual diagram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Export to kg_diagram.html
  %(prog)s --output my_kg.html          # Custom output name
  %(prog)s --kg-path custom             # Custom KG location
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
        default='kg_diagram.html',
        help='Output file path (default: kg_diagram.html)'
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

    create_html_diagram(str(kg_path), args.output)


if __name__ == "__main__":
    main()
