#!/usr/bin/env python3
"""
Visualize the Knowledge Graph contents.

Usage:
    python tools/visualize_kg.py
    python tools/visualize_kg.py --kg-path .kg
    python tools/visualize_kg.py --show-images
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.multimodal_kg import MultimodalKnowledgeGraph


def visualize_kg(kg_path: str = ".kg", show_images: bool = False):
    """Visualize the knowledge graph contents.

    Args:
        kg_path: Path to KG directory
        show_images: Whether to display image previews
    """
    kg = MultimodalKnowledgeGraph(kg_path=kg_path)

    print("="*80)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("="*80)
    print(f"Location: {Path(kg_path).absolute()}")
    print()

    # ==================== CONVENTIONS ====================
    conventions = kg.graph.get("conventions", {})
    print("📚 ANATOMICAL CONVENTIONS")
    print("─"*80)

    if conventions:
        for name, conv in conventions.items():
            print(f"\n▸ {name.upper()}")
            print(f"  Description: {conv.get('description', 'N/A')}")

            color = conv.get('color', {})
            rgb = color.get('rgb', [])
            color_name = color.get('name', 'N/A')
            print(f"  Color: {color_name} RGB{rgb}")

            if conv.get('intensity_range'):
                print(f"  Intensity Range: {conv['intensity_range']}")

            opacity = conv.get('opacity_range', [])
            print(f"  Opacity Range: {opacity}")

            if conv.get('medical_rationale'):
                print(f"  Rationale: {conv['medical_rationale']}")
    else:
        print("  (No conventions defined)")

    # ==================== REFERENCES ====================
    references = kg.graph.get("reference_renders", {})
    print("\n\n🖼️  REFERENCE IMAGES")
    print("─"*80)

    if references:
        by_category = {}
        for ref_id, ref in references.items():
            cat = ref.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(ref)

        for category, refs in by_category.items():
            print(f"\n▸ Category: {category.upper()}")
            for ref in refs:
                print(f"\n  • {ref['id']}")
                print(f"    Path: {ref['image_path']}")
                print(f"    Exists: {'✓' if ref.get('image_exists') else '✗'}")
                print(f"    Quality: {ref.get('quality', 'N/A')}")
                print(f"    Tags: {', '.join(ref.get('tags', []))}")
                print(f"    Description: {ref.get('description', 'N/A')}")

                if show_images and ref.get('image_exists'):
                    full_path = Path(kg_path) / ref['image_path']
                    if full_path.exists():
                        # Try to display image info
                        try:
                            from PIL import Image
                            with Image.open(full_path) as img:
                                print(f"    Image size: {img.size[0]}x{img.size[1]}")
                                print(f"    Format: {img.format}")
                        except ImportError:
                            print(f"    (Install Pillow to see image details)")
                        except Exception as e:
                            print(f"    (Error reading image: {e})")
    else:
        print("  (No reference images)")

    # ==================== DATASETS ====================
    datasets = kg.graph.get("dataset_knowledge", {})
    print("\n\n💾 DATASETS")
    print("─"*80)

    if datasets:
        for ds_name, ds in datasets.items():
            print(f"\n▸ {ds_name}")
            print(f"  Path: {ds.get('dataset_path', 'N/A')}")
            print(f"  Dimensions: {ds.get('dimensions', 'N/A')}")
            print(f"  Data Type: {ds.get('dtype', 'N/A')}")
            print(f"  Anatomy: {ds.get('anatomy', 'N/A')}")
            print(f"  Modality: {ds.get('modality', 'N/A')}")

            if ds.get('intensity_ranges'):
                print(f"  Intensity Ranges:")
                for struct, range_val in ds['intensity_ranges'].items():
                    print(f"    - {struct}: {range_val}")

            if ds.get('notes'):
                print(f"  Notes: {ds['notes']}")
    else:
        print("  (No datasets defined)")

    # ==================== RULES ====================
    rules = kg.graph.get("rules", {})
    print("\n\n📋 RENDERING RULES")
    print("─"*80)

    if rules:
        for rule_type, rule_list in rules.items():
            print(f"\n▸ {rule_type.upper().replace('_', ' ')}")
            for i, rule in enumerate(rule_list, 1):
                print(f"\n  {i}. {rule['text']}")
                if rule.get('parsed_data'):
                    print(f"     Data: {rule['parsed_data']}")
    else:
        print("  (No rules defined)")

    # ==================== LEARNED PARAMS ====================
    learned = kg.graph.get("learned_params", {})
    print("\n\n🎯 LEARNED RENDERING PARAMETERS")
    print("─"*80)

    if learned:
        total = sum(len(params) for params in learned.values())
        print(f"  Total learned renders: {total}")

        for dataset_name, params_list in learned.items():
            print(f"\n▸ Dataset: {dataset_name}")
            for param in params_list[:3]:  # Show first 3
                print(f"  • {param.get('id', 'N/A')}")
                print(f"    Iterations: {param.get('iterations', 'N/A')}")
                print(f"    Quality: {param.get('quality', 'N/A')}")
    else:
        print("  (No learned parameters)")

    # ==================== STATISTICS ====================
    stats = kg.get_statistics()
    print("\n\n📊 STATISTICS")
    print("─"*80)
    print(f"  Conventions: {stats['total_conventions']}")
    print(f"  Colormaps: {stats['total_colormaps']}")
    print(f"  Reference Renders: {stats['total_reference_renders']}")
    print(f"  Datasets: {stats['total_datasets']}")
    print(f"  Learned Renders: {stats['total_learned_renders']}")
    print(f"  Relationships: {stats['total_relationships']}")

    print("\n" + "="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Knowledge Graph contents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Visualize default KG at .kg
  %(prog)s --show-images      # Show image details
  %(prog)s --kg-path custom   # Visualize KG at custom path
        """
    )

    parser.add_argument(
        '--kg-path',
        default='.kg',
        help='Path to knowledge graph directory (default: .kg)'
    )

    parser.add_argument(
        '--show-images',
        action='store_true',
        help='Show image details (requires Pillow)'
    )

    args = parser.parse_args()

    kg_path = Path(args.kg_path)
    if not kg_path.exists():
        print(f"Error: KG directory not found: {kg_path}")
        print(f"Run 'python tools/import_kg_markdown.py <file.md>' first to create a KG")
        sys.exit(1)

    graph_file = kg_path / "graph.json"
    if not graph_file.exists():
        print(f"Error: No graph.json found in {kg_path}")
        sys.exit(1)

    visualize_kg(str(kg_path), args.show_images)


if __name__ == "__main__":
    main()
