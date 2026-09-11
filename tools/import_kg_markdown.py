#!/usr/bin/env python3
"""
Import knowledge graph data from markdown file.

Usage:
    python tools/import_kg_markdown.py kg_data.md
    python tools/import_kg_markdown.py my_custom_data.md --kg-path .kg
"""

import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis.kg_natural_parser import parse_natural_language_kg


def main():
    parser = argparse.ArgumentParser(
        description="Import knowledge graph data from markdown file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s kg_data.md
  %(prog)s my_data.md --kg-path .kg

Markdown Format:
  ## Convention: structure_name
  - Color: color_name [R, G, B]
  - Intensity: min-max
  - Opacity: min-max
  - Rationale: Why this convention
  - Description: What this is

  ## Dataset: dataset_name
  - Path: path/to/file.raw
  - Dimensions: x, y, z
  - Dtype: uint8
  - Description: What this contains
  - Anatomy: body_part
  - Modality: CT
  - Intensity ranges:
    - structure1: min-max
    - structure2: min-max
  - Notes: Special notes

See kg_data.md for a complete template.
        """
    )

    parser.add_argument(
        'markdown_file',
        help='Path to markdown file'
    )

    parser.add_argument(
        '--kg-path',
        default='.kg',
        help='Path to knowledge graph storage (default: .kg)'
    )

    args = parser.parse_args()

    # Check file exists
    md_path = Path(args.markdown_file)
    if not md_path.exists():
        print(f"Error: File not found: {args.markdown_file}")
        sys.exit(1)

    print("="*70)
    print("Knowledge Graph Markdown Import")
    print("="*70)
    print()

    # Parse and import
    stats = parse_natural_language_kg(str(md_path), kg_path=args.kg_path, verbose=True)

    # Success message
    print("\n✓ Import completed successfully!")
    print(f"📁 Knowledge graph: {Path(args.kg_path).absolute()}")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
