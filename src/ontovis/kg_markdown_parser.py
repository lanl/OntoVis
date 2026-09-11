"""Markdown to Knowledge Graph parser.

Parse a structured markdown file and populate the knowledge graph.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


class KGMarkdownParser:
    """Parse markdown files and populate knowledge graph."""

    def __init__(self, kg):
        """Initialize parser.

        Args:
            kg: MultimodalKnowledgeGraph instance
        """
        self.kg = kg

    def parse_file(self, markdown_path: str) -> Dict[str, Any]:
        """Parse a markdown file and populate KG.

        Args:
            markdown_path: Path to markdown file

        Returns:
            Dict with counts of items added
        """
        content = Path(markdown_path).read_text()

        stats = {
            "conventions": 0,
            "colormaps": 0,
            "datasets": 0,
            "references": 0,
            "errors": []
        }

        # Split into sections
        sections = self._split_sections(content)

        for section_type, section_content in sections:
            try:
                if section_type == "convention":
                    self._parse_convention(section_content)
                    stats["conventions"] += 1
                elif section_type == "colormap":
                    self._parse_colormap(section_content)
                    stats["colormaps"] += 1
                elif section_type == "dataset":
                    self._parse_dataset(section_content)
                    stats["datasets"] += 1
                elif section_type == "reference":
                    self._parse_reference(section_content)
                    stats["references"] += 1
            except Exception as e:
                error_msg = f"Error parsing {section_type}: {e}"
                stats["errors"].append(error_msg)
                print(f"⚠ {error_msg}")

        return stats

    def _split_sections(self, content: str) -> List[tuple]:
        """Split content into sections based on headers.

        Args:
            content: Markdown content

        Returns:
            List of (section_type, content) tuples
        """
        sections = []
        current_type = None
        current_content = []

        for line in content.split('\n'):
            # Check for section headers
            if line.startswith('## Convention:') or line.startswith('## Anatomical Convention:'):
                if current_type:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = "convention"
                current_content = [line]
            elif line.startswith('## Colormap:'):
                if current_type:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = "colormap"
                current_content = [line]
            elif line.startswith('## Dataset:'):
                if current_type:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = "dataset"
                current_content = [line]
            elif line.startswith('## Reference:'):
                if current_type:
                    sections.append((current_type, '\n'.join(current_content)))
                current_type = "reference"
                current_content = [line]
            else:
                if current_type:
                    current_content.append(line)

        # Add last section
        if current_type:
            sections.append((current_type, '\n'.join(current_content)))

        return sections

    def _parse_convention(self, content: str):
        """Parse an anatomical convention section.

        Expected format:
        ## Convention: bone
        - Color: white [255, 255, 255]
        - Intensity: 180-254
        - Opacity: 0.8-1.0
        - Rationale: Doctors trained to see bones as white in X-rays/CT
        - Description: Bone tissue - highest density
        """
        lines = content.strip().split('\n')

        # Extract name from header
        header = lines[0]
        name = header.split(':', 1)[1].strip()

        # Parse fields
        data = {}
        description = ""

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('- Color:'):
                # Parse: "white [255, 255, 255]"
                color_part = line.split(':', 1)[1].strip()
                if '[' in color_part:
                    color_name = color_part.split('[')[0].strip()
                    rgb_str = color_part.split('[')[1].split(']')[0]
                    rgb = [int(x.strip()) for x in rgb_str.split(',')]
                    data['color'] = rgb
                    data['color_name'] = color_name
                else:
                    # Named color only
                    data['color_name'] = color_part
                    data['color'] = self._color_name_to_rgb(color_part)

            elif line.startswith('- Intensity:'):
                # Parse: "180-254"
                intensity_str = line.split(':', 1)[1].strip()
                if '-' in intensity_str:
                    min_val, max_val = intensity_str.split('-')
                    data['intensity_range'] = [float(min_val), float(max_val)]

            elif line.startswith('- Opacity:'):
                # Parse: "0.8-1.0"
                opacity_str = line.split(':', 1)[1].strip()
                if '-' in opacity_str:
                    min_val, max_val = opacity_str.split('-')
                    data['opacity_range'] = [float(min_val), float(max_val)]

            elif line.startswith('- Rationale:'):
                data['rationale'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Description:'):
                description = line.split(':', 1)[1].strip()

        # Add to KG
        self.kg.add_anatomical_convention(
            name=name,
            description=description or f"{name.title()} structure",
            color=data.get('color', [128, 128, 128]),
            color_name=data.get('color_name', 'gray'),
            intensity_range=data.get('intensity_range'),
            opacity_range=data.get('opacity_range', [0.5, 1.0]),
            medical_rationale=data.get('rationale'),
            references=None
        )

    def _parse_colormap(self, content: str):
        """Parse a colormap section.

        Expected format:
        ## Colormap: heat_map
        - Type: sequential
        - Description: Heat map for intensity visualization
        - Use cases: Density analysis, Intensity distribution
        - Colors:
          - [0, 0, 100] @ 0
          - [0, 255, 255] @ 64
          - [255, 0, 0] @ 255
        """
        lines = content.strip().split('\n')

        # Extract name
        name = lines[0].split(':', 1)[1].strip()

        data = {
            'colors': [],
            'intensity_mapping': [],
            'use_cases': []
        }

        in_colors = False
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('- Type:'):
                data['type'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Description:'):
                data['description'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Use cases:'):
                use_cases_str = line.split(':', 1)[1].strip()
                data['use_cases'] = [uc.strip() for uc in use_cases_str.split(',')]

            elif line.startswith('- Colors:'):
                in_colors = True

            elif in_colors and line.startswith('  - ['):
                # Parse: "  - [255, 0, 0] @ 255"
                parts = line[4:].split('@')
                rgb_str = parts[0].strip()[1:-1]  # Remove []
                rgb = [int(x.strip()) for x in rgb_str.split(',')]
                data['colors'].append(rgb)

                if len(parts) > 1:
                    intensity = float(parts[1].strip())
                    data['intensity_mapping'].append(intensity)

        # Add to KG
        self.kg.add_colormap_convention(
            name=name,
            colormap_type=data.get('type', 'sequential'),
            colors=data['colors'],
            description=data.get('description', ''),
            use_cases=data['use_cases'],
            intensity_mapping=data['intensity_mapping'] if data['intensity_mapping'] else None
        )

    def _parse_dataset(self, content: str):
        """Parse a dataset section.

        Expected format:
        ## Dataset: VIS_male_128
        - Path: 3d_datasets/vis_male.raw
        - Dimensions: 128, 256, 256
        - Dtype: uint8
        - Description: Visible Human Male full body
        - Anatomy: full_body
        - Modality: CT
        - Intensity ranges:
          - air: 0-15
          - soft_tissue: 40-120
          - bone: 180-254
        - Notes: Low Z resolution, bones may appear disconnected
        """
        lines = content.strip().split('\n')

        # Extract name
        name = lines[0].split(':', 1)[1].strip()

        data = {
            'intensity_ranges': {}
        }

        in_intensities = False
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('- Path:'):
                data['path'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Dimensions:'):
                dims_str = line.split(':', 1)[1].strip()
                data['dimensions'] = [int(x.strip()) for x in dims_str.split(',')]

            elif line.startswith('- Dtype:'):
                data['dtype'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Description:'):
                data['description'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Anatomy:'):
                data['anatomy'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Modality:'):
                data['modality'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Notes:'):
                data['notes'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Intensity ranges:'):
                in_intensities = True

            elif in_intensities and line.startswith('  - '):
                # Parse: "  - bone: 180-254"
                struct_part = line[4:].split(':')
                struct_name = struct_part[0].strip()
                range_str = struct_part[1].strip()
                if '-' in range_str:
                    min_val, max_val = range_str.split('-')
                    data['intensity_ranges'][struct_name] = [float(min_val), float(max_val)]

        # Add to KG
        self.kg.add_dataset_knowledge(
            dataset_name=name,
            dataset_path=data.get('path', ''),
            dimensions=data.get('dimensions', [256, 256, 256]),
            dtype=data.get('dtype', 'uint8'),
            description=data.get('description', ''),
            anatomy=data.get('anatomy', 'unknown'),
            modality=data.get('modality', 'CT'),
            typical_intensity_ranges=data['intensity_ranges'],
            known_good_params=None,
            notes=data.get('notes')
        )

    def _parse_reference(self, content: str):
        """Parse a reference image section.

        Expected format:
        ## Reference: good_skeleton_render
        - Image: path/to/image.png
        - Category: skeleton
        - Quality: good
        - Tags: well_framed, white_bones, clear_structure
        - Description: Excellent full skeleton render
        """
        lines = content.strip().split('\n')

        # Extract name (optional, used for ID)
        name = lines[0].split(':', 1)[1].strip()

        data = {}

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('- Image:'):
                data['image'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Category:'):
                data['category'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Quality:'):
                data['quality'] = line.split(':', 1)[1].strip()

            elif line.startswith('- Tags:'):
                tags_str = line.split(':', 1)[1].strip()
                data['tags'] = [t.strip() for t in tags_str.split(',')]

            elif line.startswith('- Description:'):
                data['description'] = line.split(':', 1)[1].strip()

        # Add to KG (only if image exists)
        if 'image' in data and Path(data['image']).exists():
            self.kg.add_reference_render(
                image_path=data['image'],
                category=data.get('category', 'general'),
                quality=data.get('quality', 'good'),
                tags=data.get('tags', []),
                description=data.get('description', ''),
                render_params=None,
                dataset_info=None
            )

    def _color_name_to_rgb(self, color_name: str) -> List[int]:
        """Convert color name to RGB.

        Args:
            color_name: Color name

        Returns:
            RGB list
        """
        colors = {
            'white': [255, 255, 255],
            'black': [0, 0, 0],
            'red': [220, 20, 20],
            'blue': [20, 20, 220],
            'green': [20, 220, 20],
            'yellow': [255, 220, 100],
            'orange': [255, 127, 0],
            'purple': [148, 0, 211],
            'pink': [255, 192, 203],
            'gray': [128, 128, 128],
            'brown': [139, 69, 19],
            'tan': [210, 180, 140]
        }
        return colors.get(color_name.lower(), [128, 128, 128])


def parse_kg_markdown(markdown_path: str, kg_path: str = ".kg") -> Dict[str, Any]:
    """Parse markdown file and populate knowledge graph.

    Args:
        markdown_path: Path to markdown file
        kg_path: Path to KG storage

    Returns:
        Statistics dictionary
    """
    from .multimodal_kg import MultimodalKnowledgeGraph

    kg = MultimodalKnowledgeGraph(kg_path=kg_path)
    parser = KGMarkdownParser(kg)

    print(f"Parsing: {markdown_path}")
    stats = parser.parse_file(markdown_path)

    print("\n" + "="*70)
    print("Import Complete")
    print("="*70)
    print(f"✓ Conventions: {stats['conventions']}")
    print(f"✓ Colormaps: {stats['colormaps']}")
    print(f"✓ Datasets: {stats['datasets']}")
    print(f"✓ References: {stats['references']}")

    if stats['errors']:
        print(f"\n⚠ Errors: {len(stats['errors'])}")
        for error in stats['errors']:
            print(f"  - {error}")

    print()
    kg.print_summary()

    return stats
