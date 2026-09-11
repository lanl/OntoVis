"""Natural language markdown parser for Knowledge Graph.

Parses bullet-point style natural language into KG entries.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class KGNaturalParser:
    """Parse natural language markdown into knowledge graph."""

    def __init__(self, kg):
        """Initialize parser.

        Args:
            kg: MultimodalKnowledgeGraph instance
        """
        self.kg = kg
        self.rules = []  # Store parsed rules

    def parse_file(self, markdown_path: str) -> Dict[str, Any]:
        """Parse a natural language markdown file.

        Args:
            markdown_path: Path to markdown file

        Returns:
            Dict with parsing results
        """
        content = Path(markdown_path).read_text()

        stats = {
            "conventions": 0,
            "references": 0,
            "rules": 0,
            "parsed_items": []
        }

        # Parse line by line
        lines = content.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Remove leading dash
            if line.startswith('- '):
                line = line[2:]

            # Try to parse this line
            result = self._parse_line(line)
            if result:
                stats["parsed_items"].append(result)

                if result['type'] == 'convention':
                    stats["conventions"] += 1
                elif result['type'] == 'reference':
                    stats["references"] += 1
                elif result['type'] == 'rule':
                    stats["rules"] += 1

        # Store all rules in the KG
        if self.rules:
            if "rules" not in self.kg.graph:
                self.kg.graph["rules"] = {}

            # Group rules by type
            for rule in self.rules:
                rule_type = rule['type']
                if rule_type not in self.kg.graph["rules"]:
                    self.kg.graph["rules"][rule_type] = []

                self.kg.graph["rules"][rule_type].append({
                    "text": rule['text'],
                    "parsed_data": {k: v for k, v in rule.items() if k not in ['type', 'text']},
                    "added": datetime.now().isoformat()
                })

            self.kg.graph["last_modified"] = datetime.now().isoformat()
            self.kg._save_graph()

        return stats

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single line and extract meaning.

        Args:
            line: Text line

        Returns:
            Parsed result dict or None
        """
        # Pattern: "when displaying X, make them Y"
        # Example: "when displaying bones, make them white"
        match = re.match(r'when displaying ([\w\s]+), make them (\w+)', line, re.IGNORECASE)
        if match:
            structure = match.group(1).strip()
            color_name = match.group(2).strip()

            # Add convention
            self._add_color_convention(structure, color_name)

            return {
                'type': 'convention',
                'structure': structure,
                'color': color_name,
                'text': line
            }

        # Pattern: "X shows the reference image for Y from Z view"
        # Example: "images/skull_front.jpg shows the reference image for a skull from a front view"
        # Also handles: "X shows the reference image from for Y from Z view" (typo with "from for")
        match = re.match(r'([\w/\.]+)\s+shows?\s+(?:the\s+)?reference image (?:from\s+)?(?:for|of)\s+([\w\s]+?)(?:\s+from|\s+in)\s+(?:a\s+)?([\w\s]+?)(?:\s+view)?$', line, re.IGNORECASE)
        if match:
            image_path = match.group(1).strip()
            subject = match.group(2).strip()
            view = match.group(3).strip()

            # Add reference
            self._add_reference(image_path, subject, view)

            return {
                'type': 'reference',
                'image': image_path,
                'subject': subject,
                'view': view,
                'text': line
            }

        # Pattern: "show the X first" or "show X first"
        match = re.match(r'show (?:the\s+)?([\w\s]+) first', line, re.IGNORECASE)
        if match:
            preference = match.group(1).strip()
            self.rules.append({
                'type': 'preference',
                'preference': preference,
                'text': line
            })
            return {
                'type': 'rule',
                'rule_type': 'preference',
                'preference': preference,
                'text': line
            }

        # Pattern: "when displaying an image, X"
        # Example: "when displaying an image, maximize the use of the space but do not crop"
        match = re.match(r'when (?:displaying|rendering|showing) (?:an\s+)?image,?\s+(.+)', line, re.IGNORECASE)
        if match:
            instruction = match.group(1).strip()
            self.rules.append({
                'type': 'rendering',
                'instruction': instruction,
                'text': line
            })
            return {
                'type': 'rule',
                'rule_type': 'rendering',
                'instruction': instruction,
                'text': line
            }

        # Pattern: "X is Y when Z"
        # Example: "an image is cropped if you part of the image is cut off"
        match = re.match(r'(?:an?\s+)?([\w\s]+) is ([\w\s]+) (?:if|when) (.+)', line, re.IGNORECASE)
        if match:
            entity = match.group(1).strip()
            state = match.group(2).strip()
            condition = match.group(3).strip()
            self.rules.append({
                'type': 'definition',
                'entity': entity,
                'state': state,
                'condition': condition,
                'text': line
            })
            return {
                'type': 'rule',
                'rule_type': 'definition',
                'entity': entity,
                'state': state,
                'condition': condition,
                'text': line
            }

        # Pattern: "when rendering, ..."
        match = re.match(r'when rendering,?\s+(.+)', line, re.IGNORECASE)
        if match:
            instruction = match.group(1).strip()
            self.rules.append({
                'type': 'rendering_strategy',
                'instruction': instruction,
                'text': line
            })
            return {
                'type': 'rule',
                'rule_type': 'rendering_strategy',
                'instruction': instruction,
                'text': line
            }

        # Didn't match any pattern
        return {
            'type': 'unknown',
            'text': line
        }

    def _add_color_convention(self, structure: str, color_name: str):
        """Add a color convention to KG.

        Args:
            structure: Structure name
            color_name: Color name
        """
        # Map common color names to RGB
        color_map = {
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
            'grey': [128, 128, 128]
        }

        rgb = color_map.get(color_name.lower(), [128, 128, 128])

        self.kg.add_anatomical_convention(
            name=structure.lower().replace(' ', '_'),
            description=f"{structure.title()} structure",
            color=rgb,
            color_name=color_name,
            intensity_range=None,
            opacity_range=[0.8, 1.0],
            medical_rationale=f"User specified: {structure} should be displayed in {color_name}"
        )

    def _add_reference(self, image_path: str, subject: str, view: str):
        """Add a reference image to KG.

        Args:
            image_path: Path to image
            subject: What the image shows
            view: Viewing angle
        """
        category = subject.lower().replace(' ', '_')
        tags = [f"{view.replace(' ', '_')}_view"]

        # Store reference even if image doesn't exist yet
        # User may add images later
        ref_id = f"{category}_{view.replace(' ', '_')}_view"

        # Image path should be relative to the KG directory
        # The KG is at .kg/graph.json, so images at .kg/images/ should be stored as "images/"

        kg_base = Path(self.kg.kg_path)  # .kg/
        kg_images_dir = kg_base / "images"
        kg_images_dir.mkdir(parents=True, exist_ok=True)

        # Target path in KG
        target_filename = Path(image_path).name
        target_path = kg_images_dir / target_filename
        stored_path = f"images/{target_filename}"

        # Check where the source image exists
        source_locations = [
            Path(image_path),  # As specified: images/skull_front.jpg
            Path.cwd() / image_path,  # ./images/skull_front.jpg
            kg_base / image_path,  # .kg/images/skull_front.jpg (already there)
        ]

        exists = False
        source_path = None

        for loc in source_locations:
            if loc.exists() and loc.is_file():
                source_path = loc
                exists = True
                break

        # Copy/move image to KG images directory if found
        if exists and source_path:
            # If already in the right place, no need to copy
            if source_path.resolve() != target_path.resolve():
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"[Import] Copied image: {source_path} → {target_path}")
            else:
                print(f"[Import] Image already in KG: {target_path}")
        else:
            print(f"[Import] ⚠️  Image not found: {image_path} (will mark as missing)")
            exists = False

        self.kg.graph["reference_renders"][ref_id] = {
            "id": ref_id,
            "image_path": stored_path,
            "category": category,
            "quality": "good",
            "tags": tags,
            "description": f"Reference image for {subject} from {view} view",
            "render_params": None,
            "dataset_info": None,
            "added": datetime.now().isoformat(),
            "image_exists": exists
        }

        # Create relationships: Link reference to relevant conventions
        # Semantic mappings for anatomical structures
        anatomy_mappings = {
            'skull': ['bone', 'bones', 'teeth', 'tooth'],
            'skeleton': ['bone', 'bones'],
            'head': ['bone', 'bones', 'skull'],
            'face': ['bone', 'bones', 'skull'],
            'vascular': ['artery', 'arteries', 'vein', 'veins', 'vessel'],
            'heart': ['muscle', 'tissue', 'blood', 'vessel'],
        }

        # Check if any conventions match the subject/category
        subject_lower = subject.lower()
        category_lower = category.lower()

        for conv_name in self.kg.graph.get("conventions", {}).keys():
            conv_lower = conv_name.lower()

            # Direct match
            should_link = False

            if (conv_lower in subject_lower or subject_lower in conv_lower or
                conv_lower in category_lower or category_lower in conv_lower):
                should_link = True

            # Semantic match
            for anatomy_term, related_structures in anatomy_mappings.items():
                if anatomy_term in subject_lower or anatomy_term in category_lower:
                    if any(struct in conv_lower for struct in related_structures):
                        should_link = True
                        break

            if should_link:
                # Add relationship
                self.kg.graph["relationships"].append({
                    "source": ref_id,
                    "relationship": "depicts",
                    "target": conv_name,
                    "properties": {"inferred": True, "reason": f"Reference of {subject} linked to {conv_name}"},
                    "added": datetime.now().isoformat()
                })

        self.kg.graph["last_modified"] = datetime.now().isoformat()
        self.kg._save_graph()


def parse_natural_language_kg(markdown_path: str, kg_path: str = ".kg", verbose: bool = True) -> Dict[str, Any]:
    """Parse natural language markdown and populate KG.

    Args:
        markdown_path: Path to markdown file
        kg_path: Path to KG storage
        verbose: Print detailed parsing info

    Returns:
        Statistics dictionary
    """
    from .multimodal_kg import MultimodalKnowledgeGraph

    kg = MultimodalKnowledgeGraph(kg_path=kg_path)
    parser = KGNaturalParser(kg)

    if verbose:
        print(f"Parsing natural language: {markdown_path}\n")

    stats = parser.parse_file(markdown_path)

    if verbose:
        print("="*70)
        print("Parsed Items")
        print("="*70)

        for item in stats['parsed_items']:
            if item['type'] == 'convention':
                print(f"✓ Convention: {item['structure']} → {item['color']}")
                print(f"  '{item['text']}'")
            elif item['type'] == 'reference':
                print(f"✓ Reference: {item['subject']} ({item['view']} view)")
                print(f"  Image: {item['image']}")
                print(f"  '{item['text']}'")
            elif item['type'] == 'rule':
                print(f"✓ Rule ({item['rule_type']})")
                print(f"  '{item['text']}'")
            elif item['type'] == 'unknown':
                print(f"? Unknown: '{item['text']}'")
            print()

        print("="*70)
        print("Summary")
        print("="*70)
        print(f"✓ Conventions added: {stats['conventions']}")
        print(f"✓ References added: {stats['references']}")
        print(f"ℹ Rules parsed: {stats['rules']} (stored as metadata)")
        print()

    # Show KG contents
    if verbose:
        kg.print_summary()

    return stats
