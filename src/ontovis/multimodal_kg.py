"""Multimodal Knowledge Graph for volume rendering domain knowledge.

This KG stores:
- Reference images (good/bad renders)
- Rendering conventions (colors, thresholds, etc.)
- Dataset-specific parameters
- Learned rendering patterns
- Medical/scientific domain knowledge
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import shutil

import numpy as np
from PIL import Image


class MultimodalKnowledgeGraph:
    """Knowledge graph for storing multimodal rendering knowledge."""

    def __init__(self, kg_path: str = ".kg"):
        """Initialize the knowledge graph.

        Args:
            kg_path: Directory to store KG data (default: .kg)
        """
        self.kg_path = Path(kg_path)
        self.kg_path.mkdir(parents=True, exist_ok=True)

        # Initialize subdirectories
        self.images_path = self.kg_path / "images"
        self.images_path.mkdir(exist_ok=True)

        self.datasets_path = self.kg_path / "datasets"
        self.datasets_path.mkdir(exist_ok=True)

        self.conventions_path = self.kg_path / "conventions"
        self.conventions_path.mkdir(exist_ok=True)

        self.documentation_path = self.kg_path / "documentation"
        self.documentation_path.mkdir(exist_ok=True)

        # Load or initialize the graph structure
        self.graph_file = self.kg_path / "graph.json"
        self.graph = self._load_graph()

    def _load_graph(self) -> Dict[str, Any]:
        """Load the graph from disk or create a new one."""
        if self.graph_file.exists():
            with open(self.graph_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "entities": {},
                "relationships": [],
                "conventions": {},
                "reference_renders": {},
                "dataset_knowledge": {},
                "learned_params": {}
            }

    def _save_graph(self):
        """Save the graph to disk."""
        self.graph["last_modified"] = datetime.now().isoformat()
        with open(self.graph_file, 'w') as f:
            json.dump(self.graph, f, indent=2)

    # ==================== RENDERING CONVENTIONS ====================

    def add_anatomical_convention(
        self,
        name: str,
        description: str,
        color: Union[List[int], str],
        color_name: str = None,
        intensity_range: Optional[List[float]] = None,
        opacity_range: Optional[List[float]] = None,
        medical_rationale: str = None,
        references: Optional[List[str]] = None
    ):
        """Add a rendering convention for anatomical structures.

        Args:
            name: Anatomical structure name (e.g., "bone", "artery", "vein")
            description: What this structure is
            color: RGB color [r, g, b] (0-255) or hex "#RRGGBB"
            color_name: Common name for the color (e.g., "white", "red", "blue")
            intensity_range: Typical intensity range [min, max] in volume data
            opacity_range: Recommended opacity range [min, max] (0.0-1.0)
            medical_rationale: Why this color/rendering (e.g., "doctors trained on white bones")
            references: List of reference image IDs or URLs
        """
        # Normalize color to RGB list
        if isinstance(color, str):
            # Convert hex to RGB
            color = color.lstrip('#')
            color = [int(color[i:i+2], 16) for i in (0, 2, 4)]

        convention = {
            "name": name,
            "description": description,
            "color": {
                "rgb": color,
                "name": color_name or "custom",
                "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            },
            "intensity_range": intensity_range,
            "opacity_range": opacity_range or [0.5, 1.0],
            "medical_rationale": medical_rationale,
            "references": references or [],
            "added": datetime.now().isoformat()
        }

        self.graph["conventions"][name] = convention
        self._save_graph()

        print(f"✓ Added convention: {name} → {color_name or 'RGB' + str(color)}")
        return convention

    def add_colormap_convention(
        self,
        name: str,
        colormap_type: str,
        colors: List[Union[List[int], str]],
        description: str,
        use_cases: List[str],
        intensity_mapping: Optional[List[float]] = None
    ):
        """Add a colormap convention.

        Args:
            name: Colormap name (e.g., "heat_map", "bone_window", "tissue_differentiation")
            colormap_type: Type ("sequential", "diverging", "categorical")
            colors: List of RGB colors or hex strings
            description: What this colormap represents
            use_cases: When to use this colormap
            intensity_mapping: Intensity values corresponding to each color
        """
        # Normalize all colors
        normalized_colors = []
        for c in colors:
            if isinstance(c, str):
                c = c.lstrip('#')
                c = [int(c[i:i+2], 16) for i in (0, 2, 4)]
            normalized_colors.append(c)

        colormap = {
            "name": name,
            "type": colormap_type,
            "colors": normalized_colors,
            "color_stops": intensity_mapping or list(np.linspace(0, 1, len(normalized_colors))),
            "description": description,
            "use_cases": use_cases,
            "added": datetime.now().isoformat()
        }

        if "colormaps" not in self.graph["conventions"]:
            self.graph["conventions"]["colormaps"] = {}

        self.graph["conventions"]["colormaps"][name] = colormap
        self._save_graph()

        print(f"✓ Added colormap: {name} ({len(normalized_colors)} colors)")
        return colormap

    # ==================== REFERENCE IMAGES ====================

    def add_reference_render(
        self,
        image_path: str,
        category: str,
        quality: str,
        tags: List[str],
        description: str,
        render_params: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a reference render image to the KG.

        Args:
            image_path: Path to the image file
            category: Category (e.g., "bone", "full_body", "skull", "vascular")
            quality: "good" or "bad" (for learning what to aim for/avoid)
            tags: List of descriptive tags (e.g., ["well_framed", "high_contrast"])
            description: Human description of the render
            render_params: Rendering parameters used (if known)
            dataset_info: Information about the source dataset

        Returns:
            str: Unique ID for this reference
        """
        # Generate unique ID
        ref_id = f"{category}_{quality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Copy image to KG storage
        src_path = Path(image_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Store in category subdirectory
        category_dir = self.images_path / category / quality
        category_dir.mkdir(parents=True, exist_ok=True)

        dest_path = category_dir / f"{ref_id}{src_path.suffix}"
        shutil.copy(src_path, dest_path)

        # Get image dimensions
        img = Image.open(dest_path)
        dimensions = img.size

        # Create reference entry
        reference = {
            "id": ref_id,
            "category": category,
            "quality": quality,
            "tags": tags,
            "description": description,
            "image_path": str(dest_path.relative_to(self.kg_path)),
            "image_dimensions": dimensions,
            "render_params": render_params,
            "dataset_info": dataset_info,
            "added": datetime.now().isoformat()
        }

        # Store in graph
        if category not in self.graph["reference_renders"]:
            self.graph["reference_renders"][category] = []

        self.graph["reference_renders"][category].append(reference)
        self._save_graph()

        print(f"✓ Added reference: {ref_id} ({quality} {category})")
        return ref_id

    def get_reference_renders(
        self,
        category: Optional[str] = None,
        quality: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve reference renders matching criteria.

        Args:
            category: Filter by category
            quality: Filter by quality ("good" or "bad")
            tags: Filter by tags (must have ALL tags)

        Returns:
            List of matching reference entries
        """
        results = []

        # Iterate through all reference renders (stored as {ref_id: ref_dict})
        for ref_id, ref in self.graph["reference_renders"].items():
            # Filter by category (with fuzzy matching)
            if category:
                ref_category = ref.get("category", "").lower()
                search_lower = category.lower()

                # Exact match
                match = ref_category == search_lower

                # Fuzzy match: singular/plural
                if not match:
                    match = (
                        search_lower + 's' == ref_category or
                        search_lower == ref_category + 's' or
                        search_lower in ref_category or
                        ref_category in search_lower
                    )

                if not match:
                    continue

            # Filter by quality
            if quality and ref.get("quality") != quality:
                continue

            # Filter by tags
            if tags and not all(tag in ref.get("tags", []) for tag in tags):
                continue

            # Add to results
            ref_copy = ref.copy()
            # Resolve path relative to KG directory
            if ref.get("image_exists"):
                # Path in KG is relative to .kg/ directory
                full_path = self.kg_path / ref["image_path"]
                ref_copy["image_full_path"] = str(full_path)
            results.append(ref_copy)

        return results

    # ==================== DATASET KNOWLEDGE ====================

    def add_dataset_knowledge(
        self,
        dataset_name: str,
        dataset_path: str,
        dimensions: List[int],
        dtype: str,
        description: str,
        anatomy: str,
        modality: str,
        typical_intensity_ranges: Dict[str, List[float]],
        known_good_params: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ):
        """Add knowledge about a specific dataset.

        Args:
            dataset_name: Unique name for the dataset
            dataset_path: Path to the dataset file
            dimensions: [x, y, z] dimensions
            dtype: Data type (e.g., "uint8", "uint16")
            description: What this dataset contains
            anatomy: Anatomical region (e.g., "full_body", "head", "torso")
            modality: Imaging modality (e.g., "CT", "MRI", "micro-CT")
            typical_intensity_ranges: Dict of structure → [min, max] intensity
                                      e.g., {"bone": [180, 254], "soft_tissue": [40, 120]}
            known_good_params: Previously successful rendering parameters
            notes: Additional notes about the dataset
        """
        dataset_info = {
            "name": dataset_name,
            "path": dataset_path,
            "dimensions": dimensions,
            "dtype": dtype,
            "description": description,
            "anatomy": anatomy,
            "modality": modality,
            "intensity_ranges": typical_intensity_ranges,
            "known_good_params": known_good_params or {},
            "notes": notes,
            "added": datetime.now().isoformat(),
            "renders": []  # Will store successful renders
        }

        self.graph["dataset_knowledge"][dataset_name] = dataset_info
        self._save_graph()

        print(f"✓ Added dataset: {dataset_name} ({anatomy}, {modality})")
        return dataset_info

    def get_dataset_knowledge(self, dataset_name: str = None, dataset_path: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge about a dataset.

        Args:
            dataset_name: Look up by name
            dataset_path: Look up by path

        Returns:
            Dataset knowledge or None if not found
        """
        if dataset_name and dataset_name in self.graph["dataset_knowledge"]:
            return self.graph["dataset_knowledge"][dataset_name]

        if dataset_path:
            for name, info in self.graph["dataset_knowledge"].items():
                if info["path"] == dataset_path:
                    return info

        return None

    # ==================== LEARNED PARAMETERS ====================

    def store_successful_render(
        self,
        dataset_name: str,
        prompt: str,
        final_params: Dict[str, Any],
        iterations: int,
        result_image_path: str,
        vision_feedback: str,
        quality_score: Optional[float] = None
    ) -> str:
        """Store parameters from a successful render for future learning.

        Args:
            dataset_name: Which dataset was rendered
            prompt: User prompt used
            final_params: Final rendering parameters that worked
            iterations: How many iterations it took
            result_image_path: Path to the result image
            vision_feedback: Final vision AI feedback
            quality_score: Optional quality score (0-1)

        Returns:
            str: ID of the stored render
        """
        render_id = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Copy result image to learned renders directory
        learned_dir = self.images_path / "learned_renders"
        learned_dir.mkdir(exist_ok=True)

        src_path = Path(result_image_path)
        if src_path.exists():
            dest_path = learned_dir / f"{render_id}{src_path.suffix}"
            shutil.copy(src_path, dest_path)
            stored_image_path = str(dest_path.relative_to(self.kg_path))
        else:
            stored_image_path = None

        learned_render = {
            "id": render_id,
            "dataset_name": dataset_name,
            "prompt": prompt,
            "final_params": final_params,
            "iterations": iterations,
            "result_image_path": stored_image_path,
            "vision_feedback": vision_feedback,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat()
        }

        # Store in learned params
        if dataset_name not in self.graph["learned_params"]:
            self.graph["learned_params"][dataset_name] = []

        self.graph["learned_params"][dataset_name].append(learned_render)

        # Also add to dataset knowledge if exists
        if dataset_name in self.graph["dataset_knowledge"]:
            self.graph["dataset_knowledge"][dataset_name]["renders"].append(render_id)

            # Update known_good_params if this is better
            current_best = self.graph["dataset_knowledge"][dataset_name].get("known_good_params", {})
            if not current_best or (quality_score and quality_score > current_best.get("quality_score", 0)):
                self.graph["dataset_knowledge"][dataset_name]["known_good_params"] = final_params
                self.graph["dataset_knowledge"][dataset_name]["best_render_id"] = render_id

        self._save_graph()

        print(f"✓ Stored successful render: {render_id} ({iterations} iterations)")
        return render_id

    def get_learned_params(self, dataset_name: str, prompt_similarity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get learned parameters from previous successful renders.

        Args:
            dataset_name: Which dataset to get params for
            prompt_similarity: Optional prompt to find similar renders

        Returns:
            List of learned renders (most recent first)
        """
        if dataset_name not in self.graph["learned_params"]:
            return []

        renders = self.graph["learned_params"][dataset_name]

        # Sort by timestamp (most recent first)
        renders = sorted(renders, key=lambda x: x["timestamp"], reverse=True)

        # TODO: Could implement prompt similarity matching here
        # For now, return all renders for this dataset

        return renders

    # ==================== RELATIONSHIPS & QUERIES ====================

    def add_relationship(
        self,
        source_id: str,
        relationship: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Add a relationship between entities in the KG.

        Args:
            source_id: Source entity ID
            relationship: Relationship type (e.g., "similar_to", "better_than", "used_for")
            target_id: Target entity ID
            properties: Optional properties of the relationship
        """
        rel = {
            "source": source_id,
            "relationship": relationship,
            "target": target_id,
            "properties": properties or {},
            "added": datetime.now().isoformat()
        }

        self.graph["relationships"].append(rel)
        self._save_graph()

    def query_convention(self, structure_name: str) -> Optional[Dict[str, Any]]:
        """Query rendering convention for an anatomical structure.

        Args:
            structure_name: Name of the structure (e.g., "bone", "artery")

        Returns:
            Convention information or None
        """
        # Try exact match first
        if structure_name in self.graph["conventions"]:
            return self.graph["conventions"][structure_name]

        # Try fuzzy matching - singular/plural, case-insensitive
        search_lower = structure_name.lower()

        for key, value in self.graph["conventions"].items():
            key_lower = key.lower()

            # Exact match (case-insensitive)
            if key_lower == search_lower:
                return value

            # Singular <-> Plural matching
            # Try adding 's'
            if search_lower + 's' == key_lower or search_lower == key_lower + 's':
                return value

            # Try 'es' ending (box -> boxes)
            if search_lower + 'es' == key_lower or search_lower == key_lower + 'es':
                return value

            # Try removing 's' or 'es'
            if search_lower.endswith('s'):
                if search_lower[:-1] == key_lower:
                    return value
            if search_lower.endswith('es'):
                if search_lower[:-2] == key_lower:
                    return value

            # Substring match (e.g., "bone" matches "trabecular_bone")
            if search_lower in key_lower or key_lower in search_lower:
                return value

        return None

    def query_colormap(self, colormap_name: str) -> Optional[Dict[str, Any]]:
        """Query colormap convention.

        Args:
            colormap_name: Name of the colormap

        Returns:
            Colormap information or None
        """
        if "colormaps" in self.graph["conventions"]:
            return self.graph["conventions"]["colormaps"].get(colormap_name)
        return None

    def get_recommendations(
        self,
        dataset_name: str,
        structures: List[str]
    ) -> Dict[str, Any]:
        """Get rendering recommendations based on KG knowledge.

        Args:
            dataset_name: Which dataset is being rendered
            structures: Which anatomical structures to show

        Returns:
            Recommended parameters and conventions
        """
        recommendations = {
            "dataset": None,
            "structures": {},
            "reference_images": [],
            "learned_params": []
        }

        # Get dataset knowledge
        recommendations["dataset"] = self.get_dataset_knowledge(dataset_name)

        # Get conventions for each structure
        for structure in structures:
            convention = self.query_convention(structure)
            if convention:
                recommendations["structures"][structure] = convention

        # Get relevant reference images
        for structure in structures:
            refs = self.get_reference_renders(category=structure, quality="good")
            recommendations["reference_images"].extend(refs)

        # Get learned params
        recommendations["learned_params"] = self.get_learned_params(dataset_name)

        return recommendations

    # ==================== DOCUMENTATION ====================

    def get_documentation(self, doc_name: str = "rendering_strategies") -> Optional[str]:
        """Load documentation markdown file from KG.

        Args:
            doc_name: Name of the documentation file (without .md extension)

        Returns:
            Content of the documentation file, or None if not found
        """
        doc_path = self.documentation_path / f"{doc_name}.md"
        if doc_path.exists():
            return doc_path.read_text()
        return None

    def list_documentation(self) -> List[str]:
        """List all available documentation files.

        Returns:
            List of documentation file names (without .md extension)
        """
        if not self.documentation_path.exists():
            return []

        return [f.stem for f in self.documentation_path.glob("*.md")]

    def add_documentation(self, doc_name: str, content: str):
        """Add or update documentation file in KG.

        Args:
            doc_name: Name of the documentation file (without .md extension)
            content: Markdown content
        """
        doc_path = self.documentation_path / f"{doc_name}.md"
        doc_path.write_text(content)
        print(f"✓ Documentation saved: {doc_name}.md")

    # ==================== EXPORT & VISUALIZATION ====================

    def export_to_json(self, output_path: str):
        """Export the entire KG to a JSON file.

        Args:
            output_path: Where to save the JSON
        """
        with open(output_path, 'w') as f:
            json.dump(self.graph, f, indent=2)
        print(f"✓ Exported KG to: {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the KG contents."""
        stats = {
            "total_conventions": len(self.graph["conventions"]),
            "total_colormaps": len(self.graph["conventions"].get("colormaps", {})),
            "total_reference_renders": sum(
                len(refs) for refs in self.graph["reference_renders"].values()
            ),
            "reference_by_category": {
                cat: len(refs) for cat, refs in self.graph["reference_renders"].items()
            },
            "total_datasets": len(self.graph["dataset_knowledge"]),
            "total_learned_renders": sum(
                len(renders) for renders in self.graph["learned_params"].values()
            ),
            "total_relationships": len(self.graph["relationships"])
        }
        return stats

    def print_summary(self):
        """Print a summary of the KG contents."""
        stats = self.get_statistics()

        print("\n" + "="*70)
        print("Knowledge Graph Summary")
        print("="*70)
        print(f"\n📚 Conventions: {stats['total_conventions']}")
        print(f"🎨 Colormaps: {stats['total_colormaps']}")
        print(f"\n🖼️  Reference Renders: {stats['total_reference_renders']}")
        for cat, count in stats['reference_by_category'].items():
            print(f"   - {cat}: {count}")
        print(f"\n💾 Datasets: {stats['total_datasets']}")
        print(f"🎯 Learned Renders: {stats['total_learned_renders']}")
        print(f"🔗 Relationships: {stats['total_relationships']}")
        print("\n" + "="*70 + "\n")
