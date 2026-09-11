"""Knowledge Graph tool with logging integration.

This module provides a wrapper around MultimodalKnowledgeGraph that:
- Logs all queries and operations
- Integrates with RunManager
- Provides tool interface for CLI
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json

from .multimodal_kg import MultimodalKnowledgeGraph
from .run_manager import RunManager


class KGTool:
    """Knowledge Graph tool with logging."""

    def __init__(self, kg_path: str = ".kg", run_manager: Optional[RunManager] = None):
        """Initialize KG tool.

        Args:
            kg_path: Path to knowledge graph storage
            run_manager: Optional run manager for logging
        """
        self.kg = MultimodalKnowledgeGraph(kg_path=kg_path)
        self.run_manager = run_manager

    def _log(self, operation: str, details: Dict[str, Any], result: Any = None):
        """Log KG operation.

        Args:
            operation: Operation name
            details: Operation details
            result: Operation result (optional)
        """
        if self.run_manager:
            log_details = details.copy()
            if result is not None:
                # Add result summary to log
                if isinstance(result, dict):
                    log_details['result_keys'] = list(result.keys())
                elif isinstance(result, list):
                    log_details['result_count'] = len(result)
                else:
                    log_details['result_type'] = type(result).__name__

            self.run_manager.log_operation(f"kg_{operation}", log_details)

            # Also log to main logger
            self.run_manager.logger.debug(f"KG operation: {operation}")
            self.run_manager.logger.debug(f"  Details: {json.dumps(details, indent=2)}")
            if result is not None:
                if isinstance(result, (dict, list)) and len(str(result)) < 500:
                    self.run_manager.logger.debug(f"  Result: {json.dumps(result, indent=2, default=str)}")
                else:
                    self.run_manager.logger.debug(f"  Result: {type(result).__name__}")

    # ==================== CONVENTIONS ====================

    def add_anatomical_convention(
        self,
        name: str,
        description: str,
        color: List[int],
        color_name: str = None,
        intensity_range: Optional[List[float]] = None,
        opacity_range: Optional[List[float]] = None,
        medical_rationale: str = None,
        references: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Add anatomical rendering convention.

        Args:
            name: Structure name (e.g., "bone", "artery")
            description: What this structure is
            color: RGB color [r, g, b] (0-255)
            color_name: Common color name
            intensity_range: Typical intensity [min, max]
            opacity_range: Recommended opacity [min, max]
            medical_rationale: Why this convention
            references: Reference image IDs

        Returns:
            Convention dictionary
        """
        details = {
            "name": name,
            "color": color,
            "color_name": color_name,
            "intensity_range": intensity_range,
            "medical_rationale": medical_rationale
        }

        result = self.kg.add_anatomical_convention(
            name=name,
            description=description,
            color=color,
            color_name=color_name,
            intensity_range=intensity_range,
            opacity_range=opacity_range,
            medical_rationale=medical_rationale,
            references=references
        )

        self._log("add_convention", details, result)
        return result

    def query_convention(self, structure_name: str) -> Optional[Dict[str, Any]]:
        """Query rendering convention for a structure.

        Args:
            structure_name: Name of the structure

        Returns:
            Convention dictionary or None
        """
        details = {"structure_name": structure_name}
        result = self.kg.query_convention(structure_name)

        self._log("query_convention", details, result)

        if self.run_manager and result:
            self.run_manager.logger.info(f"KG Convention for '{structure_name}':")
            self.run_manager.logger.info(f"  Color: {result['color']['name']} {result['color']['rgb']}")
            if result.get('intensity_range'):
                self.run_manager.logger.info(f"  Intensity range: {result['intensity_range']}")
            if result.get('medical_rationale'):
                self.run_manager.logger.info(f"  Rationale: {result['medical_rationale'][:80]}...")

        return result

    def add_colormap(
        self,
        name: str,
        colormap_type: str,
        colors: List[List[int]],
        description: str,
        use_cases: List[str],
        intensity_mapping: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Add colormap convention.

        Args:
            name: Colormap name
            colormap_type: "sequential", "diverging", or "categorical"
            colors: List of RGB colors
            description: What this represents
            use_cases: When to use
            intensity_mapping: Intensity values for each color

        Returns:
            Colormap dictionary
        """
        details = {
            "name": name,
            "type": colormap_type,
            "num_colors": len(colors)
        }

        result = self.kg.add_colormap_convention(
            name=name,
            colormap_type=colormap_type,
            colors=colors,
            description=description,
            use_cases=use_cases,
            intensity_mapping=intensity_mapping
        )

        self._log("add_colormap", details, result)
        return result

    def query_colormap(self, colormap_name: str) -> Optional[Dict[str, Any]]:
        """Query colormap convention.

        Args:
            colormap_name: Name of the colormap

        Returns:
            Colormap dictionary or None
        """
        details = {"colormap_name": colormap_name}
        result = self.kg.query_colormap(colormap_name)

        self._log("query_colormap", details, result)

        if self.run_manager and result:
            self.run_manager.logger.info(f"KG Colormap '{colormap_name}':")
            self.run_manager.logger.info(f"  Type: {result['type']}")
            self.run_manager.logger.info(f"  Colors: {len(result['colors'])} colors")
            self.run_manager.logger.info(f"  Use cases: {', '.join(result['use_cases'][:3])}")

        return result

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
        """Add reference render to KG.

        Args:
            image_path: Path to image
            category: Category (e.g., "bone", "skull")
            quality: "good" or "bad"
            tags: Descriptive tags
            description: Human description
            render_params: Rendering parameters used
            dataset_info: Source dataset info

        Returns:
            Reference ID
        """
        details = {
            "image_path": image_path,
            "category": category,
            "quality": quality,
            "tags": tags
        }

        ref_id = self.kg.add_reference_render(
            image_path=image_path,
            category=category,
            quality=quality,
            tags=tags,
            description=description,
            render_params=render_params,
            dataset_info=dataset_info
        )

        details["ref_id"] = ref_id
        self._log("add_reference", details, ref_id)

        if self.run_manager:
            self.run_manager.logger.info(f"Added reference render: {ref_id}")
            self.run_manager.logger.info(f"  Category: {category}, Quality: {quality}")
            self.run_manager.logger.info(f"  Tags: {', '.join(tags)}")

        return ref_id

    def get_reference_renders(
        self,
        category: Optional[str] = None,
        quality: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get reference renders matching criteria.

        Args:
            category: Filter by category
            quality: Filter by quality
            tags: Filter by tags (must have all)

        Returns:
            List of matching references
        """
        details = {
            "category": category,
            "quality": quality,
            "tags": tags
        }

        results = self.kg.get_reference_renders(
            category=category,
            quality=quality,
            tags=tags
        )

        self._log("get_references", details, results)

        if self.run_manager:
            self.run_manager.logger.info(f"Found {len(results)} reference renders")
            if results:
                self.run_manager.logger.info(f"  Categories: {set(r['category'] for r in results)}")
                self.run_manager.logger.info(f"  Quality: {set(r['quality'] for r in results)}")

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
    ) -> Dict[str, Any]:
        """Add dataset knowledge.

        Args:
            dataset_name: Unique dataset name
            dataset_path: Path to dataset
            dimensions: [x, y, z]
            dtype: Data type
            description: What it contains
            anatomy: Anatomical region
            modality: Imaging modality
            typical_intensity_ranges: Structure → [min, max]
            known_good_params: Successful parameters
            notes: Additional notes

        Returns:
            Dataset info dictionary
        """
        details = {
            "dataset_name": dataset_name,
            "dimensions": dimensions,
            "anatomy": anatomy,
            "modality": modality
        }

        result = self.kg.add_dataset_knowledge(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dimensions=dimensions,
            dtype=dtype,
            description=description,
            anatomy=anatomy,
            modality=modality,
            typical_intensity_ranges=typical_intensity_ranges,
            known_good_params=known_good_params,
            notes=notes
        )

        self._log("add_dataset", details, result)

        if self.run_manager:
            self.run_manager.logger.info(f"Added dataset knowledge: {dataset_name}")
            self.run_manager.logger.info(f"  Anatomy: {anatomy}, Modality: {modality}")
            self.run_manager.logger.info(f"  Structures: {list(typical_intensity_ranges.keys())}")

        return result

    def get_dataset_knowledge(
        self,
        dataset_name: str = None,
        dataset_path: str = None
    ) -> Optional[Dict[str, Any]]:
        """Get dataset knowledge.

        Args:
            dataset_name: Lookup by name
            dataset_path: Lookup by path

        Returns:
            Dataset info or None
        """
        details = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path
        }

        result = self.kg.get_dataset_knowledge(
            dataset_name=dataset_name,
            dataset_path=dataset_path
        )

        self._log("get_dataset", details, result)

        if self.run_manager and result:
            self.run_manager.logger.info(f"Dataset: {result['name']}")
            self.run_manager.logger.info(f"  Dimensions: {result['dimensions']}")
            self.run_manager.logger.info(f"  Anatomy: {result['anatomy']}, Modality: {result['modality']}")
            self.run_manager.logger.info(f"  Intensity ranges: {list(result['intensity_ranges'].keys())}")

        return result

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
        """Store successful render for learning.

        Args:
            dataset_name: Which dataset
            prompt: User prompt
            final_params: Final parameters
            iterations: Iteration count
            result_image_path: Path to result
            vision_feedback: Vision AI feedback
            quality_score: Optional quality (0-1)

        Returns:
            Render ID
        """
        details = {
            "dataset_name": dataset_name,
            "prompt": prompt,
            "iterations": iterations,
            "quality_score": quality_score
        }

        render_id = self.kg.store_successful_render(
            dataset_name=dataset_name,
            prompt=prompt,
            final_params=final_params,
            iterations=iterations,
            result_image_path=result_image_path,
            vision_feedback=vision_feedback,
            quality_score=quality_score
        )

        details["render_id"] = render_id
        self._log("store_render", details, render_id)

        if self.run_manager:
            self.run_manager.logger.info(f"Stored successful render: {render_id}")
            self.run_manager.logger.info(f"  Dataset: {dataset_name}")
            self.run_manager.logger.info(f"  Iterations: {iterations}, Quality: {quality_score}")

        return render_id

    def get_learned_params(
        self,
        dataset_name: str,
        prompt_similarity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learned parameters from previous renders.

        Args:
            dataset_name: Which dataset
            prompt_similarity: Optional prompt matching

        Returns:
            List of learned renders
        """
        details = {
            "dataset_name": dataset_name,
            "prompt_similarity": prompt_similarity
        }

        results = self.kg.get_learned_params(
            dataset_name=dataset_name,
            prompt_similarity=prompt_similarity
        )

        self._log("get_learned_params", details, results)

        if self.run_manager:
            self.run_manager.logger.info(f"Found {len(results)} learned renders for {dataset_name}")
            if results:
                best = results[0]  # Most recent
                self.run_manager.logger.info(f"  Best render: {best['id']}")
                self.run_manager.logger.info(f"    Iterations: {best['iterations']}")
                if best.get('quality_score'):
                    self.run_manager.logger.info(f"    Quality: {best['quality_score']:.2f}")

        return results

    # ==================== RECOMMENDATIONS ====================

    def get_recommendations(
        self,
        dataset_name: str,
        structures: List[str]
    ) -> Dict[str, Any]:
        """Get rendering recommendations.

        Args:
            dataset_name: Which dataset
            structures: Which structures to render

        Returns:
            Recommendations dictionary
        """
        details = {
            "dataset_name": dataset_name,
            "structures": structures
        }

        recommendations = self.kg.get_recommendations(
            dataset_name=dataset_name,
            structures=structures
        )

        self._log("get_recommendations", details, recommendations)

        if self.run_manager:
            self.run_manager.logger.info(f"Recommendations for {dataset_name}, structures: {structures}")

            if recommendations['dataset']:
                ds = recommendations['dataset']
                self.run_manager.logger.info(f"  Dataset: {ds['anatomy']}, {ds['modality']}")
                self.run_manager.logger.info(f"  Dimensions: {ds['dimensions']}")

            self.run_manager.logger.info(f"  Conventions: {len(recommendations['structures'])} structures")
            for struct, conv in recommendations['structures'].items():
                self.run_manager.logger.info(f"    {struct}: {conv['color']['name']} {conv['color']['rgb']}")

            self.run_manager.logger.info(f"  Reference images: {len(recommendations['reference_images'])}")
            self.run_manager.logger.info(f"  Learned renders: {len(recommendations['learned_params'])}")

        return recommendations

    # ==================== UTILITY ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get KG statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.kg.get_statistics()
        self._log("get_statistics", {}, stats)
        return stats

    def export_to_json(self, output_path: str):
        """Export KG to JSON.

        Args:
            output_path: Where to save
        """
        self.kg.export_to_json(output_path)
        self._log("export", {"output_path": output_path})

        if self.run_manager:
            self.run_manager.logger.info(f"Exported KG to: {output_path}")

    def print_summary(self):
        """Print KG summary."""
        self.kg.print_summary()
        if self.run_manager:
            stats = self.kg.get_statistics()
            self.run_manager.logger.info(f"KG Summary: {stats}")
