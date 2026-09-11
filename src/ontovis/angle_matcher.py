"""Vision-guided angle matching for finding optimal camera positions."""

import io
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import numpy as np
from PIL import Image
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .config import Config
from .multimodal_kg import MultimodalKnowledgeGraph


class AngleMatcher:
    """Find optimal camera angles by comparing renders against reference images."""

    def __init__(self, kg_path: str = ".kg", config_path: Optional[str] = None):
        """Initialize the angle matcher.

        Args:
            kg_path: Path to knowledge graph directory
            config_path: Path to config file (optional)
        """
        self.kg = MultimodalKnowledgeGraph(kg_path)
        self.config = Config(config_path)

        # Initialize vision model
        llm_config = self.config.get_llm_config()
        self.vision_llm = ChatAnthropic(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            max_tokens=2048
        )

        # Load search parameters from KG documentation
        strategies = self.kg.get_documentation("rendering_strategies")
        self._load_search_params(strategies)

    def _load_search_params(self, strategies: Optional[str]):
        """Extract search parameters from KG documentation.

        Args:
            strategies: Content of rendering_strategies.md
        """
        # Default search parameters from KG strategy
        self.coarse_azimuth = [0, 45, 90, 135, 180, 225, 270, 315]
        self.coarse_elevation = [-90, -45, 0, 45, 90]
        self.coarse_roll = [-90, 0, 90]
        self.fine_step = 15
        self.fine_window = 45

        self.score_threshold_excellent = 9.5
        self.score_threshold_good = 8.5
        self.score_threshold_acceptable = 8.0
        self.score_threshold_poor = 7.0

    def compare_images(
        self,
        candidate_image: Image.Image,
        reference_image: Image.Image,
        context: str = ""
    ) -> float:
        """Compare two images using vision model.

        Args:
            candidate_image: Rendered candidate image
            reference_image: Reference image from KG
            context: Optional context about what to look for

        Returns:
            Similarity score from 0-10
        """
        # Convert images to base64
        def image_to_base64(img: Image.Image) -> str:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        candidate_b64 = image_to_base64(candidate_image)
        reference_b64 = image_to_base64(reference_image)

        # Construct vision comparison prompt (from KG strategy)
        prompt = f"""Compare these two medical volume renderings and rate their similarity from 0-10.

Reference image (target - first image)
Candidate image (rendered - second image)

Consider: viewing angle, anatomical orientation, visible structures, overall perspective.

{context}

Rate similarity 0-10:
- 10: Perfect match
- 8-9: Very similar (minor differences)
- 6-7: Somewhat similar (recognizable but different)
- 4-5: Different perspectives
- 0-3: Very different

Respond with ONLY a number from 0-10."""

        # Create message with images
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{reference_b64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{candidate_b64}"}
                }
            ]
        )

        # Get response
        response = self.vision_llm.invoke([message])
        score_text = response.content.strip()

        # Parse score
        try:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, score))
            else:
                print(f"Warning: Could not parse score from: {score_text}")
                return 0.0
        except Exception as e:
            print(f"Error parsing score: {e}")
            return 0.0

    def coarse_search(
        self,
        render_fn: Callable,
        reference_image: Image.Image,
        distance: float = 2.0
    ) -> Dict[str, Any]:
        """Perform coarse search (45° increments).

        Args:
            render_fn: Function(azimuth, elevation, roll, distance) -> Image
            reference_image: Reference image to match
            distance: Camera distance

        Returns:
            Dict with best_angles, best_score, all_results
        """
        print("\n" + "="*70)
        print("PHASE 1: COARSE SEARCH (45° increments)")
        print("="*70)

        results = []
        best_score = 0.0
        best_angles = None

        total = len(self.coarse_azimuth) * len(self.coarse_elevation) * len(self.coarse_roll)
        count = 0

        for azimuth in self.coarse_azimuth:
            for elevation in self.coarse_elevation:
                for roll in self.coarse_roll:
                    count += 1
                    print(f"\n[{count}/{total}] azimuth={azimuth}°, elevation={elevation}°, roll={roll}°", end="")

                    # Render
                    candidate_image = render_fn(azimuth, elevation, roll, distance)

                    # Compare
                    score = self.compare_images(candidate_image, reference_image)
                    print(f" → {score:.1f}/10", end="")

                    results.append({
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'roll': roll,
                        'distance': distance,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_angles = {'azimuth': azimuth, 'elevation': elevation, 'roll': roll, 'distance': distance}
                        print(" ✓ NEW BEST", end="")

                    # Early termination
                    if score >= self.score_threshold_excellent:
                        print(f"\n\n✓ Excellent match (≥{self.score_threshold_excellent})! Stopping early.")
                        return {
                            'best_angles': best_angles,
                            'best_score': best_score,
                            'all_results': results,
                            'early_termination': True
                        }

        print(f"\n\n{'='*70}")
        print(f"COARSE COMPLETE: Best={best_score:.1f}/10 at azimuth={best_angles['azimuth']}°, elevation={best_angles['elevation']}°, roll={best_angles['roll']}°")
        print("="*70)

        return {
            'best_angles': best_angles,
            'best_score': best_score,
            'all_results': results,
            'early_termination': False
        }

    def fine_search(
        self,
        render_fn: Callable,
        reference_image: Image.Image,
        coarse_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform fine search (15° increments around best).

        Args:
            render_fn: Function(azimuth, elevation, roll, distance) -> Image
            reference_image: Reference image to match
            coarse_result: Result from coarse_search

        Returns:
            Dict with best_angles, best_score, all_results
        """
        print("\n" + "="*70)
        print("PHASE 2: FINE SEARCH (15° increments)")
        print("="*70)

        best_coarse = coarse_result['best_angles']
        best_score = coarse_result['best_score']
        best_angles = best_coarse.copy()

        print(f"\nRefining around: azimuth={best_coarse['azimuth']}° ±{self.fine_window}°, elevation={best_coarse['elevation']}° ±{self.fine_window}°, roll={best_coarse['roll']}° ±{self.fine_window}°")

        # Generate ranges
        azimuth_range = range(
            best_coarse['azimuth'] - self.fine_window,
            best_coarse['azimuth'] + self.fine_window + 1,
            self.fine_step
        )
        elevation_range = range(
            max(-90, best_coarse['elevation'] - self.fine_window),
            min(90, best_coarse['elevation'] + self.fine_window) + 1,
            self.fine_step
        )
        roll_range = range(
            max(-180, best_coarse['roll'] - self.fine_window),
            min(180, best_coarse['roll'] + self.fine_window) + 1,
            self.fine_step
        )

        results = []
        no_improvement = 0
        distance = best_coarse['distance']

        for azimuth in azimuth_range:
            azimuth = azimuth % 360

            for elevation in elevation_range:
                for roll in roll_range:
                    # Skip coarse result already tested
                    if (azimuth == best_coarse['azimuth'] and
                        elevation == best_coarse['elevation'] and
                        roll == best_coarse['roll']):
                        continue

                    print(f"\nazimuth={azimuth}°, elevation={elevation}°, roll={roll}°", end="")

                    # Render
                    candidate_image = render_fn(azimuth, elevation, roll, distance)

                    # Compare
                    score = self.compare_images(candidate_image, reference_image)
                    print(f" → {score:.1f}/10", end="")

                    results.append({
                        'azimuth': azimuth,
                        'elevation': elevation,
                        'roll': roll,
                        'distance': distance,
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_angles = {'azimuth': azimuth, 'elevation': elevation, 'roll': roll, 'distance': distance}
                        print(" ✓ NEW BEST", end="")
                        no_improvement = 0
                    else:
                        no_improvement += 1

                    # Early termination
                    if score >= self.score_threshold_excellent:
                        print(f"\n\n✓ Excellent match (≥{self.score_threshold_excellent})!")
                        return {
                            'best_angles': best_angles,
                            'best_score': best_score,
                            'all_results': results,
                            'early_termination': True
                        }

                    if no_improvement >= 10:
                        print(f"\n\n⚠ No improvement after 10 tests. Stopping.")
                        return {
                            'best_angles': best_angles,
                            'best_score': best_score,
                            'all_results': results,
                            'early_termination': True,
                            'reason': 'diminishing_returns'
                        }

        print(f"\n\n{'='*70}")
        print(f"FINE COMPLETE: Best={best_score:.1f}/10 at azimuth={best_angles['azimuth']}°, elevation={best_angles['elevation']}°, roll={best_angles['roll']}°")
        print("="*70)

        return {
            'best_angles': best_angles,
            'best_score': best_score,
            'all_results': results,
            'early_termination': False
        }

    def find_best_angle(
        self,
        render_fn: Callable,
        reference_id: str,
        volume_path: str,
        distance: float = 2.0
    ) -> Dict[str, Any]:
        """Find best camera angle to match reference image.

        Args:
            render_fn: Function(azimuth, elevation, roll, distance) -> PIL.Image
            reference_id: ID of reference in KG (e.g., "a_skull_front_view")
            volume_path: Path to volume (for learned params lookup)
            distance: Camera distance

        Returns:
            Dict with final_angles, match_score, iterations, search_strategy
        """
        volume_name = Path(volume_path).stem

        # Check learned parameters
        learned = self.kg.get_learned_params(volume_name)
        if learned:
            for entry in learned:
                if entry.get('reference_id') == reference_id:
                    print("\n" + "="*70)
                    print(f"FOUND LEARNED PARAMETERS for {reference_id}")
                    print(f"Score: {entry.get('match_score'):.1f}/10")
                    print(f"Angles: {entry['final_angles']}")
                    print("="*70)
                    return {
                        'final_angles': entry['final_angles'],
                        'match_score': entry.get('match_score'),
                        'iterations': 0,
                        'search_strategy': 'learned',
                        'from_cache': True
                    }

        # Get reference image
        # The reference_id is the key in graph['reference_renders']
        if reference_id not in self.kg.graph['reference_renders']:
            raise ValueError(f"Reference '{reference_id}' not found in KG")

        ref_data = self.kg.graph['reference_renders'][reference_id]
        ref_image_path = self.kg.kg_path / ref_data['image_path']

        if not ref_image_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

        reference_image = Image.open(ref_image_path)
        print(f"\nReference: {ref_image_path} ({reference_image.size})")

        # Coarse search
        coarse_result = self.coarse_search(render_fn, reference_image, distance)
        total_iterations = len(coarse_result['all_results'])

        if coarse_result.get('early_termination'):
            result = {
                'final_angles': coarse_result['best_angles'],
                'match_score': coarse_result['best_score'],
                'iterations': total_iterations,
                'search_strategy': 'coarse',
                'from_cache': False
            }
            self._store_learned_params(volume_name, reference_id, result)
            return result

        # Fine search
        fine_result = self.fine_search(render_fn, reference_image, coarse_result)
        total_iterations += len(fine_result['all_results'])

        result = {
            'final_angles': fine_result['best_angles'],
            'match_score': fine_result['best_score'],
            'iterations': total_iterations,
            'search_strategy': 'fine',
            'from_cache': False
        }

        self._store_learned_params(volume_name, reference_id, result)
        return result

    def _store_learned_params(self, volume_name: str, reference_id: str, result: Dict[str, Any]):
        """Store successful match in KG.

        Args:
            volume_name: Volume dataset name
            reference_id: Reference image ID
            result: Match result with angles and score
        """
        entry = {
            'reference_id': reference_id,
            'final_angles': result['final_angles'],
            'match_score': result['match_score'],
            'iterations_taken': result['iterations'],
            'search_strategy': result['search_strategy'],
            'timestamp': datetime.now().isoformat()
        }

        if volume_name not in self.kg.graph['learned_params']:
            self.kg.graph['learned_params'][volume_name] = []

        self.kg.graph['learned_params'][volume_name].append(entry)
        self.kg._save_graph()

        print(f"\n✓ Stored in KG: {volume_name} → {reference_id} (score={result['match_score']:.1f}/10)")
