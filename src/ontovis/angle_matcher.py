"""Automatic camera angle matching against reference images.

This module finds the optimal camera angle by:
1. Getting reference image from knowledge graph
2. Testing multiple camera angles in a grid search
3. Using vision AI to compare each render to the reference
4. Selecting the angle with the best match
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from .config import Config
from .render_agent_v2 import VolumeRenderAgent
from .multimodal_kg import MultimodalKnowledgeGraph


class AngleMatcher:
    """Find optimal camera angle by matching against reference images."""

    def __init__(self, config_path=None):
        """Initialize angle matcher.

        Args:
            config_path: Path to config file
        """
        self.config = Config(config_path)
        llm_config = self.config.get_llm_config()

        self.llm = ChatAnthropic(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            max_tokens=4096
        )

        self.render_agent = VolumeRenderAgent(config_path)
        self.kg = MultimodalKnowledgeGraph()

    def find_best_angle(
        self,
        volume_path: str,
        reference_image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        search_strategy: str = "coarse_to_fine"
    ) -> Dict[str, Any]:
        """Find the camera angle that best matches a reference image.

        Args:
            volume_path: Path to volume file
            reference_image_path: Path to reference image from KG
            metadata: Optional volume metadata
            output_dir: Directory to save test renders
            search_strategy: "coarse_to_fine", "grid", or "custom"

        Returns:
            Dict with best angle, match score, and all tested angles
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path("angle_search_results")
            output_path.mkdir(parents=True, exist_ok=True)

        # Load reference image for comparison
        ref_path = Path(reference_image_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

        print(f"🎯 Finding best camera angle to match: {reference_image_path}")
        print(f"📊 Search strategy: {search_strategy}\n")

        # Define search space based on strategy
        if search_strategy == "coarse_to_fine":
            # Phase 1: Coarse grid (45° increments)
            # Phase 2: Fine grid around best match (15° increments)
            angles = self._coarse_to_fine_search(
                volume_path, reference_image_path, metadata, output_path
            )
        elif search_strategy == "grid":
            # Simple grid search (30° increments)
            angles = self._generate_grid_angles(elevation_step=30, azimuth_step=30)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

        # Test all angles
        results = []
        for i, (elevation, azimuth) in enumerate(angles, 1):
            print(f"Testing angle {i}/{len(angles)}: elevation={elevation}°, azimuth={azimuth}°")

            # Render at this angle
            render_path = output_path / f"test_elev{elevation}_azim{azimuth}.png"

            try:
                render_result = self.render_agent.render(
                    volume_path=volume_path,
                    prompt=f"Render bones with camera elevation {elevation} degrees and azimuth {azimuth} degrees",
                    metadata=metadata,
                    save_image=str(render_path)
                )

                if not render_result.get("rendered_image_base64"):
                    print(f"  ⚠️  Render failed")
                    continue

                # Compare to reference
                match_score = self._compare_to_reference(
                    test_render_path=str(render_path),
                    reference_path=str(ref_path)
                )

                results.append({
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "match_score": match_score,
                    "render_path": str(render_path),
                    "render_instructions": render_result.get("instructions")
                })

                print(f"  Match score: {match_score}/10")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        if not results:
            raise RuntimeError("No successful renders produced")

        # Sort by match score (higher is better)
        results.sort(key=lambda x: x["match_score"], reverse=True)
        best = results[0]

        print(f"\n✅ Best match found!")
        print(f"   Elevation: {best['elevation']}°")
        print(f"   Azimuth: {best['azimuth']}°")
        print(f"   Match score: {best['match_score']}/10")
        print(f"   Render: {best['render_path']}")

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "volume_path": volume_path,
            "reference_image": reference_image_path,
            "search_strategy": search_strategy,
            "best_match": best,
            "all_results": results,
            "angles_tested": len(results)
        }

        summary_path = output_path / "angle_search_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📄 Summary saved: {summary_path}")

        return summary

    def _coarse_to_fine_search(
        self,
        volume_path: str,
        reference_path: str,
        metadata: Optional[Dict],
        output_path: Path
    ) -> List[Tuple[int, int]]:
        """Two-phase search: coarse grid then fine refinement.

        Args:
            volume_path: Volume file path
            reference_path: Reference image path
            metadata: Volume metadata
            output_path: Output directory

        Returns:
            List of (elevation, azimuth) tuples to test
        """
        # Phase 1: Coarse grid (45° increments)
        print("Phase 1: Coarse grid search (45° increments)")
        coarse_angles = self._generate_grid_angles(elevation_step=45, azimuth_step=45)

        # Test coarse angles
        coarse_results = []
        for elevation, azimuth in coarse_angles:
            render_path = output_path / f"coarse_elev{elevation}_azim{azimuth}.png"

            try:
                render_result = self.render_agent.render(
                    volume_path=volume_path,
                    prompt=f"Render bones with camera elevation {elevation} degrees and azimuth {azimuth} degrees",
                    metadata=metadata,
                    save_image=str(render_path)
                )

                if render_result.get("rendered_image_base64"):
                    match_score = self._compare_to_reference(
                        test_render_path=str(render_path),
                        reference_path=reference_path
                    )

                    coarse_results.append({
                        "elevation": elevation,
                        "azimuth": azimuth,
                        "score": match_score
                    })

                    print(f"  elev={elevation}°, azim={azimuth}° → score={match_score}/10")

            except Exception as e:
                print(f"  elev={elevation}°, azim={azimuth}° → error: {e}")
                continue

        if not coarse_results:
            # Fallback to full grid
            return self._generate_grid_angles(elevation_step=30, azimuth_step=30)

        # Find best coarse angle
        coarse_results.sort(key=lambda x: x["score"], reverse=True)
        best_coarse = coarse_results[0]

        print(f"\nBest coarse match: elev={best_coarse['elevation']}°, azim={best_coarse['azimuth']}°")
        print("\nPhase 2: Fine grid search (15° increments around best match)")

        # Phase 2: Fine grid around best match (±45° with 15° increments)
        fine_angles = []
        for elev_offset in range(-45, 60, 15):
            for azim_offset in range(-45, 60, 15):
                elevation = best_coarse['elevation'] + elev_offset
                azimuth = (best_coarse['azimuth'] + azim_offset) % 360

                # Keep elevation in valid range
                if -90 <= elevation <= 90:
                    fine_angles.append((elevation, azimuth))

        return fine_angles

    def _generate_grid_angles(
        self,
        elevation_step: int = 30,
        azimuth_step: int = 30
    ) -> List[Tuple[int, int]]:
        """Generate a grid of camera angles to test.

        Args:
            elevation_step: Degrees between elevation samples
            azimuth_step: Degrees between azimuth samples

        Returns:
            List of (elevation, azimuth) tuples
        """
        angles = []

        # Elevation: -90 (bottom) to +90 (top)
        for elevation in range(-90, 91, elevation_step):
            # Azimuth: 0 to 360 (full circle)
            for azimuth in range(0, 360, azimuth_step):
                angles.append((elevation, azimuth))

        return angles

    def _compare_to_reference(
        self,
        test_render_path: str,
        reference_path: str
    ) -> float:
        """Compare a test render to the reference image using vision AI.

        Args:
            test_render_path: Path to test render
            reference_path: Path to reference image

        Returns:
            Match score from 0-10 (10 = perfect match)
        """
        import base64

        # Load both images as base64
        with open(test_render_path, 'rb') as f:
            test_b64 = base64.b64encode(f.read()).decode('utf-8')

        with open(reference_path, 'rb') as f:
            ref_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Ask AI to compare
        prompt = """Compare these two medical volume renderings and rate how well they match in terms of camera angle/viewpoint.

Reference image (target):
[First image below]

Test render (candidate):
[Second image below]

Focus on:
1. Camera viewing angle (front, side, top, etc.)
2. Elevation (how high/low the camera is)
3. Rotation/orientation of the anatomy
4. Overall viewpoint similarity

Ignore differences in:
- Color/brightness (we only care about angle)
- Resolution
- Minor rendering artifacts

Rate the viewpoint match from 0-10:
- 10 = Identical viewing angle
- 7-9 = Very similar angle, minor differences
- 4-6 = Similar angle, noticeable differences
- 1-3 = Different angle
- 0 = Completely different angle

Respond with ONLY a JSON object:
{
  "match_score": <number 0-10>,
  "reasoning": "<brief explanation of the score>",
  "angle_similarity": "<identical|very_similar|similar|different|very_different>"
}
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{ref_b64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{test_b64}"
                    }
                }
            ]
        )

        response = self.llm.invoke([message])

        # Parse JSON response
        try:
            response_text = response.content.strip()
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            result = json.loads(response_text)
            match_score = float(result.get("match_score", 0))

            return match_score

        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Failed to parse comparison result: {e}")
            print(f"Response: {response.content[:200]}")
            # Return neutral score on error
            return 5.0
