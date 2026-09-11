#!/usr/bin/env python3
"""
Interactive Volume Rendering Agent with Vision-Guided Angle Matching

Usage:
    python interactive_agent.py

Examples:
    > match a_skull_front_view
    > match a_skull_side_view
    > show references
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pyvista as pv
from PIL import Image

from src.ontovis import VolumeRenderAgent, AngleMatcher, MultimodalKnowledgeGraph


# Default volume configuration
DEFAULT_VOLUME = {
    'path': 'data/3d_datasets/vis_male_256x256x128_uint8.raw',
    'dimensions': (256, 256, 128),
    'dtype': 'uint8',
    'name': 'Visible Male'
}


class InteractiveRenderAgent:
    """Interactive agent for volume rendering with conversational interface."""

    def __init__(self, kg_path='.kg'):
        """Initialize the interactive agent."""
        self.kg = MultimodalKnowledgeGraph(kg_path)
        self.matcher = AngleMatcher(kg_path=kg_path)
        self.agent = VolumeRenderAgent(kg_path=kg_path)

        # Load current volume
        self.current_volume = DEFAULT_VOLUME.copy()

        print("\n" + "="*70)
        print("INTERACTIVE VOLUME RENDERING AGENT")
        print("="*70)
        print(f"Knowledge Graph: {kg_path}")
        print(f"Current Volume: {self.current_volume['name']}")
        print(f"  Path: {self.current_volume['path']}")
        print(f"  Dimensions: {self.current_volume['dimensions']}")
        print("="*70)

        # Show available references
        refs = list(self.kg.graph['reference_renders'].keys())
        if refs:
            print(f"\nAvailable reference images:")
            for ref in refs:
                ref_data = self.kg.graph['reference_renders'][ref]
                print(f"  - {ref}: {ref_data.get('description', 'No description')}")

        print("\n" + "="*70)
        print("QUICK START")
        print("="*70)
        print("  match a_skull_front_view     - Match skull front view")
        print("  match a_skull_side_view      - Match skull side view")
        print("  show references              - List all references")
        print("  help                         - Show all commands")
        print("  quit                         - Exit")
        print("="*70)

    def create_render_function(self, volume_path, dimensions, dtype):
        """Create a render function for angle matching."""
        # Load volume once
        volume_data = np.fromfile(volume_path, dtype=dtype)
        volume_data = volume_data.reshape(dimensions)

        def render_fn(azimuth, elevation, roll, distance):
            """Render volume at specified angles."""
            plotter = pv.Plotter(off_screen=True, window_size=(800, 800))

            # Create volume grid
            grid = pv.ImageData()
            grid.dimensions = np.array(dimensions) + 1
            grid.spacing = (1, 1, 1)
            grid.origin = (0, 0, 0)
            grid.point_data["values"] = volume_data.flatten(order="F")

            # Add volume
            plotter.add_volume(grid, cmap="gray_r", opacity="linear", shade=True)
            plotter.set_background("black")

            # Calculate camera position
            center = np.array([d / 2.0 for d in dimensions])
            radius = max(dimensions) * distance

            azimuth_rad = np.radians(azimuth)
            elevation_rad = np.radians(elevation)

            x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = radius * np.sin(elevation_rad)

            camera_position = center + np.array([x, y, z])

            # Calculate up vector with roll
            view_dir = center - camera_position
            view_dir = view_dir / np.linalg.norm(view_dir)

            base_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(view_dir, base_up)) > 0.99:
                base_up = np.array([0.0, 1.0, 0.0])

            right = np.cross(base_up, view_dir)
            right = right / np.linalg.norm(right)

            up = np.cross(view_dir, right)
            up = up / np.linalg.norm(up)

            # Apply roll
            if roll != 0:
                roll_rad = np.radians(roll)
                cos_roll = np.cos(roll_rad)
                sin_roll = np.sin(roll_rad)
                k_dot_v = np.dot(view_dir, up)
                k_cross_v = np.cross(view_dir, up)
                up = up * cos_roll + k_cross_v * sin_roll + view_dir * k_dot_v * (1 - cos_roll)
                up = up / np.linalg.norm(up)

            # Set camera
            plotter.camera_position = [
                tuple(camera_position),
                tuple(center),
                tuple(up)
            ]

            # Render
            img_array = plotter.screenshot(return_img=True)
            plotter.close()

            return Image.fromarray(img_array)

        return render_fn

    def match_reference(self, reference_id, distance=2.0):
        """Use vision-guided angle matching."""
        print(f"\n{'='*70}")
        print(f"MATCHING: {reference_id}")
        print(f"{'='*70}")

        # Check if reference exists
        if reference_id not in self.kg.graph['reference_renders']:
            print(f"\n❌ Reference '{reference_id}' not found")
            print("\nAvailable references:")
            for ref in self.kg.graph['reference_renders'].keys():
                print(f"  - {ref}")
            return None

        # Create render function
        print("\nPreparing...")
        render_fn = self.create_render_function(
            self.current_volume['path'],
            self.current_volume['dimensions'],
            self.current_volume['dtype']
        )

        # Find best angle
        print("Searching for best angle...")
        print("(First run: 5-15 min, subsequent runs: instant)\n")

        result = self.matcher.find_best_angle(
            render_fn=render_fn,
            reference_id=reference_id,
            volume_path=self.current_volume['path'],
            distance=distance
        )

        # Save final render
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("renders") / f"{timestamp}_matched_{reference_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        final_image = render_fn(
            result['final_angles']['azimuth'],
            result['final_angles']['elevation'],
            result['final_angles']['roll'],
            result['final_angles']['distance']
        )

        output_path = output_dir / "matched_render.png"
        final_image.save(output_path)

        # Save metadata
        metadata = {
            'reference_id': reference_id,
            'volume': self.current_volume,
            'result': result,
            'timestamp': timestamp
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Display results
        print(f"\n{'='*70}")
        print("✓ MATCH COMPLETE")
        print(f"{'='*70}")
        print(f"Score:      {result['match_score']:.1f}/10")
        print(f"Strategy:   {result['search_strategy']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Cached:     {result.get('from_cache', False)}")
        print(f"\nAngles:")
        print(f"  Azimuth:   {result['final_angles']['azimuth']}°")
        print(f"  Elevation: {result['final_angles']['elevation']}°")
        print(f"  Roll:      {result['final_angles']['roll']}°")
        print(f"\nSaved: {output_path}")
        print(f"{'='*70}\n")

        return result

    def show_references(self):
        """Show all available reference images."""
        refs = self.kg.graph['reference_renders']

        print(f"\n{'='*70}")
        print("AVAILABLE REFERENCES")
        print(f"{'='*70}")

        if not refs:
            print("No references in KG")
        else:
            for ref_id, ref_data in refs.items():
                print(f"\n{ref_id}:")
                print(f"  {ref_data.get('description', 'N/A')}")
                print(f"  Image: {ref_data.get('image_path', 'N/A')}")

        print(f"{'='*70}\n")

    def run(self):
        """Run interactive loop."""
        print("\nReady! Type a command (or 'help' for help, 'quit' to exit)\n")

        while True:
            try:
                user_input = input("> ").strip()

                if not user_input:
                    continue

                parts = user_input.lower().split()
                command = parts[0]

                if command in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                elif command == 'help':
                    print("\n" + "="*70)
                    print("COMMANDS")
                    print("="*70)
                    print("  match <ref>        - Match reference (e.g., match a_skull_front_view)")
                    print("  show references    - List available references")
                    print("  set volume <path>  - Change volume")
                    print("  help               - Show this help")
                    print("  quit               - Exit")
                    print("\nExamples:")
                    print("  > match a_skull_front_view")
                    print("  > match a_skull_side_view")
                    print("  > show references")
                    print("="*70 + "\n")

                elif command == 'show' and len(parts) > 1 and parts[1] == 'references':
                    self.show_references()

                elif command == 'match':
                    if len(parts) < 2:
                        print("\n❌ Usage: match <reference_id>")
                        print("Type 'show references' to see available references\n")
                    else:
                        reference_id = parts[1]
                        self.match_reference(reference_id)

                elif command == 'set' and len(parts) > 1 and parts[1] == 'volume':
                    if len(parts) < 3:
                        print("\n❌ Usage: set volume <path>\n")
                    else:
                        volume_path = ' '.join(parts[2:])
                        if not Path(volume_path).exists():
                            print(f"\n❌ Volume not found: {volume_path}\n")
                        else:
                            self.current_volume['path'] = volume_path
                            print(f"\n✓ Volume: {volume_path}\n")

                else:
                    print(f"\n❌ Unknown: {command}")
                    print("Type 'help' for commands\n")

            except KeyboardInterrupt:
                print("\n\n(Ctrl+C) Type 'quit' to exit\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()


def main():
    """Entry point."""
    try:
        agent = InteractiveRenderAgent(kg_path='.kg')
        agent.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
