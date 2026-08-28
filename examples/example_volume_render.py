"""Example usage of VolumeRenderAgent for 3D volume rendering."""

from ontovis import VolumeRenderAgent
import base64


def save_image_from_base64(base64_str: str, output_path: str):
    """Save base64 image to file."""
    image_data = base64.b64decode(base64_str)
    with open(output_path, 'wb') as f:
        f.write(image_data)
    print(f"Saved rendering to: {output_path}")


def render_foot_bones():
    """Example: Render foot dataset showing bones."""
    print("\n" + "="*70)
    print("Example 1: Render Foot - Show Bones")
    print("="*70 + "\n")

    agent = VolumeRenderAgent()

    results = agent.render(
        volume_path='../data/foot_256x256x256_uint8.raw',
        prompt="Show the bones in white and make everything else transparent. I want to see the skeletal structure clearly.",
        metadata={
            'dimensions': [256, 256, 256],
            'dtype': 'uint8',
            'name': 'Foot CT'
        },
        save_image='foot_bones.png'
    )

    print("Rendering complete!")
    print(f"\nExplanation: {results['explanation']}")
    print(f"\nRender parameters used:")
    print(f"  Camera position: {results['render_params']['camera_position']}")
    print(f"  Background: {results['render_params']['background_color']}")
    print(f"  Opacity points: {len(results['render_params']['opacity_mapping'])}")
    print(f"  Color points: {len(results['render_params']['color_mapping'])}")


def render_foot_heatmap():
    """Example: Render foot dataset as density heatmap."""
    print("\n" + "="*70)
    print("Example 2: Render Foot - Density Heatmap")
    print("="*70 + "\n")

    agent = VolumeRenderAgent()

    results = agent.render(
        volume_path='../data/foot_256x256x256_uint8.raw',
        prompt="Create a heat map visualization showing density variations. Use a blue-to-red color scale where blue is low density and red is high density.",
        metadata={
            'dimensions': [256, 256, 256],
            'dtype': 'uint8',
            'name': 'Foot CT'
        },
        save_image='foot_heatmap.png'
    )

    print("Rendering complete!")
    print(f"\nExplanation: {results['explanation']}")


def render_foot_tissue():
    """Example: Render showing both bones and soft tissue."""
    print("\n" + "="*70)
    print("Example 3: Render Foot - Bones and Soft Tissue")
    print("="*70 + "\n")

    agent = VolumeRenderAgent()

    results = agent.render(
        volume_path='../data/foot_256x256x256_uint8.raw',
        prompt="Show both bones and soft tissue. Make bones opaque white and soft tissue semi-transparent pink. Background should be black.",
        metadata={
            'dimensions': [256, 256, 256],
            'dtype': 'uint8',
            'name': 'Foot CT'
        },
        save_image='foot_tissue.png'
    )

    print("Rendering complete!")
    print(f"\nExplanation: {results['explanation']}")


def render_custom():
    """Example: Custom rendering with specific request."""
    print("\n" + "="*70)
    print("Example 4: Custom Rendering")
    print("="*70 + "\n")

    agent = VolumeRenderAgent()

    results = agent.render(
        volume_path='../data/foot_256x256x256_uint8.raw',
        prompt="Show only the high-density structures. Use a grayscale colormap and make the camera view from above.",
        metadata={
            'dimensions': [256, 256, 256],
            'dtype': 'uint8',
            'name': 'Foot CT'
        },
        save_image='foot_custom.png'
    )

    print("Rendering complete!")
    print(f"\nExplanation: {results['explanation']}")
    print(f"\nFull render parameters:")
    import json
    print(json.dumps(results['render_params'], indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "bones":
            render_foot_bones()
        elif example == "heatmap":
            render_foot_heatmap()
        elif example == "tissue":
            render_foot_tissue()
        elif example == "custom":
            render_custom()
        else:
            print(f"Unknown example: {example}")
            print("Available: bones, heatmap, tissue, custom")
    else:
        print("Running all examples...\n")
        render_foot_bones()
        render_foot_heatmap()
        render_foot_tissue()
        render_custom()

    print("\n" + "="*70)
    print("✓ All examples complete!")
    print("="*70)
