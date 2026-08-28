"""
Example usage of the VisionAgent for analyzing images.

Before running:
1. Copy .config.example to .config
2. Add your Anthropic API key to .config
"""

from ontovis import VisionAgent


def main():
    try:
        # Initialize the vision agent (reads from .config)
        agent = VisionAgent()
        print("Vision agent initialized successfully!\n")

        # Example 1: Basic image analysis
        image_path = "/home/pascalgrosset/projects/OntoVis/data/skull.png"  # Replace with your image path
        prompt="What do you see in this image? which direction is if facing"

        print(f"Analyzing image: {image_path}")
        analysis = agent.analyze(
            image_path,
            prompt=prompt
        )
        print(f"\nAnalysis:\n{analysis}\n")

        # Example 2: Extract entities for knowledge graph
        print("=" * 60)
        print("Extracting entities and relationships...")
        entities = agent.extract_entities_from_image(image_path)
        print(f"\nExtracted Entities:\n{entities}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. Copy .config.example to .config")
        print("2. Add your Anthropic API key to .config")
        print("3. Provide a valid image path")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()