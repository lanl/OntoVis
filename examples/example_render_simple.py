"""Simple example of using the generic VolumeRenderAgent."""

from ontovis import VolumeRenderAgent

# Initialize the agent once
agent = VolumeRenderAgent()

print("="*70)
print("Generic Volume Rendering Agent - Examples")
print("="*70)

# Example 1: Show bones
print("\nExample 1: Show bones")
print("-" * 70)
result = agent.render(
    volume_path='../data/foot_256x256x256_uint8.raw',
    prompt='Show the bones',
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8'
    },
    save_image='example1_bones.png'
)
print(f"✓ Saved to: example1_bones.png")
print(f"Explanation: {result['explanation']}")
print(f"Threshold used: {result['instructions']['threshold']}")

# Example 2: High-density structures only
print("\n" + "="*70)
print("\nExample 2: Only high-density structures")
print("-" * 70)
result = agent.render(
    volume_path='../data/foot_256x256x256_uint8.raw',
    prompt='Show only high-density structures above 150',
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8'
    },
    save_image='example2_highdensity.png'
)
print(f"✓ Saved to: example2_highdensity.png")
print(f"Explanation: {result['explanation']}")
print(f"Threshold used: {result['instructions']['threshold']}")

# Example 3: All tissue
print("\n" + "="*70)
print("\nExample 3: All tissue structures")
print("-" * 70)
result = agent.render(
    volume_path='../data/foot_256x256x256_uint8.raw',
    prompt='Show all tissue structures including soft tissue',
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8'
    },
    save_image='example3_alltissue.png'
)
print(f"✓ Saved to: example3_alltissue.png")
print(f"Explanation: {result['explanation']}")
print(f"Threshold used: {result['instructions']['threshold']}")

# Example 4: Specific threshold
print("\n" + "="*70)
print("\nExample 4: Specific threshold value")
print("-" * 70)
result = agent.render(
    volume_path='../data/foot_256x256x256_uint8.raw',
    prompt='Visualize structures at threshold 100',
    metadata={
        'dimensions': [256, 256, 256],
        'dtype': 'uint8'
    },
    save_image='example4_threshold100.png'
)
print(f"✓ Saved to: example4_threshold100.png")
print(f"Explanation: {result['explanation']}")
print(f"Threshold used: {result['instructions']['threshold']}")

print("\n" + "="*70)
print("✓ All examples complete!")
print("="*70)
print("\nHow it works:")
print("1. You provide: volume path + natural language prompt")
print("2. AI interprets your prompt and data statistics")
print("3. AI chooses rendering parameters (threshold, color, etc.)")
print("4. System generates 3D visualization")
print("\nThe agent automatically:")
print("- Analyzes your data (min, max, percentiles)")
print("- Understands your prompt (bones, tissue, threshold, etc.)")
print("- Selects appropriate visualization parameters")
print("- Renders and saves the image")
