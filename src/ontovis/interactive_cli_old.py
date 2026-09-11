"""Interactive CLI for talking to an LLM with OntoVis agents as tools."""

import sys
from pathlib import Path
from typing import Annotated, TypedDict, Literal
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ontovis import VolumeAnalysisAgent, VolumeRenderAgent, VisionAgent, SmartVolumeRenderAgent, Config


# Define tools that wrap our agents
@tool
def analyze_volume(
    volume_path: str,
    description: str,
    dimensions_x: int = None,
    dimensions_y: int = None,
    dimensions_z: int = None,
    dtype: str = "uint8"
) -> str:
    """Analyze a volume dataset to understand its intensity distribution and identify features.

    Use this when the user wants to:
    - Understand what's in a dataset
    - Get statistics (min, max, mean, percentiles)
    - Identify features or materials
    - See a histogram
    - Know what intensity ranges correspond to what structures

    Args:
        volume_path: Path to volume file (.raw, .npy, .dat)
        description: What the dataset contains (e.g., "CT scan of foot")
        dimensions_x: X dimension (required for .raw files)
        dimensions_y: Y dimension (required for .raw files)
        dimensions_z: Z dimension (required for .raw files)
        dtype: Data type for .raw files (default: uint8)

    Returns:
        Analysis results with statistics and feature identification
    """
    try:
        agent = VolumeAnalysisAgent()

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        histogram_path = Path(volume_path).stem + "_histogram.png"

        results = agent.analyze_volume(
            volume_path=volume_path,
            user_description=description,
            metadata=metadata,
            save_histogram=histogram_path
        )

        stats = results['volume_stats']
        analysis = results['feature_analysis']

        output = f"""Volume Analysis Complete!

Statistics:
- Shape: {stats['shape']}
- Value range: {stats['min']:.1f} to {stats['max']:.1f}
- Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}
- Std dev: {stats['std']:.1f}

Histogram saved to: {histogram_path}

AI Feature Analysis:
{analysis}

You can now ask me questions about this data or ask me to render it.
"""
        return output

    except Exception as e:
        return f"Error analyzing volume: {e}"


@tool
def render_volume(
    volume_path: str,
    prompt: str,
    dimensions_x: int = None,
    dimensions_y: int = None,
    dimensions_z: int = None,
    dtype: str = "uint8",
    output_path: str = None
) -> str:
    """Render a 3D visualization of a volume dataset based on a natural language prompt.

    Use this when the user wants to:
    - Visualize the data
    - See bones, tissue, or specific structures
    - Create a 3D rendering
    - Show structures at a specific threshold

    Args:
        volume_path: Path to volume file
        prompt: What to render (e.g., "Show the bones", "Visualize at threshold 100")
        dimensions_x: X dimension (required for .raw files)
        dimensions_y: Y dimension (required for .raw files)
        dimensions_z: Z dimension (required for .raw files)
        dtype: Data type for .raw files (default: uint8)
        output_path: Where to save the image (default: auto-generated)

    Returns:
        Rendering results with explanation
    """
    try:
        agent = VolumeRenderAgent()

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        if output_path is None:
            output_path = Path(volume_path).stem + "_render.png"

        results = agent.render(
            volume_path=volume_path,
            prompt=prompt,
            metadata=metadata,
            save_image=output_path
        )

        instructions = results['instructions']

        return f"""Rendering Complete!

Saved to: {output_path}

Method: {instructions['method']}
Threshold: {instructions['threshold']}
Color: RGB{instructions['color']}
Opacity: {instructions['opacity']}

Explanation: {results['explanation']}

The 3D visualization has been saved. You can ask me to render it differently or analyze other datasets.
"""

    except Exception as e:
        return f"Error rendering volume: {e}"


@tool
def list_files(directory: str = "data") -> str:
    """List available volume data files in a directory.

    Use this when the user asks:
    - What files are available?
    - What datasets do I have?
    - Show me my data files

    Args:
        directory: Directory to search (default: data)

    Returns:
        List of available files
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Directory not found: {directory}"

        files = []
        for ext in ['*.raw', '*.npy', '*.dat']:
            files.extend(path.glob(ext))

        if not files:
            return f"No volume files found in {directory}"

        output = f"Available files in {directory}:\n\n"
        for f in sorted(files):
            size_mb = f.stat().st_size / (1024 * 1024)
            output += f"- {f.name} ({size_mb:.1f} MB)\n"

        return output

    except Exception as e:
        return f"Error listing files: {e}"


@tool
def analyze_image(
    image_path: str,
    prompt: str = "Describe this image in detail."
) -> str:
    """Analyze an image using vision AI capabilities.

    Use this when the user wants to:
    - Understand what's in an image
    - Describe image contents
    - Extract entities or relationships from images
    - Analyze visualizations, charts, or renderings
    - Get information about structure or orientation from images

    Args:
        image_path: Path to image file (.jpg, .jpeg, .png, .gif, .webp)
        prompt: What to analyze in the image (default: general description)

    Returns:
        Analysis results describing the image content
    """
    try:
        agent = VisionAgent()

        analysis = agent.analyze(
            image_path=image_path,
            prompt=prompt
        )

        return f"""Image Analysis Complete!

Image: {image_path}

Analysis:
{analysis}

You can ask follow-up questions about this image or analyze other images.
"""

    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        return f"Error analyzing image: {e}"


@tool
def smart_render_volume(
    volume_path: str,
    prompt: str,
    dimensions_x: int = None,
    dimensions_y: int = None,
    dimensions_z: int = None,
    dtype: str = "uint8",
    output_path: str = None,
    max_iterations: int = 5,
    save_intermediates: bool = False
) -> str:
    """Render a 3D volume with automatic framing adjustment using vision AI.

    This tool iteratively renders the volume and uses vision AI to check if the
    volume properly fills the image frame (not too small, not cropped). It adjusts
    the camera distance automatically until the framing is optimal.

    Use this when the user wants:
    - A well-framed 3D visualization with proper sizing
    - To ensure the full volume is visible without cropping
    - To avoid too much empty space around the volume
    - Better automatic framing than the basic render tool
    - To see all intermediate renders (set save_intermediates=True)

    Args:
        volume_path: Path to volume file
        prompt: What to render (e.g., "Show the bones", "Visualize at threshold 100")
        dimensions_x: X dimension (required for .raw files)
        dimensions_y: Y dimension (required for .raw files)
        dimensions_z: Z dimension (required for .raw files)
        dtype: Data type for .raw files (default: uint8)
        output_path: Where to save the image (default: auto-generated)
        max_iterations: Maximum iterations for adjustment (default: 5)
        save_intermediates: Save all iteration images to see progression (default: False)

    Returns:
        Rendering results with framing feedback
    """
    try:
        # Setup output directory for intermediates if requested
        output_dir = "render_iterations" if save_intermediates else None

        agent = SmartVolumeRenderAgent(
            max_iterations=max_iterations,
            save_intermediates=save_intermediates,
            output_dir=output_dir
        )

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        if output_path is None:
            output_path = Path(volume_path).stem + "_smart_render.png"

        results = agent.render(
            volume_path=volume_path,
            prompt=prompt,
            metadata=metadata,
            save_image=output_path
        )

        iterations = results.get('iterations', 1)
        all_iterations = results.get('all_iterations', [])

        intermediate_info = ""
        if save_intermediates and all_iterations:
            intermediate_info = f"\n\nAll {len(all_iterations)} iteration(s) saved to {output_dir}/:\n"
            for iter_info in all_iterations:
                intermediate_info += f"  - {Path(iter_info['path']).name} (adjustment: {iter_info['adjustment']})\n"

        return f"""Smart Rendering Complete!

Saved to: {output_path}
Iterations: {iterations} (automatically adjusted framing using vision AI)

The volume has been rendered with optimal framing - it fills the image properly
without being cropped or too small. The vision AI verified the framing across
{iterations} iteration(s) to ensure proper sizing.{intermediate_info}

You can ask me to render differently or analyze other datasets.
"""

    except Exception as e:
        return f"Error in smart rendering: {e}"


# Build the agent
def create_interactive_agent():
    """Create the conversational agent with tools."""

    # Load config
    config = Config()
    llm_config = config.get_llm_config()

    # Create LLM with tools
    llm = ChatAnthropic(
        model=llm_config['model'],
        api_key=llm_config['api_key'],
        base_url=llm_config['base_url'],
        max_tokens=4096
    )

    tools = [analyze_volume, render_volume, smart_render_volume, list_files, analyze_image]
    llm_with_tools = llm.bind_tools(tools)

    # Define agent logic
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        """Decide whether to use tools or end."""
        messages = state['messages']
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        """Call the LLM."""
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def main():
    """Run the interactive CLI."""

    print("="*70)
    print("OntoVis Interactive CLI")
    print("="*70)
    print("\nChat with an AI that can use volume analysis and rendering tools!")
    print("\nAvailable commands:")
    print("  - Ask to analyze a dataset")
    print("  - Ask to render visualizations")
    print("  - Ask what files are available")
    print("  - Type 'quit' or 'exit' to leave")
    print("\nExamples:")
    print('  "Analyze the foot dataset at data/foot_256x256x256_uint8.raw"')
    print('  "Show me the bones"')
    print('  "What files do I have?"')
    print("="*70)
    print()

    # Create agent
    try:
        agent = create_interactive_agent()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Make sure your .config file is set up correctly.")
        return

    # Conversation state
    state = {"messages": []}

    # Add system message
    system_msg = AIMessage(content="""I'm your OntoVis assistant! I can help you analyze and visualize 3D volume datasets.

I have access to these tools:
- **analyze_volume**: Analyze intensity distributions, get statistics, identify features
- **render_volume**: Create 3D visualizations based on your prompts
- **smart_render_volume**: Create optimally-framed 3D visualizations using iterative vision AI feedback
- **analyze_image**: Analyze images using vision AI (describe contents, extract information)
- **list_files**: Show available data files

Just tell me what you want to do in natural language, and I'll use the right tools!""")

    print("Assistant: " + system_msg.content)
    print()

    # Main loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break

            # Add user message
            state["messages"].append(HumanMessage(content=user_input))

            # Get response
            print("\nAssistant: ", end="", flush=True)

            result = agent.invoke(state)
            state = {"messages": result["messages"]}

            # Print assistant's response
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(last_message.content)

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


if __name__ == "__main__":
    main()
