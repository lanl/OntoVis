"""Interactive CLI for OntoVis with run management and logging."""

import sys
from pathlib import Path
from typing import Optional, List
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from . import VolumeAnalysisAgent, VolumeRenderAgent, VisionAgent, SmartVolumeRenderAgent, Config
from .run_manager import RunManager
from .kg_tool import KGTool
from .angle_matcher import AngleMatcher


# Global run manager (set by run_interactive_cli)
_run_manager: Optional[RunManager] = None
_kg_tool: Optional[KGTool] = None


def get_run_manager() -> RunManager:
    """Get the current run manager."""
    if _run_manager is None:
        raise RuntimeError("Run manager not initialized")
    return _run_manager


def get_kg_tool() -> KGTool:
    """Get the current KG tool."""
    if _kg_tool is None:
        raise RuntimeError("KG tool not initialized")
    return _kg_tool


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
    """Analyze a volume dataset to understand its intensity distribution and identify features."""
    run_mgr = get_run_manager()
    run_mgr.log_operation("analyze_volume", {
        "volume_path": volume_path,
        "description": description,
        "dimensions": [dimensions_x, dimensions_y, dimensions_z] if dimensions_x else None
    })

    try:
        agent = VolumeAnalysisAgent()

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        # Save histogram to run directory
        histogram_path = run_mgr.get_histogram_path(f"{Path(volume_path).stem}_histogram.png")

        results = agent.analyze_volume(
            volume_path=volume_path,
            user_description=description,
            metadata=metadata,
            save_histogram=str(histogram_path)
        )

        stats = results['volume_stats']
        analysis = results['feature_analysis']

        run_mgr.logger.info(f"Analysis complete: {volume_path}")
        run_mgr.logger.info(f"  Shape: {stats['shape']}, Range: [{stats['min']:.1f}, {stats['max']:.1f}]")

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
        run_mgr.logger.error(f"Error analyzing volume: {e}", exc_info=True)
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
    """Render a 3D visualization of a volume dataset based on a natural language prompt."""
    run_mgr = get_run_manager()
    run_mgr.log_operation("render_volume", {
        "volume_path": volume_path,
        "prompt": prompt
    })

    try:
        agent = VolumeRenderAgent()

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        if output_path is None:
            output_path = run_mgr.get_render_path(f"{Path(volume_path).stem}_render.png")
        else:
            # If path already contains renders_dir, don't double it
            output_path_obj = Path(output_path)
            if str(run_mgr.renders_dir) in str(output_path_obj):
                # Already has full path, use as-is
                output_path = output_path_obj
            else:
                # Just a filename, add renders_dir
                output_path = run_mgr.get_render_path(output_path)

        results = agent.render(
            volume_path=volume_path,
            prompt=prompt,
            metadata=metadata,
            save_image=str(output_path)
        )

        instructions = results['instructions']

        run_mgr.logger.info(f"Render complete: {output_path}")

        return f"""Rendering Complete!

Saved to: {output_path}

Method: {instructions['method']}
Threshold: {instructions['threshold']}
Color: RGB{instructions['color']}
Opacity: {instructions['opacity']}

Explanation: {results['explanation']}

The 3D visualization has been saved.
"""

    except Exception as e:
        run_mgr.logger.error(f"Error rendering volume: {e}", exc_info=True)
        return f"Error rendering volume: {e}"


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
    """Render a 3D volume with automatic framing adjustment using vision AI."""
    run_mgr = get_run_manager()
    run_mgr.log_operation("smart_render_volume", {
        "volume_path": volume_path,
        "prompt": prompt,
        "max_iterations": max_iterations,
        "save_intermediates": save_intermediates
    })

    try:
        # Setup output directory for intermediates if requested
        # Create a unique timestamped subfolder for this render session
        if save_intermediates:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            volume_name = Path(volume_path).stem
            output_dir = run_mgr.renders_dir / "iterations" / f"{volume_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            run_mgr.logger.info(f"Saving iterations to: {output_dir}")
        else:
            output_dir = None

        agent = SmartVolumeRenderAgent(
            max_iterations=max_iterations,
            save_intermediates=save_intermediates,
            output_dir=str(output_dir) if output_dir else None
        )

        metadata = {}
        if volume_path.endswith('.raw'):
            if not all([dimensions_x, dimensions_y, dimensions_z]):
                return "Error: .raw files require dimensions_x, dimensions_y, dimensions_z"
            metadata['dimensions'] = [dimensions_x, dimensions_y, dimensions_z]
            metadata['dtype'] = dtype

        if output_path is None:
            output_path = run_mgr.get_render_path(f"{Path(volume_path).stem}_smart_render.png")
        else:
            # If path already contains renders_dir, don't double it
            output_path_obj = Path(output_path)
            if str(run_mgr.renders_dir) in str(output_path_obj):
                # Already has full path, use as-is
                output_path = output_path_obj
            else:
                # Just a filename, add renders_dir
                output_path = run_mgr.get_render_path(output_path)

        results = agent.render(
            volume_path=volume_path,
            prompt=prompt,
            metadata=metadata,
            save_image=str(output_path)
        )

        iterations = results.get('iterations', 1)
        all_iterations = results.get('all_iterations', [])

        run_mgr.logger.info(f"Smart render complete: {output_path} ({iterations} iterations)")

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
        run_mgr.logger.error(f"Error in smart rendering: {e}", exc_info=True)
        return f"Error in smart rendering: {e}"


@tool
def list_files(directory: str = "data") -> str:
    """List available volume data files in a directory."""
    run_mgr = get_run_manager()

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
    """Analyze an image using vision AI capabilities."""
    run_mgr = get_run_manager()
    run_mgr.log_operation("analyze_image", {"image_path": image_path})

    try:
        agent = VisionAgent()

        analysis = agent.analyze(
            image_path=image_path,
            prompt=prompt
        )

        run_mgr.logger.info(f"Image analysis complete: {image_path}")

        return f"""Image Analysis Complete!

Image: {image_path}

Analysis:
{analysis}

You can ask follow-up questions about this image or analyze other images.
"""

    except FileNotFoundError:
        return f"Error: Image not found at {image_path}"
    except Exception as e:
        run_mgr.logger.error(f"Error analyzing image: {e}", exc_info=True)
        return f"Error analyzing image: {e}"


# ==================== KNOWLEDGE GRAPH TOOLS ====================

@tool
def kg_query_convention(structure_name: str) -> str:
    """Query the knowledge graph for rendering convention of an anatomical structure.

    Use this to get:
    - Recommended color for a structure (bones, arteries, veins, etc.)
    - Typical intensity range in volume data
    - Opacity recommendations
    - Medical rationale for the convention

    Args:
        structure_name: Name of structure (e.g., "bone", "artery", "vein", "soft_tissue")

    Returns:
        Convention information or error message
    """
    kg = get_kg_tool()

    try:
        convention = kg.query_convention(structure_name)

        if not convention:
            return f"No convention found for '{structure_name}'. Available structures: bone, artery, vein, soft_tissue, fat, air"

        output = f"""Rendering Convention for {structure_name.title()}:

Color: {convention['color']['name'].title()} - RGB{convention['color']['rgb']}
Opacity Range: {convention['opacity_range']}
"""

        if convention.get('intensity_range'):
            output += f"Typical Intensity Range: {convention['intensity_range']}\n"

        if convention.get('medical_rationale'):
            output += f"\nMedical Rationale:\n{convention['medical_rationale']}\n"

        return output

    except Exception as e:
        return f"Error querying convention: {e}"


@tool
def kg_get_recommendations(dataset_name: str, structures: List[str]) -> str:
    """Get comprehensive rendering recommendations from the knowledge graph.

    Use this before rendering to get:
    - Dataset-specific information (intensity ranges, dimensions, notes)
    - Rendering conventions for each structure
    - Reference images (good examples)
    - Previously successful rendering parameters

    Args:
        dataset_name: Name of dataset (e.g., "VIS_male_128", "skull_256", "foot_256")
        structures: List of structures to render (e.g., ["bone"], ["bone", "artery"])

    Returns:
        Comprehensive recommendations
    """
    kg = get_kg_tool()

    try:
        recommendations = kg.get_recommendations(dataset_name, structures)

        output = f"""Knowledge Graph Recommendations for {dataset_name}:

"""

        # Dataset info
        if recommendations['dataset']:
            ds = recommendations['dataset']
            output += f"""Dataset Information:
- Anatomy: {ds['anatomy']}
- Modality: {ds['modality']}
- Dimensions: {ds['dimensions']}
- Data Type: {ds['dtype']}

Typical Intensity Ranges:
"""
            for struct, range_val in ds['intensity_ranges'].items():
                output += f"  - {struct}: {range_val}\n"

            if ds.get('notes'):
                output += f"\nNotes: {ds['notes']}\n"

        # Conventions
        if recommendations['structures']:
            output += f"\nRendering Conventions:\n"
            for struct, conv in recommendations['structures'].items():
                output += f"  - {struct}: {conv['color']['name']} {conv['color']['rgb']}\n"
                if conv.get('intensity_range'):
                    output += f"    Intensity: {conv['intensity_range']}\n"

        # References
        ref_count = len(recommendations['reference_images'])
        output += f"\nReference Images: {ref_count} available\n"

        if ref_count > 0:
            output += "\nAvailable Reference Images:\n"
            for ref in recommendations['reference_images']:
                full_path = ref.get('image_full_path', 'N/A')
                exists = "✓" if ref.get('image_exists') else "✗"
                output += f"  {exists} {ref['id']}: {full_path}\n"
                output += f"     Tags: {', '.join(ref.get('tags', []))}\n"

        # Learned params
        learned_count = len(recommendations['learned_params'])
        output += f"Previously Successful Renders: {learned_count}\n"

        if learned_count > 0:
            best = recommendations['learned_params'][0]
            output += f"  Most recent: {best['id']} ({best['iterations']} iterations)\n"

        return output

    except Exception as e:
        return f"Error getting recommendations: {e}"


@tool
def kg_get_dataset_info(dataset_name: str) -> str:
    """Get information about a dataset from the knowledge graph.

    Args:
        dataset_name: Name of dataset (e.g., "VIS_male_128", "skull_256", "foot_256")

    Returns:
        Dataset information or error message
    """
    kg = get_kg_tool()

    try:
        dataset = kg.get_dataset_knowledge(dataset_name=dataset_name)

        if not dataset:
            return f"No dataset knowledge found for '{dataset_name}'. Try: VIS_male_128, skull_256, foot_256"

        output = f"""Dataset: {dataset['name']}

Description: {dataset['description']}
Anatomy: {dataset['anatomy']}
Modality: {dataset['modality']}
Dimensions: {dataset['dimensions']}
Data Type: {dataset['dtype']}

Typical Intensity Ranges:
"""
        for struct, range_val in dataset['intensity_ranges'].items():
            output += f"  - {struct}: {range_val}\n"

        if dataset.get('notes'):
            output += f"\nNotes:\n{dataset['notes']}\n"

        return output

    except Exception as e:
        return f"Error getting dataset info: {e}"


@tool
def kg_get_statistics() -> str:
    """Get statistics about the knowledge graph contents.

    Returns:
        KG statistics summary
    """
    kg = get_kg_tool()

    try:
        stats = kg.get_statistics()

        output = f"""Knowledge Graph Statistics:

Conventions: {stats['total_conventions']}
Colormaps: {stats['total_colormaps']}
Reference Renders: {stats['total_reference_renders']}
"""

        if stats['reference_by_category']:
            output += "\nReferences by Category:\n"
            for cat, count in stats['reference_by_category'].items():
                output += f"  - {cat}: {count}\n"

        output += f"""
Datasets: {stats['total_datasets']}
Learned Renders: {stats['total_learned_renders']}
Relationships: {stats['total_relationships']}
"""

        return output

    except Exception as e:
        return f"Error getting statistics: {e}"


@tool
def list_recent_renders(count: int = 10) -> str:
    """List the most recent rendered images in this run with their full paths.

    This shows the last N images that were created, making it easy to see results.

    Args:
        count: Number of recent renders to show (default: 10)

    Returns:
        List of recent render paths with timestamps and sizes
    """
    run_mgr = get_run_manager()

    try:
        # Get all PNG files from renders directory (not iterations subdirectory)
        renders_dir = run_mgr.renders_dir

        # Find all PNG files, excluding the iterations subfolder
        png_files = []
        for png_file in renders_dir.glob("*.png"):
            if png_file.is_file():
                png_files.append(png_file)

        if not png_files:
            return f"No renders found in {renders_dir}"

        # Sort by modification time (newest first)
        png_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Take the N most recent
        recent = png_files[:count]

        output = f"Recent Renders (showing {len(recent)} of {len(png_files)} total):\n\n"

        for i, render_path in enumerate(recent, 1):
            stat = render_path.stat()
            size_kb = stat.st_size / 1024
            from datetime import datetime
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")

            output += f"{i}. {render_path.name}\n"
            output += f"   Path: {render_path}\n"
            output += f"   Time: {mod_time}, Size: {size_kb:.1f} KB\n\n"

        output += f"\nTo analyze any render, use: analyze_image(image_path=\"{recent[0]}\")"

        return output

    except Exception as e:
        return f"Error listing renders: {e}"


@tool
def find_matching_angle(volume_path: str, reference_category: str, search_strategy: str = "coarse_to_fine") -> str:
    """Automatically find the camera angle that best matches a reference image from the knowledge graph.

    This tool tests multiple camera angles and uses vision AI to compare each render
    against the reference image, returning the best matching angle.

    Args:
        volume_path: Path to the volume file to render
        reference_category: Category of reference image in KG (e.g., "skull", "bone")
        search_strategy: "coarse_to_fine" (default, faster) or "grid" (exhaustive)

    Returns:
        Best matching camera angle with elevation and azimuth values
    """
    run_mgr = get_run_manager()
    kg_tool = get_kg_tool()

    try:
        # Get reference image from KG
        refs = kg_tool.kg.get_reference_renders(category=reference_category, quality="good")

        if not refs:
            return f"No reference images found for category: {reference_category}"

        # Use the first reference
        ref = refs[0]
        ref_image_path = kg_tool.kg.kg_path / ref['image_path']

        if not ref_image_path.exists():
            return f"Reference image not found at: {ref_image_path}"

        # Create output directory for this search
        output_dir = run_mgr.renders_dir / "angle_search" / f"{reference_category}_{run_mgr.timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get volume metadata
        metadata = None
        if volume_path.endswith('.raw'):
            # Try to infer from filename
            stem = Path(volume_path).stem
            parts = stem.split('_')
            if len(parts) >= 2:
                try:
                    dims_str = parts[1]  # e.g., "256x256x128"
                    dims = tuple(map(int, dims_str.split('x')))
                    dtype = parts[2] if len(parts) > 2 else 'uint8'
                    metadata = {'dimensions': dims, 'dtype': dtype}
                except:
                    pass

        # Log the operation
        run_mgr.log_operation(
            operation="find_matching_angle",
            details={
                "volume_path": volume_path,
                "reference_category": reference_category,
                "reference_image": str(ref_image_path),
                "search_strategy": search_strategy,
                "output_dir": str(output_dir)
            }
        )

        # Find best angle
        matcher = AngleMatcher()
        result = matcher.find_best_angle(
            volume_path=volume_path,
            reference_image_path=str(ref_image_path),
            metadata=metadata,
            output_dir=str(output_dir),
            search_strategy=search_strategy
        )

        best = result['best_match']

        output = f"✅ Found best matching camera angle!\n\n"
        output += f"Reference: {reference_category}\n"
        output += f"Reference image: {ref_image_path.name}\n\n"
        output += f"Best Match:\n"
        output += f"  Elevation: {best['elevation']}° (0=eye level, 90=top view, -90=bottom view)\n"
        output += f"  Azimuth: {best['azimuth']}° (0=front, 90=right, 180=back, 270=left)\n"
        output += f"  Match score: {best['match_score']}/10\n"
        output += f"  Render: {best['render_path']}\n\n"
        output += f"Tested {result['angles_tested']} angles using '{search_strategy}' strategy\n"
        output += f"Summary: {output_dir}/angle_search_summary.json\n\n"
        output += f"To render at this angle, use:\n"
        output += f'render_volume(volume_path="{volume_path}", '
        output += f'prompt="Render bones with camera elevation {best["elevation"]} degrees '
        output += f'and azimuth {best["azimuth"]} degrees")'

        return output

    except Exception as e:
        return f"Error finding matching angle: {e}"


@tool
def kg_get_reference_images(category: str = None, quality: str = "good") -> str:
    """Get reference images from the knowledge graph with full file paths.

    Use this to find example images showing how to render specific structures.
    The returned paths can be analyzed with the analyze_image tool.

    Args:
        category: Filter by category (e.g., "skull", "bone", "vascular"). If None, returns all.
        quality: Filter by quality - "good" or "bad" (default: "good")

    Returns:
        List of reference images with full paths
    """
    kg = get_kg_tool()

    try:
        refs = kg.get_reference_renders(category=category, quality=quality)

        if not refs:
            available = list(kg.kg.graph.get("reference_renders", {}).keys())
            return f"""No reference images found for category '{category}'.

Available categories: {', '.join(set(r.split('_')[0] for r in available)) if available else 'None'}

Try: kg_get_reference_images(category="skull") or kg_get_reference_images() to see all.
"""

        output = f"Reference Images"
        if category:
            output += f" (category: {category})"
        output += f":\n\n"

        for ref in refs:
            exists = "✓ Available" if ref.get('image_exists') else "✗ Missing"
            full_path = ref.get('image_full_path', 'N/A')

            output += f"""• {ref['id']}
  Status: {exists}
  Path: {full_path}
  Category: {ref.get('category', 'N/A')}
  Tags: {', '.join(ref.get('tags', []))}
  Description: {ref.get('description', 'N/A')}

"""

        output += f"\nTo analyze a reference image, use: analyze_image(image_path=\"{refs[0].get('image_full_path')}\")"

        return output

    except Exception as e:
        return f"Error querying reference images: {e}"


def create_interactive_agent(run_mgr: RunManager):
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

    tools = [
        analyze_volume,
        render_volume,
        smart_render_volume,
        list_files,
        list_recent_renders,
        analyze_image,
        find_matching_angle,
        kg_query_convention,
        kg_get_recommendations,
        kg_get_dataset_info,
        kg_get_statistics,
        kg_get_reference_images
    ]
    llm_with_tools = llm.bind_tools(tools)

    # Define agent logic
    def should_continue(state: MessagesState):
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


def run_interactive_cli(
    run_name: Optional[str] = None,
    base_dir: str = "runs",
    log_level: str = "INFO",
    kg_path: str = ".kg"
):
    """Run the interactive CLI."""
    global _run_manager, _kg_tool

    # Convert log level string to logging constant
    log_level_int = getattr(logging, log_level.upper())

    # Initialize run manager
    _run_manager = RunManager(
        run_name=run_name,
        base_dir=base_dir,
        log_level=log_level_int
    )

    run_mgr = _run_manager

    # Initialize KG tool with run manager
    _kg_tool = KGTool(kg_path=kg_path, run_manager=run_mgr)
    run_mgr.logger.info(f"Knowledge Graph initialized at: {kg_path}")

    print("\nChat with an AI that can use volume analysis and rendering tools!")
    print("\nAvailable commands:")
    print("  - Ask to analyze a dataset")
    print("  - Ask to render visualizations")
    print("  - Ask about reference images in the knowledge graph")
    print("  - Ask what files are available")
    print("  - Type 'quit' or 'exit' to leave")
    print("\nExamples:")
    print('  "Analyze the foot dataset at 3d_datasets/foot_256x256_uint8.raw"')
    print('  "Show me the bones"')
    print('  "Show me the skull reference images from the KG"')
    print('  "Render the skull to match the front view reference"')
    print('  "What files do I have?"')
    print("="*70)
    print()

    print(f"💡 TIP: The knowledge graph at {kg_path} contains:")
    print(f"   - Conventions for rendering (colors, opacity)")
    print(f"   - Reference images showing how structures should look")
    print(f"   - Rules and preferences for rendering")
    print(f"   Ask me to query the KG or show reference images!")
    print("="*70)
    print()

    # Create agent
    try:
        agent = create_interactive_agent(run_mgr)
        run_mgr.logger.info("Interactive agent initialized")
    except Exception as e:
        run_mgr.logger.error(f"Error initializing agent: {e}", exc_info=True)
        print(f"Error initializing agent: {e}")
        print("Make sure your .config file is set up correctly.")
        run_mgr.finalize(status="failed")
        return

    # Conversation state
    state = {"messages": []}

    # Add system message
    system_msg = AIMessage(content="""I'm your OntoVis assistant! I can help you analyze and visualize 3D volume datasets.

I have access to these tools:

Volume Analysis & Rendering:
- **analyze_volume**: Analyze intensity distributions, get statistics, identify features
- **render_volume**: Create 3D visualizations based on your prompts
- **smart_render_volume**: Create optimally-framed 3D visualizations using iterative vision AI feedback
- **analyze_image**: Analyze images using vision AI (describe contents, extract information)
- **list_files**: Show available data files

Knowledge Graph (rendering conventions & dataset knowledge):
- **kg_query_convention**: Get rendering conventions for anatomical structures (colors, intensities)
- **kg_get_recommendations**: Get comprehensive recommendations before rendering
- **kg_get_dataset_info**: Get information about a specific dataset
- **kg_get_statistics**: See what's in the knowledge graph

Just tell me what you want to do in natural language, and I'll use the right tools!""")

    print("Assistant: " + system_msg.content)
    print()

    # Track interaction count
    interaction_count = 0

    # Main loop
    try:
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye!")
                    break

                interaction_count += 1
                run_mgr.log_operation("user_interaction", {
                    "interaction": interaction_count,
                    "input": user_input
                })

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
                run_mgr.logger.error(f"Error during interaction: {e}", exc_info=True)
                print(f"\nError: {e}")
                print()

    finally:
        # Finalize run
        summary = {
            "interactions": interaction_count
        }
        run_mgr.finalize(status="completed", summary=summary)

        print(f"\nRun complete: {run_mgr.run_id}")
        print(f"Logs and artifacts saved to: {run_mgr.run_dir.absolute()}")
