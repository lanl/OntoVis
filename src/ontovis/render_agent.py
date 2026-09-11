"""Volume rendering agent for creating 3D visualizations from volume datasets."""

import io
import base64
from pathlib import Path
from typing import Annotated, TypedDict, Optional, Dict, Any, List
import json
from datetime import datetime

import numpy as np
import pyvista as pv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .config import Config
from .multimodal_kg import MultimodalKnowledgeGraph


def spherical_to_cartesian(azimuth_deg, elevation_deg, radius):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        azimuth_deg: Azimuth angle in degrees (0-360)
                     0° = +X axis, 90° = +Y axis
        elevation_deg: Elevation angle in degrees (-90 to 90)
                      0° = XY plane, 90° = +Z axis
        radius: Distance from origin

    Returns:
        tuple: (x, y, z) Cartesian coordinates
    """
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    return x, y, z


class RenderState(TypedDict):
    """State for the volume rendering agent."""
    messages: Annotated[list, add_messages]
    volume_path: str
    metadata: Dict[str, Any]
    user_prompt: str
    volume_data: Optional[np.ndarray]
    histogram_summary: Optional[Dict[str, Any]]
    render_params: Optional[Dict[str, Any]]
    rendered_image: Optional[str]
    output_dir: Optional[str]  # Directory for storing intermediate outputs


class VolumeRenderAgent:
    """Agent for rendering 3D volumes with AI-guided parameter selection."""

    def __init__(self, config_path=None, kg_path=".kg"):
        """Initialize the volume rendering agent.

        Args:
            config_path: Path to the config file (optional)
            kg_path: Path to knowledge graph directory (default: .kg)
        """
        self.config = Config(config_path)
        llm_config = self.config.get_llm_config()

        self.llm = ChatAnthropic(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            max_tokens=4096
        )

        # Initialize knowledge graph
        self.kg = MultimodalKnowledgeGraph(kg_path)

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(RenderState)

        workflow.add_node("load_volume", self._load_volume)
        workflow.add_node("analyze_histogram", self._analyze_histogram)
        workflow.add_node("determine_params", self._determine_params)
        workflow.add_node("render_volume", self._render_volume)

        workflow.add_edge(START, "load_volume")
        workflow.add_edge("load_volume", "analyze_histogram")
        workflow.add_edge("analyze_histogram", "determine_params")
        workflow.add_edge("determine_params", "render_volume")
        workflow.add_edge("render_volume", END)

        return workflow.compile()

    def _load_volume(self, state: RenderState) -> Dict[str, Any]:
        """Load volume data from file.

        Args:
            state: Current state

        Returns:
            dict: Updated state with volume_data
        """
        volume_path = state.get("volume_path")
        metadata = state.get("metadata", {})

        path = Path(volume_path)
        if not path.exists():
            raise FileNotFoundError(f"Volume file not found: {volume_path}")

        ext = path.suffix.lower()

        if ext == '.raw':
            dims = metadata.get('dimensions')
            dtype = metadata.get('dtype', 'uint8')

            if not dims:
                raise ValueError("For .raw files, metadata must include 'dimensions'")

            np_dtype = getattr(np, dtype)
            volume_data = np.fromfile(volume_path, dtype=np_dtype)
            volume_data = volume_data.reshape(dims)

        elif ext == '.npy':
            volume_data = np.load(volume_path)

        elif ext == '.dat':
            import struct
            with open(volume_path, 'rb') as f:
                dims = struct.unpack('III', f.read(12))
                dtype = metadata.get('dtype', 'uint8')
                np_dtype = getattr(np, dtype)
                volume_data = np.fromfile(f, dtype=np_dtype)
                volume_data = volume_data.reshape(dims)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        return {
            "volume_data": volume_data,
            "messages": state.get("messages", [])
        }

    def _analyze_histogram(self, state: RenderState) -> Dict[str, Any]:
        """Analyze histogram to inform rendering parameters.

        Args:
            state: Current state

        Returns:
            dict: Updated state with histogram_summary
        """
        volume_data = state.get("volume_data")
        output_dir = state.get("output_dir")

        if volume_data is None:
            raise ValueError("No volume data available")

        # Compute histogram and statistics
        hist, bin_edges = np.histogram(volume_data.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks
        from scipy import signal
        peaks, _ = signal.find_peaks(hist, prominence=hist.max() * 0.05)
        peak_values = bin_centers[peaks].tolist()

        # Compute percentiles for dynamic range
        percentiles = {
            'p01': float(np.percentile(volume_data, 1)),
            'p05': float(np.percentile(volume_data, 5)),
            'p25': float(np.percentile(volume_data, 25)),
            'p50': float(np.percentile(volume_data, 50)),
            'p75': float(np.percentile(volume_data, 75)),
            'p95': float(np.percentile(volume_data, 95)),
            'p99': float(np.percentile(volume_data, 99))
        }

        histogram_summary = {
            'min': float(volume_data.min()),
            'max': float(volume_data.max()),
            'mean': float(volume_data.mean()),
            'median': float(np.median(volume_data)),
            'std': float(volume_data.std()),
            'peaks': peak_values,
            'percentiles': percentiles,
            'shape': volume_data.shape,
            'dtype': str(volume_data.dtype)
        }

        # Save histogram plot if output directory is specified
        if output_dir:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], color='steelblue', alpha=0.7)
            plt.xlabel('Intensity')
            plt.ylabel('Frequency')
            plt.title('Volume Intensity Histogram')

            # Mark peaks
            if peaks.size > 0:
                plt.plot(bin_centers[peaks], hist[peaks], 'ro', markersize=8, label='Detected Peaks')
                plt.legend()

            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            histogram_path = Path(output_dir) / '01_histogram.png'
            plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  📊 Saved histogram: {histogram_path}")

        return {
            "histogram_summary": histogram_summary
        }

    def _determine_params(self, state: RenderState) -> Dict[str, Any]:
        """Use LLM to determine rendering parameters based on user prompt.

        Args:
            state: Current state

        Returns:
            dict: Updated state with render_params
        """
        histogram_summary = state.get("histogram_summary")
        user_prompt = state.get("user_prompt")
        metadata = state.get("metadata", {})

        # Load rendering strategies from KG documentation
        kg_strategies = self.kg.get_documentation("rendering_strategies")
        if not kg_strategies:
            kg_strategies = "No additional rendering strategies available."

        # Construct prompt for LLM
        analysis_prompt = f"""You are a scientific visualization expert. Based on the user's request, volume statistics, and knowledge graph strategies, determine optimal rendering parameters.

=== KNOWLEDGE GRAPH RENDERING STRATEGIES ===
{kg_strategies}

=== VOLUME DATA ANALYSIS ===

Dataset Information:
- Shape: {histogram_summary['shape']}
- Value range: {histogram_summary['min']:.2f} to {histogram_summary['max']:.2f}
- Mean: {histogram_summary['mean']:.2f}, Median: {histogram_summary['median']:.2f}
- Detected peaks: {histogram_summary.get('peaks', [])}
- Percentiles: {histogram_summary['percentiles']}

Additional metadata: {metadata}

User's Request: "{user_prompt}"

Please provide rendering parameters as a JSON object with the following structure:
{{
  "camera_azimuth": 0-360,      // Horizontal rotation (0=+X, 90=+Y, 180=-X, 270=-Y)
  "camera_elevation": -90 to 90, // Vertical angle (0=side, 90=top, -90=bottom)
  "camera_distance": 1.0-3.0,    // Distance multiplier (1.0=close, 2.0=standard, 3.0=far)
  "camera_roll": -180 to 180,    // Camera roll around viewing axis (0=default, ±90=rotated)
  "opacity_mapping": [
    {{"value": intensity, "opacity": 0.0-1.0}},
    // IMPORTANT: For sparse datasets (median near 0), use aggressive opacity:
    //   - 0 to p75: opacity 0.0 (completely transparent)
    //   - p75 to p95: opacity 0.0 to 0.3 (gradual fade-in)
    //   - p95 to max: opacity 0.5 to 1.0 (fully visible)
    // At least 5 points needed
  ],
  "color_mapping": [
    {{"value": intensity, "color": [r, g, b]}},
    // At least 3-5 points defining the color transfer function (RGB 0-255)
  ],
  "background_color": [r, g, b],  // RGB 0-255
  "lighting": {{
    "ambient": 0.0-1.0,
    "diffuse": 0.0-1.0,
    "specular": 0.0-1.0
  }},
  "volume_scale": [sx, sy, sz],  // Scaling factors for each axis
  "explanation": "Brief explanation of why these parameters were chosen"
}}

Consider:
1. What features the user wants to see (bones, tissue, specific structures)
2. Whether to make background transparent or visible
3. Appropriate color schemes (medical: grayscale/bone colors; scientific: heat maps)
4. Opacity curves that highlight the requested features
5. Camera angle for best visualization:
   - For "front face view": azimuth=270, elevation=0, roll=-90 (person upright)
   - For "side profile": azimuth=0, elevation=0, roll=-90
   - For "top view": azimuth=0, elevation=90, roll=0
   - For diagonal 3D view: azimuth=45, elevation=30, roll=0
6. Use roll to correct volume orientation (e.g., make lying person appear upright)

Respond ONLY with the JSON object, no additional text.
"""

        # Get rendering parameters from LLM
        message = HumanMessage(content=analysis_prompt)
        response = self.llm.invoke([message])

        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.content.strip()
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            render_params = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Response: {response.content}")
            # Fallback to default parameters
            render_params = self._get_default_params(histogram_summary)

        # Save parameters to file if output directory is specified
        output_dir = state.get("output_dir")
        if output_dir:
            params_path = Path(output_dir) / '02_render_params.json'
            with open(params_path, 'w') as f:
                json.dump(render_params, f, indent=2)
            print(f"  📋 Saved parameters: {params_path}")

        return {
            "render_params": render_params,
            "messages": [message, response]
        }

    def _get_default_params(self, histogram_summary: Dict) -> Dict:
        """Generate default rendering parameters.

        Args:
            histogram_summary: Histogram statistics

        Returns:
            dict: Default rendering parameters
        """
        vmin = histogram_summary['percentiles']['p01']
        vmax = histogram_summary['percentiles']['p99']
        mid = (vmin + vmax) / 2

        return {
            "camera_azimuth": 45.0,
            "camera_elevation": 30.0,
            "camera_distance": 2.0,
            "camera_roll": 0.0,
            "opacity_mapping": [
                {"value": vmin, "opacity": 0.0},
                {"value": mid * 0.5, "opacity": 0.1},
                {"value": mid, "opacity": 0.5},
                {"value": vmax * 0.9, "opacity": 0.8},
                {"value": vmax, "opacity": 1.0}
            ],
            "color_mapping": [
                {"value": vmin, "color": [0, 0, 0]},
                {"value": mid, "color": [128, 128, 128]},
                {"value": vmax, "color": [255, 255, 255]}
            ],
            "background_color": [255, 255, 255],
            "lighting": {
                "ambient": 0.3,
                "diffuse": 0.6,
                "specular": 0.3
            },
            "volume_scale": [1.0, 1.0, 1.0],
            "explanation": "Default parameters: simple grayscale rendering"
        }

    def _render_volume(self, state: RenderState) -> Dict[str, Any]:
        """Render the volume using PyVista.

        Args:
            state: Current state

        Returns:
            dict: Updated state with rendered_image
        """
        volume_data = state.get("volume_data")
        render_params = state.get("render_params")

        if volume_data is None or render_params is None:
            raise ValueError("Missing volume data or render parameters")

        # Create PyVista volume
        grid = pv.ImageData()
        grid.dimensions = np.array(volume_data.shape) + 1

        # Apply volume scaling
        scale = render_params.get("volume_scale", [1.0, 1.0, 1.0])
        grid.spacing = scale

        grid.cell_data["values"] = volume_data.flatten(order="F")

        # Create plotter
        plotter = pv.Plotter(off_screen=True, window_size=[800, 800])

        # Set background color
        bg_color = render_params.get("background_color", [255, 255, 255])
        plotter.background_color = [c / 255.0 for c in bg_color]

        # Build color and opacity transfer functions
        color_map = render_params.get("color_mapping", [])
        opacity_map = render_params.get("opacity_mapping", [])

        color_values = [p["value"] for p in color_map]
        colors_rgb = [[c / 255.0 for c in p["color"]] for p in color_map]

        # Create matplotlib colormap from RGB values
        from matplotlib.colors import LinearSegmentedColormap
        if len(colors_rgb) > 1:
            # Normalize color positions to 0-1 range
            vmin = min(color_values)
            vmax = max(color_values)
            if vmax > vmin:
                norm_positions = [(v - vmin) / (vmax - vmin) for v in color_values]
            else:
                norm_positions = [0.0, 1.0]
                colors_rgb = [colors_rgb[0], colors_rgb[0]]

            # Create colormap
            cmap = LinearSegmentedColormap.from_list(
                'custom',
                list(zip(norm_positions, colors_rgb))
            )
        else:
            cmap = "gray"

        # Create custom opacity transfer function
        # PyVista expects either a string or a 1D array of opacity values
        if opacity_map and len(opacity_map) > 1:
            opacity_x = np.array([p["value"] for p in opacity_map])
            opacity_y = np.array([p["opacity"] for p in opacity_map])

            # Create interpolated opacity for the data range
            data_min, data_max = volume_data.min(), volume_data.max()
            data_range = np.linspace(data_min, data_max, 256)
            opacity_array = np.interp(data_range, opacity_x, opacity_y)
        else:
            # Fallback: linear ramp favoring high values
            opacity_array = np.linspace(0, 1, 256) ** 3  # Cubic ramp

        # Add volume with custom opacity
        volume_actor = plotter.add_volume(
            grid,
            scalars="values",
            opacity=opacity_array,
            cmap=cmap,
            shade=True,
            show_scalar_bar=False
        )

        # Set lighting
        lighting = render_params.get("lighting", {"ambient": 0.3, "diffuse": 0.6, "specular": 0.3})
        volume_actor.prop.ambient = lighting.get("ambient", 0.3)
        volume_actor.prop.diffuse = lighting.get("diffuse", 0.6)
        volume_actor.prop.specular = lighting.get("specular", 0.3)

        # Set camera position using spherical coordinates
        azimuth = render_params.get("camera_azimuth", 45.0)
        elevation = render_params.get("camera_elevation", 30.0)
        distance = render_params.get("camera_distance", 2.0)
        roll = render_params.get("camera_roll", 0.0)

        center = np.array(volume_data.shape) * np.array(scale) / 2

        # Calculate camera position
        max_dim = max(volume_data.shape)
        scaled_distance = distance * max_dim

        cam_x, cam_y, cam_z = spherical_to_cartesian(azimuth, elevation, scaled_distance)
        camera_position = center + np.array([cam_x, cam_y, cam_z])

        # Calculate proper up vector with roll
        view_dir = center - camera_position
        view_dir = view_dir / np.linalg.norm(view_dir)

        # Default up vector (Z-axis)
        base_up = np.array([0.0, 0.0, 1.0])

        # If view_dir is parallel to Z, use Y as base_up
        if abs(np.dot(view_dir, base_up)) > 0.99:
            base_up = np.array([0.0, 1.0, 0.0])

        # Calculate right vector
        right = np.cross(base_up, view_dir)
        right = right / np.linalg.norm(right)

        # Calculate corrected up vector
        up = np.cross(view_dir, right)
        up = up / np.linalg.norm(up)

        # Apply roll rotation using Rodrigues' formula
        if roll != 0:
            roll_rad = np.radians(roll)
            cos_roll = np.cos(roll_rad)
            sin_roll = np.sin(roll_rad)

            k_dot_v = np.dot(view_dir, up)
            k_cross_v = np.cross(view_dir, up)

            up = up * cos_roll + k_cross_v * sin_roll + view_dir * k_dot_v * (1 - cos_roll)
            up = up / np.linalg.norm(up)

        plotter.camera_position = [
            tuple(camera_position),
            tuple(center),
            tuple(up)
        ]

        # Render
        plotter.show(auto_close=False)

        # Capture screenshot
        img_bytes = plotter.screenshot(return_img=True)
        plotter.close()

        # Save rendered image to output directory if specified
        output_dir = state.get("output_dir")
        if output_dir:
            from PIL import Image
            img = Image.fromarray(img_bytes)
            render_path = Path(output_dir) / '03_final_render.png'
            img.save(render_path)
            print(f"  🎨 Saved render: {render_path}")

        # Convert to base64
        from PIL import Image
        img = Image.fromarray(img_bytes)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        rendered_image = base64.b64encode(buf.read()).decode('utf-8')

        return {
            "rendered_image": rendered_image
        }

    def render(
        self,
        volume_path: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_image: Optional[str] = None,
        output_base_dir: str = "renders",
        use_angle_matching: bool = False,
        reference_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Render a volume dataset based on a natural language prompt.

        Args:
            volume_path: Path to volume data file (.raw, .npy, .dat)
            prompt: Natural language description of desired rendering
                   Examples:
                   - "Show the bones in white, make everything else transparent"
                   - "Highlight soft tissue in red, show bones in grayscale"
                   - "Create a heat map visualization showing density"
                   - "Show only high-density structures"
            metadata: Dictionary with dataset metadata (dimensions, dtype for .raw files)
            save_image: Optional path to save rendered image
            output_base_dir: Base directory for storing intermediate outputs (default: "renders")
            use_angle_matching: If True, use vision-guided angle matching instead of LLM
            reference_id: Reference image ID for angle matching (e.g., "a_skull_front_view")

        Returns:
            dict: Results containing:
                - rendered_image_base64: Base64-encoded PNG
                - render_params: Parameters used for rendering
                - explanation: Why these parameters were chosen
                - output_dir: Directory where intermediate outputs are stored
                - angle_match_result: (if use_angle_matching=True) Angle matching details
        """
        if metadata is None:
            metadata = {}

        # Create unique output directory for this interaction
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        volume_name = Path(volume_path).stem
        output_dir = Path(output_base_dir) / f"{timestamp}_{volume_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the prompt
        prompt_path = output_dir / '00_prompt.txt'
        with open(prompt_path, 'w') as f:
            f.write(f"User Prompt:\n{prompt}\n\n")
            f.write(f"Volume: {volume_path}\n")
            f.write(f"Metadata: {json.dumps(metadata, indent=2)}\n")

        print(f"\n📁 Output directory: {output_dir}")
        print(f"  📝 Saved prompt: {prompt_path}")

        initial_state = {
            "messages": [],
            "volume_path": volume_path,
            "metadata": metadata,
            "user_prompt": prompt,
            "volume_data": None,
            "histogram_summary": None,
            "render_params": None,
            "rendered_image": None,
            "output_dir": str(output_dir)
        }

        result = self.graph.invoke(initial_state)

        # Save image if requested (backward compatibility)
        if save_image and result.get("rendered_image"):
            image_data = base64.b64decode(result["rendered_image"])
            with open(save_image, 'wb') as f:
                f.write(image_data)

        print(f"  ✅ Rendering complete!\n")

        return {
            "rendered_image_base64": result.get("rendered_image"),
            "render_params": result.get("render_params"),
            "explanation": result.get("render_params", {}).get("explanation", ""),
            "output_dir": str(output_dir)
        }
