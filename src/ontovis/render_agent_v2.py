"""Simplified generic volume rendering agent - path + prompt → visualization."""

import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any
import json

import numpy as np
import pyvista as pv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from .config import Config


class SimpleRenderState(TypedDict):
    """Simplified state for generic rendering."""
    messages: Annotated[list, add_messages]
    volume_path: str
    user_prompt: str
    metadata: Dict[str, Any]
    volume_data: Optional[np.ndarray]
    render_instructions: Optional[Dict[str, Any]]
    rendered_image: Optional[str]


class VolumeRenderAgent:
    """Generic volume rendering agent controlled by natural language."""

    def __init__(self, config_path=None):
        """Initialize the rendering agent."""
        self.config = Config(config_path)
        llm_config = self.config.get_llm_config()

        self.llm = ChatAnthropic(
            model=llm_config['model'],
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
            max_tokens=4096
        )

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the workflow."""
        workflow = StateGraph(SimpleRenderState)

        workflow.add_node("load_volume", self._load_volume)
        workflow.add_node("determine_rendering", self._determine_rendering)
        workflow.add_node("render", self._render)

        workflow.add_edge(START, "load_volume")
        workflow.add_edge("load_volume", "determine_rendering")
        workflow.add_edge("determine_rendering", "render")
        workflow.add_edge("render", END)

        return workflow.compile()

    def _load_volume(self, state: SimpleRenderState) -> Dict[str, Any]:
        """Load volume data."""
        volume_path = state["volume_path"]
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
            raise ValueError(f"Unsupported format: {ext}")

        return {"volume_data": volume_data}

    def _determine_rendering(self, state: SimpleRenderState) -> Dict[str, Any]:
        """Use AI to determine how to render based on prompt."""
        volume_data = state["volume_data"]
        user_prompt = state["user_prompt"]

        # Analyze volume
        stats = {
            'min': float(volume_data.min()),
            'max': float(volume_data.max()),
            'mean': float(volume_data.mean()),
            'median': float(np.median(volume_data)),
            'std': float(volume_data.std()),
            'p01': float(np.percentile(volume_data, 1)),
            'p25': float(np.percentile(volume_data, 25)),
            'p75': float(np.percentile(volume_data, 75)),
            'p95': float(np.percentile(volume_data, 95)),
            'p99': float(np.percentile(volume_data, 99)),
            'shape': volume_data.shape
        }

        # Ask AI to interpret the prompt and determine rendering
        prompt = f"""You are a scientific visualization expert. Based on the user's prompt and volume statistics, determine the rendering approach.

Volume Statistics:
- Shape: {stats['shape']}
- Value range: {stats['min']:.1f} to {stats['max']:.1f}
- Mean: {stats['mean']:.1f}, Median: {stats['median']:.1f}
- Percentiles: p01={stats['p01']:.1f}, p25={stats['p25']:.1f}, p75={stats['p75']:.1f}, p95={stats['p95']:.1f}, p99={stats['p99']:.1f}

User's Prompt: "{user_prompt}"

Provide rendering instructions as JSON:
{{
  "method": "isosurface",  // Use isosurface for reliability
  "threshold": float,      // Intensity value for surface extraction
  "color": [r, g, b],     // RGB 0-255
  "opacity": 0.0-1.0,     // Surface opacity
  "camera_distance": float,  // Camera distance multiplier (1.0=close, 2.0=medium, 5.0+=far, 10.0+=very far)
  "camera_elevation": float,  // Camera elevation angle in degrees (0=eye level, 90=top view, -90=bottom view)
  "camera_azimuth": float,    // Camera azimuth angle in degrees (0=front, 90=right side, 180=back, 270=left side)
  "background": [r, g, b],     // RGB 0-255
  "explanation": "Why these parameters based on the prompt and data"
}}

Guidelines:
- If prompt mentions "bones", "high-density", "dense": use threshold near p95-p99
- If prompt mentions "tissue", "soft", "medium": use threshold near p50-p75
- If prompt mentions "all structures", "everything": use threshold near p25
- Choose colors that match the prompt (white for bones, pink for tissue, etc.)
- Camera distance: 1.5-2.5=normal view, 3-5=overview, 6-10=far view, 10+=bird's eye
- If user says "zoom out", "far away", "distant": use distance 5.0 or higher
- Camera angles: Extract elevation and azimuth from prompt (e.g., "elevation 10 degrees" → 10, "azimuth 45 degrees" → 45)
  - Front view: azimuth 0, elevation 0
  - Side view: azimuth 90 or 270
  - Top view: elevation 90
  - If no angles specified: use elevation 0, azimuth 0

Respond with ONLY the JSON object.
"""

        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])

        # Parse JSON
        try:
            response_text = response.content.strip()
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            instructions = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response: {e}")
            print(f"Response: {response.content}")
            # Fallback
            instructions = {
                "method": "isosurface",
                "threshold": stats['p95'],
                "color": [255, 255, 255],
                "opacity": 1.0,
                "camera_distance": 2.0,
                "camera_elevation": 0,
                "camera_azimuth": 0,
                "background": [0, 0, 0],
                "explanation": "Using default parameters"
            }

        return {
            "render_instructions": instructions,
            "messages": [message, response]
        }

    def _render(self, state: SimpleRenderState) -> Dict[str, Any]:
        """Render using isosurface extraction (more reliable than volume rendering)."""
        volume_data = state["volume_data"]
        instructions = state["render_instructions"]

        # Create PyVista volume - use point_data for contouring
        grid = pv.ImageData()
        grid.dimensions = volume_data.shape
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = volume_data.flatten(order="F")

        # Extract isosurface at threshold
        threshold = instructions["threshold"]

        try:
            surface = grid.contour([threshold])

            # Setup plotter
            plotter = pv.Plotter(off_screen=True, window_size=[800, 800])

            # Background
            bg = instructions["background"]
            plotter.background_color = [c / 255.0 for c in bg]

            # Add surface
            color_rgb = instructions["color"]
            color = [c / 255.0 for c in color_rgb]
            opacity = instructions["opacity"]

            plotter.add_mesh(
                surface,
                color=color,
                opacity=opacity,
                smooth_shading=True,
                show_edges=False
            )

            # Camera - use spherical coordinates (elevation, azimuth)
            center = np.array(volume_data.shape) / 2
            distance = instructions["camera_distance"]

            # Get angles (default to front view if not specified)
            elevation_deg = instructions.get("camera_elevation", 0)
            azimuth_deg = instructions.get("camera_azimuth", 0)

            # Convert to radians
            elevation = np.radians(elevation_deg)
            azimuth = np.radians(azimuth_deg)

            # Calculate camera position using spherical coordinates
            # Elevation: 0 = eye level, 90 = top view, -90 = bottom view
            # Azimuth: 0 = front, 90 = right, 180 = back, 270 = left
            radius = np.max(volume_data.shape) * distance

            cam_x = center[0] + radius * np.cos(elevation) * np.sin(azimuth)
            cam_y = center[1] + radius * np.cos(elevation) * np.cos(azimuth)
            cam_z = center[2] + radius * np.sin(elevation)

            plotter.camera_position = [
                (cam_x, cam_y, cam_z),  # Camera position
                tuple(center),           # Focal point (look at center)
                (0, 0, 1)               # View up vector
            ]

            # Render
            plotter.show(auto_close=False)
            img_bytes = plotter.screenshot(return_img=True)
            plotter.close()

            # Convert to base64
            from PIL import Image
            img = Image.fromarray(img_bytes)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            rendered_image = base64.b64encode(buf.read()).decode('utf-8')

            return {"rendered_image": rendered_image}

        except Exception as e:
            print(f"Rendering failed: {e}")
            # Return empty image
            return {"rendered_image": None}

    def render(
        self,
        volume_path: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """Render volume based on natural language prompt.

        Args:
            volume_path: Path to volume file
            prompt: Natural language description of what to render
                   Examples:
                   - "Show the bones"
                   - "Visualize high-density structures"
                   - "Show soft tissue at threshold 80"
                   - "Display everything above intensity 100"
            metadata: Optional metadata (dimensions, dtype for .raw files)
            save_image: Optional output path

        Returns:
            dict: Results with rendered_image_base64, instructions, explanation
        """
        if metadata is None:
            metadata = {}

        initial_state = {
            "messages": [],
            "volume_path": volume_path,
            "user_prompt": prompt,
            "metadata": metadata,
            "volume_data": None,
            "render_instructions": None,
            "rendered_image": None
        }

        result = self.graph.invoke(initial_state)

        # Save if requested
        if save_image and result.get("rendered_image"):
            # Ensure parent directory exists (important for parallel renders)
            save_path = Path(save_image)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            image_data = base64.b64decode(result["rendered_image"])
            with open(save_path, 'wb') as f:
                f.write(image_data)

        return {
            "rendered_image_base64": result.get("rendered_image"),
            "instructions": result.get("render_instructions"),
            "explanation": result.get("render_instructions", {}).get("explanation", "")
        }
