"""Volume dataset analysis agent for identifying features from intensity distributions."""

import io
import base64
from pathlib import Path
from typing import Annotated, TypedDict, Optional, Dict, Any
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .config import Config


class VolumeState(TypedDict):
    """State for the volume analysis agent."""
    messages: Annotated[list, add_messages]
    volume_path: str
    metadata: Dict[str, Any]
    user_description: str
    volume_data: Optional[np.ndarray]
    histogram_data: Optional[Dict[str, Any]]
    histogram_image: Optional[str]
    feature_analysis: str


class VolumeAnalysisAgent:
    """Agent for analyzing volume datasets and identifying features from intensity distributions."""

    def __init__(self, config_path=None):
        """Initialize the volume analysis agent.

        Args:
            config_path: Path to the config file (optional)
        """
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
        """Build the LangGraph workflow."""
        workflow = StateGraph(VolumeState)

        workflow.add_node("load_volume", self._load_volume)
        workflow.add_node("compute_histogram", self._compute_histogram)
        workflow.add_node("plot_histogram", self._plot_histogram)
        workflow.add_node("analyze_features", self._analyze_features)

        workflow.add_edge(START, "load_volume")
        workflow.add_edge("load_volume", "compute_histogram")
        workflow.add_edge("compute_histogram", "plot_histogram")
        workflow.add_edge("plot_histogram", "analyze_features")
        workflow.add_edge("analyze_features", END)

        return workflow.compile()

    def _load_volume(self, state: VolumeState) -> Dict[str, Any]:
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

        # Determine file format and load
        ext = path.suffix.lower()

        if ext == '.raw':
            # Raw binary format - need dimensions from metadata
            dims = metadata.get('dimensions')
            dtype = metadata.get('dtype', 'uint8')

            if not dims:
                raise ValueError("For .raw files, metadata must include 'dimensions' (e.g., [256, 256, 256])")

            # Convert dtype string to numpy dtype
            np_dtype = getattr(np, dtype)

            # Read raw binary data
            volume_data = np.fromfile(volume_path, dtype=np_dtype)
            volume_data = volume_data.reshape(dims)

        elif ext == '.npy':
            # NumPy binary format
            volume_data = np.load(volume_path)

        elif ext == '.dat':
            # Assume structured DAT format (dimensions + data)
            with open(volume_path, 'rb') as f:
                # Read dimensions (assume 3 uint32 values)
                dims = struct.unpack('III', f.read(12))
                # Read data
                dtype = metadata.get('dtype', 'uint8')
                np_dtype = getattr(np, dtype)
                volume_data = np.fromfile(f, dtype=np_dtype)
                volume_data = volume_data.reshape(dims)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .raw, .npy, .dat")

        return {
            "volume_data": volume_data,
            "messages": state.get("messages", [])
        }

    def _compute_histogram(self, state: VolumeState) -> Dict[str, Any]:
        """Compute histogram statistics from volume data.

        Args:
            state: Current state

        Returns:
            dict: Updated state with histogram_data
        """
        volume_data = state.get("volume_data")

        if volume_data is None:
            raise ValueError("No volume data available")

        # Compute histogram with automatic binning
        hist, bin_edges = np.histogram(volume_data.flatten(), bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute statistics
        stats = {
            'min': float(volume_data.min()),
            'max': float(volume_data.max()),
            'mean': float(volume_data.mean()),
            'median': float(np.median(volume_data)),
            'std': float(volume_data.std()),
            'shape': volume_data.shape,
            'dtype': str(volume_data.dtype),
            'total_voxels': int(volume_data.size)
        }

        # Find peaks in histogram (potential feature boundaries)
        from scipy import signal
        peaks, properties = signal.find_peaks(hist, prominence=hist.max() * 0.05)
        peak_values = bin_centers[peaks].tolist()

        histogram_data = {
            'hist': hist,
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'stats': stats,
            'peaks': peak_values,
            'peak_indices': peaks.tolist()
        }

        return {
            "histogram_data": histogram_data
        }

    def _plot_histogram(self, state: VolumeState) -> Dict[str, Any]:
        """Create histogram visualization following dataviz guidelines.

        Args:
            state: Current state

        Returns:
            dict: Updated state with histogram_image
        """
        histogram_data = state.get("histogram_data")
        metadata = state.get("metadata", {})

        if histogram_data is None:
            raise ValueError("No histogram data available")

        # Color palette following dataviz guidelines
        # Sequential palette for magnitude data (histogram bars)
        bar_color = '#0969da'  # Primary blue for main data
        peak_color = '#cf222e'  # Red for peaks/important features
        grid_color = '#d0d7de'  # Recessive grid
        text_color = '#1f2328'  # Primary text

        # Create figure with proper sizing
        fig = Figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_subplot(111)

        hist = histogram_data['hist']
        bin_edges = histogram_data['bin_edges']
        bin_centers = histogram_data['bin_centers']
        stats = histogram_data['stats']
        peaks = histogram_data.get('peaks', [])

        # Plot histogram bars (thin marks as per guidelines)
        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0],
               color=bar_color, alpha=0.7, edgecolor='none')

        # Mark peaks with vertical lines
        for peak in peaks:
            ax.axvline(peak, color=peak_color, linestyle='--',
                      linewidth=1.5, alpha=0.6)

        # Style the plot following dataviz guidelines
        ax.set_xlabel('Intensity Value', fontsize=11, color=text_color)
        ax.set_ylabel('Frequency (voxel count)', fontsize=11, color=text_color)

        # Title with dataset info
        dataset_name = metadata.get('name', 'Volume Dataset')
        ax.set_title(f'{dataset_name} - Intensity Distribution',
                    fontsize=13, fontweight='bold', color=text_color, pad=16)

        # Recessive grid (following guidelines)
        ax.grid(True, alpha=0.15, color=grid_color, linewidth=0.5)
        ax.set_axisbelow(True)

        # Clean up spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(grid_color)
            ax.spines[spine].set_linewidth(0.8)

        # Add statistics annotation in corner
        stats_text = (
            f"Min: {stats['min']:.2f}\n"
            f"Max: {stats['max']:.2f}\n"
            f"Mean: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"Std: {stats['std']:.2f}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                        edgecolor=grid_color, linewidth=0.8),
               color=text_color, family='monospace')

        # Tight layout
        fig.tight_layout()

        # Convert to base64 for LLM vision analysis
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return {
            "histogram_image": image_data
        }

    def _analyze_features(self, state: VolumeState) -> Dict[str, Any]:
        """Analyze histogram using LLM to identify features.

        Args:
            state: Current state

        Returns:
            dict: Updated state with feature_analysis
        """
        histogram_data = state.get("histogram_data")
        histogram_image = state.get("histogram_image")
        user_description = state.get("user_description", "")
        metadata = state.get("metadata", {})

        if histogram_data is None or histogram_image is None:
            raise ValueError("Missing histogram data or image")

        stats = histogram_data['stats']
        peaks = histogram_data.get('peaks', [])

        # Construct analysis prompt
        prompt = f"""You are analyzing a 3D volume dataset's intensity distribution histogram.

Dataset Information:
- User Description: {user_description}
- Dimensions: {stats['shape']}
- Data Type: {stats['dtype']}
- Total Voxels: {stats['total_voxels']:,}
- Intensity Range: {stats['min']:.2f} to {stats['max']:.2f}
- Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std Dev: {stats['std']:.2f}

"""

        if metadata:
            prompt += "\nAdditional Metadata:\n"
            for key, value in metadata.items():
                if key not in ['dimensions', 'dtype']:
                    prompt += f"- {key}: {value}\n"

        if peaks:
            prompt += f"\nDetected Histogram Peaks at intensities: {', '.join(f'{p:.2f}' for p in peaks)}\n"

        prompt += """
Based on the histogram visualization and the dataset information, please:

1. **Identify distinct intensity regions**: Analyze the distribution shape and identify distinct regions or clusters in the intensity histogram.

2. **Map intensities to features**: Based on the dataset description and typical patterns in volume data, suggest what physical features or materials each intensity range likely represents. Consider:
   - Background/air (typically low intensities)
   - Different tissue types, materials, or phases
   - Boundaries and interfaces
   - Noise characteristics

3. **Suggest intensity thresholds**: Recommend specific intensity values that could be used as thresholds to segment or isolate different features.

4. **Identify patterns**: Note any unusual patterns, bimodal distributions, long tails, or other characteristics that provide insights about the data.

5. **Recommend visualization parameters**: Suggest transfer function ranges or windowing parameters that would be effective for visualizing this dataset.

Provide your analysis in a structured format with clear sections.
"""

        # Create message with histogram image
        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": histogram_image
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        # Get analysis from LLM
        response = self.llm.invoke([message])
        analysis = response.content

        return {
            "messages": [message, response],
            "feature_analysis": analysis
        }

    def analyze_volume(
        self,
        volume_path: str,
        user_description: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_histogram: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a volume dataset and identify features.

        Args:
            volume_path: Path to volume data file (.raw, .npy, .dat)
            user_description: Text description of what the dataset contains
            metadata: Dictionary with dataset metadata:
                - dimensions: [x, y, z] for .raw files
                - dtype: numpy dtype string (e.g., 'uint8', 'float32')
                - name: dataset name for plot title
                - Any other relevant metadata
            save_histogram: Optional path to save histogram plot

        Returns:
            dict: Analysis results containing:
                - feature_analysis: LLM analysis of features
                - histogram_data: Statistical data
                - volume_stats: Basic volume statistics
        """
        if metadata is None:
            metadata = {}

        initial_state = {
            "messages": [],
            "volume_path": volume_path,
            "metadata": metadata,
            "user_description": user_description,
            "volume_data": None,
            "histogram_data": None,
            "histogram_image": None,
            "feature_analysis": ""
        }

        result = self.graph.invoke(initial_state)

        # Save histogram if requested
        if save_histogram and result.get("histogram_image"):
            image_data = base64.b64decode(result["histogram_image"])
            with open(save_histogram, 'wb') as f:
                f.write(image_data)

        return {
            "feature_analysis": result["feature_analysis"],
            "histogram_data": result.get("histogram_data"),
            "volume_stats": result.get("histogram_data", {}).get("stats", {}),
            "histogram_image_base64": result.get("histogram_image")
        }

    def save_annotated_histogram(
        self,
        histogram_data: Dict[str, Any],
        feature_annotations: Dict[float, str],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create and save an annotated histogram with feature labels.

        Args:
            histogram_data: Histogram data from analyze_volume
            feature_annotations: Dict mapping intensity values to feature labels
            output_path: Where to save the annotated plot
            metadata: Optional metadata for title
        """
        if metadata is None:
            metadata = {}

        # Color palette
        bar_color = '#0969da'
        annotation_color = '#cf222e'
        grid_color = '#d0d7de'
        text_color = '#1f2328'

        fig = Figure(figsize=(12, 6), facecolor='white')
        ax = fig.add_subplot(111)

        hist = histogram_data['hist']
        bin_edges = histogram_data['bin_edges']
        bin_centers = histogram_data['bin_centers']
        stats = histogram_data['stats']

        # Plot histogram
        ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0],
               color=bar_color, alpha=0.7, edgecolor='none')

        # Add feature annotations
        y_max = hist.max()
        for intensity, label in sorted(feature_annotations.items()):
            ax.axvline(intensity, color=annotation_color, linestyle='--',
                      linewidth=1.5, alpha=0.6)
            ax.text(intensity, y_max * 0.95, label,
                   rotation=90, verticalalignment='top',
                   fontsize=9, color=annotation_color, fontweight='bold')

        # Styling
        ax.set_xlabel('Intensity Value', fontsize=11, color=text_color)
        ax.set_ylabel('Frequency (voxel count)', fontsize=11, color=text_color)

        dataset_name = metadata.get('name', 'Volume Dataset')
        ax.set_title(f'{dataset_name} - Feature-Annotated Distribution',
                    fontsize=13, fontweight='bold', color=text_color, pad=16)

        ax.grid(True, alpha=0.15, color=grid_color, linewidth=0.5)
        ax.set_axisbelow(True)

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(grid_color)
            ax.spines[spine].set_linewidth(0.8)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
