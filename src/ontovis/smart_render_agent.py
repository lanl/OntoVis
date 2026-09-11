"""Smart volume rendering agent that iteratively adjusts rendering to fill the image properly."""

import base64
from pathlib import Path
from typing import Annotated, TypedDict, Optional, Dict, Any
import json
import tempfile
import shutil

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .config import Config
from .render_agent_v2 import VolumeRenderAgent
from .vision_agent import VisionAgent


class SmartRenderState(TypedDict):
    """State for the smart rendering agent."""
    messages: Annotated[list, add_messages]
    volume_path: str
    metadata: Dict[str, Any]
    user_prompt: str
    iteration: int
    max_iterations: int
    rendered_image_path: str
    rendered_image_base64: str
    vision_feedback: str
    adjustment_needed: bool
    zoom_instruction: str
    all_iterations: list  # Store all intermediate renders
    save_intermediates: bool
    output_dir: Optional[str]
    final_result: Optional[Dict[str, Any]]


class SmartVolumeRenderAgent:
    """Agent that iteratively renders volumes with vision-guided adjustments for optimal framing."""

    def __init__(self, config_path=None, max_iterations=5, save_intermediates=False, output_dir=None):
        """Initialize the smart rendering agent.

        Args:
            config_path: Path to the config file (optional)
            max_iterations: Maximum number of rendering iterations (default: 5)
            save_intermediates: Save all intermediate renders (default: False)
            output_dir: Directory to save intermediates (default: current directory)
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
        self.vision_agent = VisionAgent(config_path)
        self.max_iterations = max_iterations
        self.save_intermediates = save_intermediates
        self.output_dir = output_dir if output_dir else "."

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(SmartRenderState)

        workflow.add_node("initial_render", self._initial_render)
        workflow.add_node("analyze_framing", self._analyze_framing)
        workflow.add_node("adjust_and_rerender", self._adjust_and_rerender)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "initial_render")
        workflow.add_edge("initial_render", "analyze_framing")

        # Conditional edge based on whether adjustment is needed
        workflow.add_conditional_edges(
            "analyze_framing",
            self._should_adjust,
            {
                "adjust": "adjust_and_rerender",
                "finalize": "finalize"
            }
        )

        workflow.add_edge("adjust_and_rerender", "analyze_framing")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _should_adjust(self, state: SmartRenderState) -> str:
        """Decide whether to adjust rendering or finalize.

        Args:
            state: Current state

        Returns:
            str: "adjust" or "finalize"
        """
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        adjustment_needed = state.get("adjustment_needed", False)

        if adjustment_needed and iteration < max_iterations:
            return "adjust"
        return "finalize"

    def _initial_render(self, state: SmartRenderState) -> Dict[str, Any]:
        """Perform initial volume rendering.

        Args:
            state: Current state

        Returns:
            dict: Updated state with initial render
        """
        volume_path = state.get("volume_path")
        user_prompt = state.get("user_prompt")
        metadata = state.get("metadata", {})
        save_intermediates = state.get("save_intermediates", False)
        output_dir = state.get("output_dir", ".")

        # Create file for render
        if save_intermediates:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            temp_path = str(Path(output_dir) / "iteration_01.png")
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_path = temp_file.name
            temp_file.close()

        # Perform initial render
        print(f"[Iteration 1] Performing initial render...")
        result = self.render_agent.render(
            volume_path=volume_path,
            prompt=user_prompt,
            metadata=metadata,
            save_image=temp_path
        )

        iteration_info = {
            "iteration": 1,
            "path": temp_path,
            "prompt": user_prompt,
            "adjustment": "initial",
            "instructions": result.get("instructions", {})
        }

        # Save iteration info immediately if save_intermediates
        if save_intermediates:
            info_path = str(Path(temp_path).with_suffix('.json'))
            import json
            with open(info_path, 'w') as f:
                json.dump(iteration_info, f, indent=2)
            print(f"[Iteration 1] Saved to: {temp_path}")
            print(f"[Iteration 1] Analysis saved to: {info_path}")

        return {
            "iteration": 1,
            "rendered_image_path": temp_path,
            "rendered_image_base64": result.get("rendered_image_base64"),
            "zoom_instruction": "",
            "all_iterations": [iteration_info],
            "messages": []
        }

    def _analyze_framing(self, state: SmartRenderState) -> Dict[str, Any]:
        """Analyze the rendered image to check if volume fills the frame properly.

        Args:
            state: Current state

        Returns:
            dict: Updated state with vision feedback and adjustment decision
        """
        rendered_image_path = state.get("rendered_image_path")
        iteration = state.get("iteration", 1)

        print(f"[Iteration {iteration}] Analyzing framing with vision AI...")

        # Use vision agent to analyze the framing
        vision_prompt = """Analyze this 3D volume rendering and determine if the object properly fills the image frame.

Please assess:
1. **Frame occupancy**: Does the rendered volume occupy a good portion of the image (ideally 60-85% of the frame)?
2. **Cropping**: Is any part of the volume cut off or cropped at the edges?
3. **Empty space**: Is there too much empty/background space around the volume?
4. **Size assessment**: Is the volume too small (needs to zoom in) or too large (needs to zoom out)?

Respond with a JSON object in this exact format:
{
  "is_well_framed": true/false,
  "occupancy_percent": estimated percentage (0-100),
  "issue": "none" | "too_small" | "too_large" | "cropped" | "too_much_empty_space",
  "suggested_action": "zoom_in_significantly" | "zoom_in_slightly" | "zoom_out_slightly" | "zoom_out_significantly" | "good",
  "explanation": "Brief explanation of the assessment"
}

Be precise and objective in your assessment. Respond ONLY with the JSON object.
"""

        analysis = self.vision_agent.analyze(
            image_path=rendered_image_path,
            prompt=vision_prompt
        )

        # Save vision analysis immediately if save_intermediates
        save_intermediates = state.get("save_intermediates", False)
        if save_intermediates:
            import json
            analysis_path = str(Path(rendered_image_path).parent / f"iteration_{iteration:02d}_vision_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump({"raw_analysis": analysis}, f, indent=2)
            print(f"[Iteration {iteration}] Vision analysis saved to: {analysis_path}")

        # Parse the JSON response
        try:
            # Clean up markdown code blocks if present
            analysis_clean = analysis.strip()
            if analysis_clean.startswith('```'):
                lines = analysis_clean.split('\n')
                # Remove first and last lines (```)
                analysis_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else analysis_clean
                # Remove 'json' tag if present in first line
                if lines[0].strip() in ['```json', '```JSON']:
                    analysis_clean = '\n'.join(lines[1:-1])

            feedback = json.loads(analysis_clean)
            is_well_framed = feedback.get("is_well_framed", False)
            suggested_action = feedback.get("suggested_action", "good")
            explanation = feedback.get("explanation", "")

            print(f"[Iteration {iteration}] Vision feedback: {explanation}")
            print(f"[Iteration {iteration}] Well framed: {is_well_framed}, Suggested: {suggested_action}")

            # Convert suggested action to zoom instruction
            zoom_map = {
                "zoom_in_significantly": "Move the camera much closer to fill more of the frame",
                "zoom_in_slightly": "Move the camera slightly closer to better fill the frame",
                "zoom_out_slightly": "Move the camera slightly farther to avoid cropping",
                "zoom_out_significantly": "Move the camera much farther back to show the full volume",
                "good": ""
            }
            zoom_instruction = zoom_map.get(suggested_action, "")

            return {
                "vision_feedback": analysis,
                "adjustment_needed": not is_well_framed,
                "zoom_instruction": zoom_instruction
            }

        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse vision feedback as JSON: {e}")
            print(f"Response: {analysis}")
            # If we can't parse, assume it's good enough
            return {
                "vision_feedback": analysis,
                "adjustment_needed": False,
                "zoom_instruction": ""
            }

    def _adjust_and_rerender(self, state: SmartRenderState) -> Dict[str, Any]:
        """Adjust camera and re-render based on vision feedback.

        Args:
            state: Current state

        Returns:
            dict: Updated state with new render
        """
        volume_path = state.get("volume_path")
        user_prompt = state.get("user_prompt")
        metadata = state.get("metadata", {})
        iteration = state.get("iteration", 1)
        zoom_instruction = state.get("zoom_instruction", "")
        save_intermediates = state.get("save_intermediates", False)
        output_dir = state.get("output_dir", ".")
        all_iterations = state.get("all_iterations", [])

        new_iteration = iteration + 1
        print(f"[Iteration {new_iteration}] Adjusting framing and re-rendering...")

        # Modify the prompt to include the zoom instruction
        adjusted_prompt = f"{user_prompt}. CAMERA ADJUSTMENT: {zoom_instruction}"

        # Create file for render
        if save_intermediates:
            temp_path = str(Path(output_dir) / f"iteration_{new_iteration:02d}.png")
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            temp_path = temp_file.name
            temp_file.close()

        # Re-render with adjusted prompt
        result = self.render_agent.render(
            volume_path=volume_path,
            prompt=adjusted_prompt,
            metadata=metadata,
            save_image=temp_path
        )

        iteration_info = {
            "iteration": new_iteration,
            "path": temp_path,
            "prompt": adjusted_prompt,
            "adjustment": zoom_instruction,
            "instructions": result.get("instructions", {})
        }
        all_iterations.append(iteration_info)

        # Save iteration info immediately if save_intermediates
        if save_intermediates:
            import json
            info_path = str(Path(temp_path).with_suffix('.json'))
            with open(info_path, 'w') as f:
                json.dump(iteration_info, f, indent=2)
            print(f"[Iteration {new_iteration}] Saved to: {temp_path}")
            print(f"[Iteration {new_iteration}] Analysis saved to: {info_path}")

        return {
            "iteration": new_iteration,
            "rendered_image_path": temp_path,
            "rendered_image_base64": result.get("rendered_image_base64"),
            "all_iterations": all_iterations
        }

    def _finalize(self, state: SmartRenderState) -> Dict[str, Any]:
        """Finalize the rendering process.

        Args:
            state: Current state

        Returns:
            dict: Final state with result
        """
        iteration = state.get("iteration", 1)
        rendered_image_path = state.get("rendered_image_path")
        vision_feedback = state.get("vision_feedback", "")
        all_iterations = state.get("all_iterations", [])
        save_intermediates = state.get("save_intermediates", False)

        print(f"[Complete] Finalized after {iteration} iteration(s)")
        print(f"[Complete] Final image: {rendered_image_path}")

        if save_intermediates and all_iterations:
            print(f"[Complete] All {len(all_iterations)} iteration(s) saved:")
            for iter_info in all_iterations:
                print(f"  - Iteration {iter_info['iteration']}: {iter_info['path']}")

            # Save complete summary
            import json
            from datetime import datetime
            output_dir = state.get("output_dir", ".")
            summary_path = str(Path(output_dir) / "SUMMARY.json")
            summary = {
                "total_iterations": iteration,
                "final_image": rendered_image_path,
                "all_iterations": all_iterations,
                "volume_path": state.get("volume_path"),
                "user_prompt": state.get("user_prompt"),
                "final_vision_feedback": vision_feedback,
                "timestamp": datetime.now().isoformat()
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"[Complete] Summary saved to: {summary_path}")

            # Save human-readable README
            readme_path = str(Path(output_dir) / "README.md")
            with open(readme_path, 'w') as f:
                f.write(f"# Smart Render Session\n\n")
                f.write(f"**Volume**: `{state.get('volume_path')}`\n")
                f.write(f"**Prompt**: {state.get('user_prompt')}\n")
                f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"## Results\n\n")
                f.write(f"- **Total Iterations**: {iteration}\n")
                f.write(f"- **Final Image**: `{Path(rendered_image_path).name}`\n\n")
                f.write(f"## Iteration Details\n\n")
                for iter_info in all_iterations:
                    f.write(f"### Iteration {iter_info['iteration']}\n")
                    f.write(f"- **Image**: `{Path(iter_info['path']).name}`\n")
                    f.write(f"- **JSON**: `{Path(iter_info['path']).stem}.json`\n")
                    f.write(f"- **Adjustment**: {iter_info['adjustment']}\n")
                    if iter_info.get('instructions'):
                        inst = iter_info['instructions']
                        f.write(f"- **Rendering Details**:\n")
                        f.write(f"  - Method: {inst.get('method', 'N/A')}\n")
                        f.write(f"  - Threshold: {inst.get('threshold', 'N/A')}\n")
                        f.write(f"  - Color: {inst.get('color', 'N/A')}\n")
                        f.write(f"  - Opacity: {inst.get('opacity', 'N/A')}\n")
                    f.write(f"\n")
                f.write(f"\n## Files\n\n")
                f.write(f"- `SUMMARY.json` - Complete session data\n")
                f.write(f"- `README.md` - This file\n")
                f.write(f"- `iteration_XX.png` - Rendered images\n")
                f.write(f"- `iteration_XX.json` - Render parameters for each iteration\n")
                f.write(f"- `iteration_XX_vision_analysis.json` - Vision AI feedback\n")
            print(f"[Complete] README saved to: {readme_path}")

        final_result = {
            "rendered_image_path": rendered_image_path,
            "rendered_image_base64": state.get("rendered_image_base64"),
            "iterations": iteration,
            "vision_feedback": vision_feedback,
            "all_iterations": all_iterations if save_intermediates else []
        }

        return {
            "final_result": final_result
        }

    def render(
        self,
        volume_path: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_image: Optional[str] = None
    ) -> Dict[str, Any]:
        """Render a volume with iterative vision-guided framing adjustments.

        Args:
            volume_path: Path to volume data file (.raw, .npy, .dat)
            prompt: Natural language description of desired rendering
            metadata: Dictionary with dataset metadata (dimensions, dtype for .raw files)
            save_image: Optional path to save final rendered image

        Returns:
            dict: Results containing:
                - rendered_image_path: Path to final rendered image
                - rendered_image_base64: Base64-encoded PNG of final render
                - iterations: Number of iterations performed
                - vision_feedback: Final vision analysis feedback
                - all_iterations: List of all iteration info (if save_intermediates=True)
        """
        if metadata is None:
            metadata = {}

        initial_state = {
            "messages": [],
            "volume_path": volume_path,
            "metadata": metadata,
            "user_prompt": prompt,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "rendered_image_path": "",
            "rendered_image_base64": "",
            "vision_feedback": "",
            "adjustment_needed": True,
            "zoom_instruction": "",
            "all_iterations": [],
            "save_intermediates": self.save_intermediates,
            "output_dir": self.output_dir,
            "final_result": None
        }

        result = self.graph.invoke(initial_state)
        final_result = result.get("final_result", {})

        # Copy to final save location if requested
        if save_image and final_result.get("rendered_image_path"):
            temp_path = final_result["rendered_image_path"]
            if Path(temp_path).exists():
                shutil.copy(temp_path, save_image)
                final_result["rendered_image_path"] = save_image

        return final_result
