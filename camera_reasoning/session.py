import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .action_parser import extract_action
from .camera_actions import VALID_ACTIONS, apply_action
from .camera_state import (
    camera_distance,
    get_camera_state,
    load_camera_state_json,
    save_camera_state_json,
    set_camera_state,
)
from .chatgpt_client import ask_chatgpt
from .prompt_writer import append_attached_image_context, write_llm_prompt
from .spatial_knowledge import (
    build_camera_view_hint,
    dump_spatial_knowledge_json,
    extract_diagnosis_sections,
    extract_structured_fields,
    load_simple_spatial_knowledge,
)
from .volume_scene import build_isosurface_pipeline, load_raw_volume, save_screenshot

MIN_CAMERA_DISTANCE = 1e-3


class CameraReasoningSession:
    def __init__(
        self,
        raw_path: str,
        dimensions: tuple,
        scalar_type: str = "uint8",
        isovalue: float = 80,
        output_dir: str = "output",
        target_description: str = "No target description provided.",
        target_image_path: Optional[str] = None,
        simple_spatial_knowledge_path: Optional[str] = None,
    ):
        self.raw_path = raw_path
        self.dimensions = dimensions
        self.scalar_type = scalar_type
        self.isovalue = isovalue
        self.output_dir = Path(output_dir)
        self.target_description = target_description
        self.target_image_path = target_image_path

        self._actor = None
        self._renderer = None
        self._render_window = None
        self._image_data = None

        self._step = 0
        self._action_history: List[Dict] = []
        self._camera_state_stack: List[Dict] = []

        # Optional lightweight spatial-knowledge spec (see camera_reasoning/spatial_knowledge.py).
        # Fully backward compatible: if no path is given, or loading fails, self._spatial_context
        # stays None and the prompt/response format is unchanged from before this feature existed.
        self._spatial_data: Optional[dict] = None
        self._spatial_context: Optional[str] = None
        if simple_spatial_knowledge_path:
            self._spatial_data = load_simple_spatial_knowledge(simple_spatial_knowledge_path)
            if self._spatial_data:
                self._spatial_context = dump_spatial_knowledge_json(self._spatial_data)
                print("[spatial_knowledge] Spatial context for this session (full JSON dump):")
                print(self._spatial_context)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self):
        """Load data, build VTK scene, and reset the camera."""
        self._make_output_dirs()
        self._image_data = load_raw_volume(self.raw_path, self.dimensions, self.scalar_type)
        self._actor, self._renderer, self._render_window = build_isosurface_pipeline(
            self._image_data, self.isovalue
        )
        self._renderer.ResetCamera()
        print(f"Scene initialized. Isovalue={self.isovalue}, dims={self.dimensions}.")

    def render_and_save(self) -> str:
        """Render the current scene and save screenshots. Returns the latest screenshot path."""
        self._require_initialized()
        self._render_window.Render()
        return self._save_screenshot(action_name=None)

    def write_llm_prompt(
        self,
        reference_items: Optional[List[Tuple[str, str, str]]] = None,
        candidate_items: Optional[List[Tuple[str, str, str]]] = None,
    ) -> str:
        """Write the LLM prompt file and return its text content.

        ``reference_items`` contains stable reference-view examples used to identify and
        compare viewpoints. ``candidate_items`` normally contains COARSE one-step renders
        generated from the exact current camera state. Candidate evidence is integrated
        directly into action-by-action evaluation. A COARSE render also grounds the
        MEDIUM/FINE actions from the same movement family, with reduced predicted magnitude.

        Both use ``(label, image_path, description)`` triples, but the generated prompt
        assigns them different semantic roles so the model does not confuse a reference
        view with an action outcome.
        """
        self._require_initialized()
        camera_state = get_camera_state(self._renderer.GetActiveCamera())
        camera_view_hint = build_camera_view_hint(self._spatial_data, camera_state)
        return write_llm_prompt(
            output_dir=str(self.output_dir),
            camera_state=camera_state,
            action_history=self._action_history,
            target_description=self.target_description,
            screenshot_path=str(self.output_dir / "screenshots" / "latest.png"),
            target_image_path=self.target_image_path,
            spatial_context=self._spatial_context,
            camera_view_hint=camera_view_hint,
            reference_view_items=self._prompt_item_metadata(reference_items),
            candidate_action_items=self._prompt_item_metadata(candidate_items),
        )

    def ask_chatgpt(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        reference_items: Optional[List[Tuple[str, str, str]]] = None,
        candidate_items: Optional[List[Tuple[str, str, str]]] = None,
    ) -> str:
        """Send the prompt and all image inputs to the OpenAI API.

        Parameters
        ----------
        reference_items:
            Stable reference views, as ``(label, image_path, description)`` triples.
            These images ground view recognition and current/target comparison. They are
            not interpreted as results of camera actions.
        candidate_items:
            Counterfactual one-step action renders, using the same triple format. Use the
            exact COARSE camera action name as each label. The model evaluates that render
            in the COARSE action row and reuses its direction/effect only for MEDIUM/FINE
            actions from the same movement family, with a smaller predicted magnitude.

        The lower-level chat client currently exposes one ``reference_items`` attachment
        channel, so this method concatenates the two groups internally in a deterministic
        order: reference views first, candidate renders second. Their roles remain separate
        because the prompt includes two labeled inventories.
        """
        self._require_initialized()
        reference_view_metadata = self._prompt_item_metadata(reference_items)
        candidate_action_metadata = self._prompt_item_metadata(candidate_items)

        if prompt is None:
            prompt = self.write_llm_prompt(
                reference_items=reference_items,
                candidate_items=candidate_items,
            )
        else:
            # Preserve custom prompts while defining both image groups and integrating
            # candidate-render evidence into the matching action evaluations.
            prompt = append_attached_image_context(
                prompt,
                reference_view_items=reference_view_metadata,
                candidate_action_items=candidate_action_metadata,
            )
            prompt_path = self.output_dir / "llm_prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt)

        attached_items = self._merge_image_items(reference_items, candidate_items)
        return ask_chatgpt(
            prompt=prompt,
            screenshot_path=str(self.output_dir / "screenshots" / "latest.png"),
            target_image_path=self.target_image_path,
            reference_items=attached_items or None,
            model=model,
        )

    def ask_chatgpt_and_process(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        reference_items: Optional[List[Tuple[str, str, str]]] = None,
        candidate_items: Optional[List[Tuple[str, str, str]]] = None,
    ) -> str:
        """Ask ChatGPT, apply the selected action, and return that action name."""
        response = self.ask_chatgpt(
            prompt=prompt,
            model=model,
            reference_items=reference_items,
            candidate_items=candidate_items,
        )
        print(response)
        return self.process_chatgpt_response(response)

    def process_chatgpt_response(self, response: str) -> str:
        """Parse a pasted ChatGPT response, apply one action, and advance the loop.

        When spatial knowledge is enabled, the model answers free-form (no forced JSON
        schema) but the prompt requires it to cover: what it sees, an inferred camera
        position, and then the action — logged here for visibility, not gated on.
        """
        self._require_initialized()

        if self._spatial_context:
            self._log_diagnosis_sections(response)

        action = extract_action(response)
        print(f"Extracted action: {action}")

        if action == "STOP":
            print("STOP received — alignment marked as complete.")
            self._append_history(action)
            self._save_action_history()
            return action

        if action == "UNDO_LAST":
            self._do_undo()
        else:
            self._apply_and_advance(action)
        return action

    def reset_camera(self):
        """Hard-reset camera to fit the scene (destroys manual alignment)."""
        self._require_initialized()
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        print("Camera hard-reset to fit scene.")

    def load_camera_state(self, path: str):
        """Restore camera from a saved JSON file."""
        self._require_initialized()
        state = load_camera_state_json(path)
        set_camera_state(self._renderer.GetActiveCamera(), state)
        self._renderer.ResetCameraClippingRange()
        print(f"Camera state loaded from {path}.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_item_metadata(
        items: Optional[List[Tuple[str, str, str]]],
    ) -> List[Tuple[str, str]]:
        """Strip image paths while preserving label/description prompt metadata."""
        if not items:
            return []
        return [(str(label), str(description)) for label, _, description in items]

    @staticmethod
    def _merge_image_items(
        reference_items: Optional[List[Tuple[str, str, str]]],
        candidate_items: Optional[List[Tuple[str, str, str]]],
    ) -> List[Tuple[str, str, str]]:
        """Merge attachment groups in the same order declared by the prompt inventories."""
        return list(reference_items or []) + list(candidate_items or [])

    def _log_diagnosis_sections(self, response: str):
        """Best-effort log of the free-form "Visual observation" / "Camera position
        inference" sections and the current_view_inclination field, for visibility
        only — never blocks action application.
        """
        sections = extract_diagnosis_sections(response)
        fields = extract_structured_fields(response)

        if not sections and "current_view_inclination" not in fields:
            print(
                "[spatial_knowledge] No 'Visual observation' / 'Camera position inference' "
                "sections found in response (model may not have followed the requested format)."
            )
            return

        print("[spatial_knowledge] Diagnosis sections:")
        if "visual_observation" in sections:
            print(f"  Visual observation: {sections['visual_observation']}")
        if "camera_position_inference" in sections:
            print(f"  Camera position inference: {sections['camera_position_inference']}")
        if "current_view_inclination" in fields:
            print(f"  current_view_inclination: {fields['current_view_inclination']}")

    def _require_initialized(self):
        if self._renderer is None:
            raise RuntimeError("Call session.initialize() first.")

    def _make_output_dirs(self):
        for subdir in ("screenshots", "camera_states"):
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _apply_and_advance(self, action: str):
        camera = self._renderer.GetActiveCamera()
        prev_state = get_camera_state(camera)
        self._camera_state_stack.append(prev_state)

        apply_action(action, camera, self._renderer)

        new_state = get_camera_state(camera)
        if camera_distance(new_state) < MIN_CAMERA_DISTANCE:
            print("WARNING: camera too close to focal point — reverting.")
            set_camera_state(camera, prev_state)
            self._camera_state_stack.pop()
            return

        self._step += 1
        self._append_history(action)

        self._render_window.Render()
        self._save_screenshot(action_name=action)
        self._save_camera_state(action_name=action)
        self._save_action_history()
        self.write_llm_prompt()
        print(f"Step {self._step}: {action} applied.")

    def _do_undo(self):
        if not self._camera_state_stack:
            print("Nothing to undo.")
            return
        camera = self._renderer.GetActiveCamera()
        prev_state = self._camera_state_stack.pop()
        set_camera_state(camera, prev_state)
        self._renderer.ResetCameraClippingRange()

        self._step += 1
        self._append_history("UNDO_LAST")
        self._render_window.Render()
        self._save_screenshot(action_name="UNDO_LAST")
        self._save_camera_state(action_name="UNDO_LAST")
        self._save_action_history()
        self.write_llm_prompt()
        print(f"Step {self._step}: UNDO_LAST applied.")

    def _save_screenshot(self, action_name: Optional[str]) -> str:
        latest = str(self.output_dir / "screenshots" / "latest.png")
        if action_name is None:
            step_name = f"step_{self._step:03d}.png"
        else:
            step_name = f"step_{self._step:03d}_{action_name}.png"
        step_path = str(self.output_dir / "screenshots" / step_name)

        save_screenshot(self._render_window, latest)
        shutil.copy(latest, step_path)
        return latest

    def _save_camera_state(self, action_name: Optional[str]):
        camera = self._renderer.GetActiveCamera()
        state = get_camera_state(camera)
        latest = self.output_dir / "camera_states" / "latest_camera.json"
        save_camera_state_json(state, latest)
        if action_name:
            step_path = (
                self.output_dir / "camera_states" / f"step_{self._step:03d}_{action_name}.json"
            )
            save_camera_state_json(state, step_path)

    def _append_history(self, action: str):
        self._action_history.append({"step": self._step, "action": action})

    def _save_action_history(self):
        path = self.output_dir / "action_history.json"
        with open(path, "w") as f:
            json.dump(self._action_history, f, indent=2)