"""Final goal verification: runs once after every planned task succeeds, checking the
complete original user instruction rather than assuming local per-task success implies
global success (see prompts/verifier_prompt.py).

Deliberately judges from the rendered image alone (plus the instruction/success
criteria) -- no textual state summary or task-record history is included. An earlier
version handed the verifier the raw numeric camera state (position/focal_point/
view_up) alongside the image, which invited exactly the wrong kind of reasoning: it
inferred "upside-down" from view_up's sign (which has no universal correct value --
it depends on this dataset's own coordinate convention and the current azimuth/
elevation) and overrode a correct visual judgment from the orientation specialist with
an incorrect numeric one. This is a pure image-vs-goal comparison now, matching what
the verifier is actually meant to check.
"""
from typing import Optional

from camera_reasoning.chatgpt_client import ask_chatgpt

from .json_utils import extract_json_object
from .models import FinalVerificationResult
from .prompts.verifier_prompt import build_verifier_prompt
from .state import VisualizationState


class FinalVerifier:
    def __init__(self, model: Optional[str] = None):
        self.model = model

    def verify(
        self,
        user_instruction: str,
        state: VisualizationState,
        final_success_criteria: list,
    ) -> FinalVerificationResult:
        prompt = build_verifier_prompt(user_instruction, final_success_criteria)
        raw_response = ask_chatgpt(
            prompt=prompt, screenshot_path=state.rendered_image_path, model=self.model
        )

        parsed = extract_json_object(raw_response)
        if parsed is None:
            return FinalVerificationResult(
                success=False,
                confidence=0.0,
                satisfied_criteria=[],
                unsatisfied_criteria=list(final_success_criteria),
                diagnosis="Verifier response was not valid JSON; treating as unverified.",
                suggested_capabilities=[],
            )

        return FinalVerificationResult(
            success=bool(parsed.get("success", False)),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            satisfied_criteria=list(parsed.get("satisfied_criteria", []) or []),
            unsatisfied_criteria=list(parsed.get("unsatisfied_criteria", []) or []),
            diagnosis=str(parsed.get("diagnosis", "") or ""),
            suggested_capabilities=list(parsed.get("suggested_capabilities", []) or []),
        )
