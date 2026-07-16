"""
Run the visual-rollout camera-action-selection loop (camera_reasoning/visual_rollout_agent.py):
every iteration renders one candidate image per available action from the CURRENT
camera state, then asks the LLM to select the next action by comparing the actual
rendered outcomes against the target — not by reasoning about action names.

By default only a small ACTION_SUBSET is used to keep each API call small/cheap;
set ACTION_SUBSET = None to evaluate every action in VALID_ACTIONS (the full,
unrestricted behavior).

Run with: .venv/bin/python examples/run_visual_rollout_alignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from camera_reasoning import CameraReasoningSession
from camera_reasoning.visual_rollout_agent import run_visual_rollout_alignment_loop

ACTION_SUBSET = {
    "AZIMUTH_LEFT_COARSE", "AZIMUTH_RIGHT_COARSE",
    "ELEVATION_UP_COARSE", "ELEVATION_DOWN_COARSE",
    "ZOOM_IN", "ZOOM_OUT", "STOP",
}  # set to None to evaluate every action in VALID_ACTIONS

session = CameraReasoningSession(
    raw_path="data/foot_256x256x256_uint8.raw",
    dimensions=(256, 256, 256),
    scalar_type="uint8",
    isovalue=80,
    output_dir="output",
    target_description=(
        "Top view of the foot bones. Big toe on the right, little toe on the left, "
        "toes near the bottom, ankle/heel region near the top."
    ),
)
session.initialize()

result = run_visual_rollout_alignment_loop(
    session,
    max_iterations=10,
    action_subset=ACTION_SUBSET,
)

print("\nApplied actions:", result["applied_actions"])
