"""Runnable demo of the hierarchical, capability-based visualization orchestrator.

Uses VisualizationOrchestrator (orchestrator.py) -- the facade that wires the existing
camera agent (camera_reasoning.visual_rollout_agent), the isovalue agent
(visualization_orchestrator.specialists.isovalue_adapter), and the planner/executor/
verifier stack behind one object -- to run a handful of example instructions against the
skull dataset, printing the interpreted goal, generated task graph, selected agent per
task, task results, final state summary, and final verification result for each one.

Requires a working OPENAI_API_KEY (see .env.example) since the planner, both
specialists, and the verifier all make real LLM calls, and a working VTK install for
rendering. Run with:

    .venv/bin/python -m visualization_orchestrator.demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization_orchestrator import VisualizationOrchestrator
from visualization_orchestrator.specialists import load_simple_reference_bank

DATASET_PATH = "data/skull_256x256x256_uint8.raw"
DIMENSIONS = (256, 256, 256)
INITIAL_ISOVALUE = 40

# reference_views_medical/skull/ holds real reference photographs of a physical skull
# (not renders of this specific VTK dataset) -- pass it explicitly so the camera
# specialist's blind-loop candidate selection is reference-grounded instead of falling
# back to a directional sweep. A photo bank is fine here: Pass 1 judges viewpoint
# resemblance by eye, not exact pixel/rendering match (see
# camera_reasoning/blind_visual_rollout_agent.py's "Reference grounding" docstring).
# Not loaded automatically by VisualizationOrchestrator.
REFERENCE_BANK_DESCRIPTIONS_PATH = "reference_views_medical/skull/reference_views_simple.json"

DEMO_INSTRUCTIONS = [
    "Show the inside of the head from behind.",
    "Show the skull from the left.",
    "Reduce the surface noise.",
    "Rotate to the right.",
    "Move to the posterior view first, then find the best isovalue for showing the inside.",
]


def main():
    reference_image_paths, node_descriptions = load_simple_reference_bank(REFERENCE_BANK_DESCRIPTIONS_PATH)
    orchestrator = VisualizationOrchestrator(
        dataset_path=DATASET_PATH,
        dimensions=DIMENSIONS,
        isovalue=INITIAL_ISOVALUE,
        output_dir="output/orchestrator_demo",
        camera_reference_image_paths=reference_image_paths,
        camera_node_descriptions=node_descriptions,
    )

    for instruction in DEMO_INSTRUCTIONS:
        print("\n" + "=" * 100)
        print(f"USER INSTRUCTION: {instruction}")
        print("=" * 100)

        result = orchestrator.run(instruction)

        for record in result.task_records:
            print(f"  - {record.task_id}: agent={record.agent_id} status={record.status} reason={record.reason}")
        if result.final_verification:
            print(
                f"[Demo] Final verification: success={result.final_verification.success} "
                f"confidence={result.final_verification.confidence} "
                f"diagnosis={result.final_verification.diagnosis}"
            )
        print(f"[Demo] Final state: isovalue={result.final_state.isovalue} "
              f"camera_position={result.final_state.camera_position} "
              f"rendered_image_path={result.final_state.rendered_image_path}")


if __name__ == "__main__":
    main()
