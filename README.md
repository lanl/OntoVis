# Autonomous Volume Visualization Framework

A VTK visualization system driven by an LLM-orchestrated capability pipeline (camera
alignment, isovalue/transfer-function selection, orientation correction) behind a single
entry point, `VisualizationOrchestrator`.

## Quick Start

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` (see `.env.example`) -- the planner, every specialist, and the final
verifier all make real LLM calls.

Each demo notebook needs a raw volume under `data/` (already in the repo) and a
reference-view bank under `reference_views_medical/<object>/`, which the camera
specialist uses to ground its viewpoint judgments. Using the head example
(`visualization_orchestrator_demo_head.ipynb`, dataset
`data/vis_male_128x256x256_uint8.raw`, bank `reference_views_medical/skull/`):

`reference_views_medical/skull/reference_views_simple.json` already exists in this repo,
so you can skip straight to "Run the notebook" below. These are the steps that produced
it, and the steps to follow for a new object or a rebuilt bank:

1. Add seed photos and a `config.json` under `reference_views_medical/<object>/`.
   `reference_views_medical/skull/config.json` is a worked example: three seed photos
   (`skull_001.png` = right, `skull_002.png` = front, `skull_003.png` = left) and a list
   of in-plane rotation angles.
2. Generate the rotated images and the full landmark schema:
   ```bash
   .venv/bin/python examples/generate_medical_reference_views.py \
       --config reference_views_medical/skull/config.json
   ```
   This calls the LLM to diagnose each image, and writes `skull_00N_rot045.png` and so on
   next to each seed, plus `reference_views.json`.
3. Generate the flat label mapping the notebook actually loads:
   ```bash
   .venv/bin/python examples/generate_simple_reference_labels.py \
       --config reference_views_medical/skull/config.json
   ```
   This reuses the images from step 2, makes no LLM call, and writes
   `reference_views_simple.json`.

### Run the notebook

```bash
jupyter notebook notebooks/visualization_orchestrator_demo_head.ipynb   # data/vis_male_128x256x256_uint8.raw
jupyter notebook notebooks/visualization_orchestrator_demo_foot.ipynb    # data/foot_256x256x256_uint8.raw
```

## Workflow

1. `orchestrator = VisualizationOrchestrator(dataset_path=..., ...)` loads and renders the
   volume once.
2. `orchestrator.run("Show the skull in a lateral view, with the superior aspect at the top.")`
   -- any free-form instruction. The planner turns it into a small task graph, runs whichever
   specialist(s) it requires (camera, isovalue, orientation -- never a fixed pipeline), and
   verifies the result against the instruction.
3. Call `.run(...)` again with the next instruction, in any order, as many times as you like
   -- state carries forward between calls like a multi-turn conversation
   (`orchestrator.history` holds every past `ExecutionResult`).

See `TECHNICAL_DOCUMENTATION.md` for how planning, specialist dispatch, state ownership,
and the camera/isovalue/orientation agents actually work.

## Project Structure

```
camera_reasoning/
  __init__.py
  session.py                      # CameraReasoningSession -- VTK scene + LLM prompt/response plumbing
  camera_actions.py               # Action definitions and apply_action()
  camera_state.py                 # get/set/save/load camera state helpers
  volume_scene.py                 # VTK scene construction and screenshot saving
  prompt_writer.py                # LLM prompt generation
  action_parser.py                # ChatGPT response parsing
  chatgpt_client.py               # LLM call wrapper
  spatial_knowledge.py            # spatial-context helpers
  view_description_generator.py   # JSON-extraction helper reused by visual_rollout_agent
  visual_rollout_agent.py         # candidate-rollout camera alignment loop
  blind_visual_rollout_agent.py   # label-blind, three-pass candidate selection (used by CameraSpecialist)
  medical_reference_views.py      # reference-view generation engine for reference_views_medical/
  simple_reference_labels.py      # writes the flat {node_id: label} reference_views_simple.json
visualization_orchestrator/
  __init__.py
  orchestrator.py                 # VisualizationOrchestrator -- the single entry point
  planner.py / replanner.py       # LLM task-graph planning
  executor.py                     # deterministic scheduling/retries/state patches
  registry.py / capabilities.py   # capability -> specialist resolution
  state.py / models.py            # VisualizationState, VisualizationPlan
  specialists/                    # camera_adapter.py, isovalue_adapter.py, orientation_adapter.py
notebooks/
  visualization_orchestrator_demo_head.ipynb
  visualization_orchestrator_demo_foot.ipynb
examples/
  generate_medical_reference_views.py   # full landmark/relations schema (reference_views.json)
  generate_simple_reference_labels.py   # flat {node_id: label} schema (reference_views_simple.json)
reference_views_medical/
  skull/, foot/                   # seed photographs + generated rotations/landmark JSON
output/                           # generated at runtime, gitignored
data/
  vis_male_128x256x256_uint8.raw
  foot_256x256x256_uint8.raw
```

## Tests

```bash
.venv/bin/pip install pytest  # not in requirements.txt; only needed to run the suite
.venv/bin/python -m pytest tests/
```

`tests/test_orchestrator_*.py` use fake specialists/planner/verifier
(`tests/orchestrator_fakes.py`) -- no real LLM or VTK calls -- and cover plan validation,
registry selection, state-patch ownership, and executor scheduling/replanning behavior (see
each file's docstring for the exact list).

`tests/test_isovalue_band_selection.py` is different: it runs the real VTK pipeline against
the real `data/vis_male_128x256x256_uint8.raw` dataset (fixed-window computation, per-window
preview rendering, applying the derived transfer function) and mocks only the LLM call
(`ask_chatgpt`) -- it's the one place the window math and the actual renderer are exercised
together rather than through a fake.
