# Camera Reasoning App

A notebook-first, manual LLM-guided VTK camera alignment system.

## Quick Start

```bash
pip install -r requirements.txt
```

Place your raw volume at `data/foot_256x256x256_uint8.raw`, then open the notebook:

```bash
jupyter notebook notebooks/manual_chatgpt_loop.ipynb
```

Or run the script equivalent:

```bash
python examples/manual_chatgpt_loop_example.py
```

## Workflow

1. Run the **Initialize** cell (or script). This loads the volume, renders it, saves a screenshot to `output/screenshots/latest.png`, and writes the LLM prompt to `output/llm_prompt.txt`.
2. Copy the prompt text and the screenshot and send them to ChatGPT.
3. Paste ChatGPT's full response into the **Process Response** cell and run it. The notebook will:
   - Extract the chosen action.
   - Apply it to the VTK camera.
   - Save a new screenshot (e.g. `step_001_ELEVATION_UP_MEDIUM.png`).
   - Save the camera state JSON.
   - Update `action_history.json`.
   - Write the next LLM prompt.
4. Repeat until ChatGPT returns `STOP`.

## Project Structure

```
camera_reasoning_app/
  requirements.txt
  camera_reasoning/
    __init__.py
    session.py          # CameraReasoningSession — main API
    camera_actions.py   # Action definitions and apply_action()
    camera_state.py     # get/set/save/load camera state helpers
    volume_scene.py     # VTK scene construction and screenshot saving
    prompt_writer.py    # LLM prompt generation
    action_parser.py    # ChatGPT response parsing
  notebooks/
    manual_chatgpt_loop.ipynb
  examples/
    manual_chatgpt_loop_example.py
  output/
    screenshots/
    camera_states/
    action_history.json
    llm_prompt.txt
  data/
    foot_256x256x256_uint8.raw   # place your volume here
```

## Extending Later

| Goal | What to add |
|---|---|
| GUI | Wrap `CameraReasoningSession` in a Qt/Tk window; wire buttons to `process_chatgpt_response` |
| File watcher | Watch `output/chatgpt_response.txt` for changes and auto-call `process_chatgpt_response` |
| Full autonomous agent | Replace the manual paste step with an API call in `session.py` |
| Mesh support | Add `build_mesh_pipeline(path)` to `volume_scene.py` |
| Volume rendering | Replace `build_isosurface_pipeline` with `vtkSmartVolumeMapper` pipeline |

## Visualization Orchestrator

`visualization_orchestrator/` sits on top of the existing camera and isovalue agents and
turns a single free-form instruction (e.g. *"Show the inside of the head from behind"*)
into a small task graph that runs the right specialist(s), in the right order, without
a fixed `isovalue -> camera` pipeline hard-coded anywhere.

### Architecture

```
User Request
    |
    v
Planner LLM (planner.py)              -- decides WHAT needs to happen, as capabilities
    |
    v
VisualizationPlan (models.py)         -- validated task graph (plan_validator.py)
    |
    v
VisualizationExecutor (executor.py)   -- deterministic Python: scheduling, retries,
    |                                     transactional state patches, replanning
    v
AgentRegistry (registry.py)           -- decides WHICH agent satisfies a capability
    |
    +--> CameraSpecialist   (specialists/camera_adapter.py)   -- wraps camera_reasoning's
    |                                                             label-blind, three-pass
    |                                                             visual-rollout loop
    +--> IsovalueSpecialist (specialists/isovalue_adapter.py) -- histogram-informed
    |                                                             one-shot candidate sweep
    +--> OrientationSpecialist (specialists/orientation_adapter.py) -- plain LLM roll
    |                                                             correction, separate
    |                                                             from positioning
    +--> future specialists (transfer-function, clipping, segmentation, ...)
    |
    v
VisualizationState (state.py)         -- shared state, field-owned per agent
    |
    v
FinalVerifier (verifier.py)           -- checks the ORIGINAL instruction end-to-end
```

The planner never talks about agents by name -- it only emits `required_capability`
values (e.g. `"reveal_internal_structure"`, `"show_object_from_direction"`). The registry
is the single place that resolves a capability to a concrete specialist, so a capability
can later be served by more than one agent (e.g. `reveal_internal_structure` by either the
isovalue agent or a future opacity/clipping/segmentation agent) without touching the
planner or executor.

Each specialist still owns its entire internal render -> evaluate -> act -> repeat loop
(`run_until_complete`) -- the executor calls it as one opaque unit and never drives
individual camera moves or isovalue increments itself. There are two independent levels
of iteration: the executor iterates over **tasks**; each specialist iterates over its own
**internal steps**.

### Camera specialist scope and reference grounding

`CameraSpecialist` wraps `camera_reasoning.blind_visual_rollout_agent`, not the plain
`visual_rollout_agent` -- every iteration, candidate views are rendered, stripped of their
real action names (replaced with opaque, reshuffled IDs), diagnosed by the LLM, selected
deterministically in Python, and only then resolved back to a real action. Its candidate
set is azimuth/elevation framing only (`show_object_from_direction`, `adjust_viewpoint`)
-- it never proposes zoom, pan, roll, or undo, so those capabilities are not registered
for this agent at all (a future specialist could add them).

Selection quality depends heavily on having a reference-view bank for Pass 1 to compare
against; without one, Pass 1 reports everything as "unclear" and Pass 2 falls back to a
directional sweep. `CameraSpecialist(session, reference_image_paths=..., node_descriptions=...)`
and `VisualizationOrchestrator(..., camera_reference_image_paths=..., camera_node_descriptions=...)`
accept one, but nothing loads a bank automatically -- the caller loads and passes one
explicitly. Two bank formats/loaders are supported (see
`specialists/camera_adapter.py`):

- `load_simple_reference_bank(descriptions_path)` -- the flat `{node_id: description}`
  format under `reference_views_medical/<object>/reference_views_simple.json`, with each
  node's image conventionally at `<node_id>.png` next to it. This doesn't need to be a
  render of the exact same dataset/isovalue -- Pass 1 judges viewpoint resemblance by eye,
  not exact pixel/rendering match, so real reference photographs work fine (see
  `demo.py`/`notebooks/visualization_orchestrator_demo.ipynb`, which use
  `reference_views_medical/skull/` against the skull VTK dataset).
- `load_reference_bank(nodes_path, descriptions_path)` -- the graph-based
  `camera_nodes.json` + `view_descriptions.json` pair produced by
  `examples/generate_camera_relative_views.py`, for a bank rendered from the exact
  dataset/isovalue in use.

By default (`CameraSpecialist(..., sequential_diagnosis=True)` /
`VisualizationOrchestrator(..., camera_sequential_diagnosis=True)`), Pass 1 diagnoses
candidates one LLM call at a time (`diagnose_blind_candidates_sequentially` in
`blind_visual_rollout_agent.py`) rather than all 7 in a single batched call -- this costs
more LLM calls per iteration, but avoids a real observed failure mode where the model
would mislabel which candidate a score/description belonged to (e.g. reporting a
left/right judgment that actually applied to a different candidate) when several
similar-looking renders were shown together. Set it to `False` to go back to the original
single-call-per-iteration behavior once that gets revisited for cost/latency.

### Isovalue specialist: histogram-informed, multi-angle candidate sweep

`IsovalueSpecialist` is a candidate sweep, not an iterative loop:
`compute_histogram_local_minima_isovalues` (in `specialists/isovalue_adapter.py`) finds the
LOCAL MINIMA ("valleys") of the volume's intensity histogram -- thresholds that sit
*between* two materials, where relatively few voxels share that exact intensity, which
tends to produce a clean, low-noise surface. Local maxima (the middle of a homogeneous
material) are deliberately excluded, since thresholding there is what produces fragmented,
speckly surfaces -- directly relevant to `reduce_surface_noise`.

By default (`multi_angle=True`), `render_isovalue_candidates` renders each candidate from 6
angles -- front/right/back/left/top/bottom, computed RELATIVE to the session's current
camera via `camera_actions.apply_action` (not fixed absolute world axes like
`examples/render_head_iso_candidates.ipynb`'s `six_view_cameras`, which was tuned
specifically for a different dataset and wouldn't transfer) -- since noise can be visible
from one angle and hidden from another. Those 6 angles are combined into ONE labeled tiled
image per candidate (`_combine_views_into_grid`, each tile's angle name burned directly
into the pixels) rather than sent as 6 separate attachments, so the single final comparison
call still only has one image per candidate -- N candidates means N image attachments
either way, `multi_angle` only changes what each one shows. Set `multi_angle=False` to
render just the current angle per candidate instead of a 6-way grid. An earlier version of
this specialist used a blind iterative "should the isovalue go up or down?" loop with no
grounding in the volume's actual data distribution; it's no longer used.

### Orientation specialist: roll correction, separate from positioning

`camera_reasoning`'s blind visual-rollout pipeline (wrapped by `CameraSpecialist`) only
searches POSITIONAL movements -- azimuth/elevation, i.e. which side of the object is shown.
`ROLL` isn't even in its candidate action subset, so a correctly-positioned view can still
render tilted or upside-down with nothing in that pipeline able to fix it. `OrientationSpecialist`
(`specialists/orientation_adapter.py`, capability `correct_view_orientation`) handles that
as its own separate task, deliberately NOT folded into `CameraSpecialist`'s loop: one plain
LLM call per iteration ("is this image upright? if not, how many degrees should the camera
roll?"), no blind candidate rendering, no reference-bank matching -- "is this tilted" is a
much simpler visual question than "which side is this", so it doesn't need the heavier
pipeline. It only ever adjusts `view_up` (roll) and never re-examines which side is shown.

Because the planner prompt is built dynamically from the registry's capability catalog,
registering this specialist is all that's needed for the planner to start sequencing
`correct_view_orientation` tasks after camera-positioning tasks when relevant -- no executor
or prompt-template changes required.

### Registering a new specialist

1. Implement `VisualizationSpecialist.run_until_complete(goal, state, constraints,
   success_criteria) -> AgentExecutionResult` (see `specialists/base.py`). Internally it
   can do whatever render/evaluate/act loop it needs -- the executor doesn't care.
2. Declare an `AgentSpec` (see `capabilities.py`) listing the capability names it
   provides, which `VisualizationState` fields it **owns** (may write) and which it
   **requires** (must already be present to run).
3. `registry.register(my_specialist_instance, MY_AGENT_SPEC)`.

Nothing else needs to change -- the planner prompt is built dynamically from
`registry.capability_catalog()`, so the new capability shows up automatically, and the
executor already knows how to schedule any capability-labelled task.

### How task planning works

`Planner.plan()` builds a prompt from the user instruction, a summary of the current
`VisualizationState`, and the registry's live capability catalog, then asks the LLM for a
`VisualizationPlan` (a small DAG of `PlannedTask`s, pydantic-validated). `plan_validator.py`
then deterministically rejects duplicate task IDs, unknown capabilities, missing/self
dependencies, cycles, and empty success criteria -- `Planner` retries with the rejection
reasons appended to the prompt (bounded by `max_attempts`) before giving up.

Explicit ordering language in the user's instruction ("first", "then", "after") is
expected to show up as `dependencies` in the plan; the executor only ever runs a task once
every task_id in its `dependencies` has completed.

### How state ownership works

`VisualizationState` (state.py) is a single dataclass shared read-only by every
specialist. A specialist returns a `state_patch` dict instead of a new state outright;
`validate_patch()` rejects any patch that touches a field outside `AgentSpec.
owned_state_fields` (e.g. the camera specialist cannot write `isovalue`). The executor
snapshots the state before calling a specialist (`clone_state`) and only calls
`apply_patch` after both goal-satisfaction and patch-ownership checks pass -- on failure
the snapshot is kept and the mutated attempt is discarded.

### How replanning works

When a specialist exhausts its `max_attempts` without satisfying its task's success
criteria, or when final verification fails after every task individually succeeded, the
executor hands a structured failure report (reason + `suggested_capabilities` +
unsatisfied criteria) to `Replanner`, which re-invokes the planner with that context so it
can substitute a different capability, insert a prerequisite task, or give up cleanly.
Already-completed tasks are not re-run. Replanning is bounded by `max_replans`, and total
work is separately bounded by `max_total_tasks` / `max_total_agent_calls`, so nothing
retries or replans indefinitely.

### Running the demo

```bash
.venv/bin/python -m visualization_orchestrator.demo
```

Requires a working `OPENAI_API_KEY` (planner, both specialists, and the verifier all make
real LLM calls) and a VTK install. It runs several example instructions (camera-only,
isovalue-only, and compound) against `data/skull_256x256x256_uint8.raw` and prints the
interpreted goal, generated task graph, selected agent per task, task results, final state
summary, and final verification result for each one.

### Tests

```bash
for f in tests/test_orchestrator_*.py; do .venv/bin/python "$f"; done
```

These use fake specialists/planner/verifier (`tests/orchestrator_fakes.py`) -- no real LLM
or VTK calls -- and cover plan validation, registry selection, state-patch ownership, and
executor scheduling/replanning behavior (see each file's docstring for the exact list).
