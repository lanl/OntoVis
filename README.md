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
    +--> IsovalueSpecialist (specialists/isovalue_adapter.py) -- histogram-band
    |                                                             selection + derived ramp
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

### Isovalue specialist: fixed-window selection + derived opacity ramp

`IsovalueSpecialist` doesn't pick a single isovalue number, and doesn't ask the LLM to
compare many similar-looking threshold candidates. Instead (`specialists/isovalue_adapter.py`):

1. `compute_fixed_windows` splits the volume's full intensity range into
   `DEFAULT_NUM_WINDOWS` (8 by default) equal-width `[low, high)` windows -- pure
   arithmetic, no file read, no dependence on the dataset's actual data distribution.
2. One cheap preview per window is rendered (`render_window_previews`, reusing the
   multi-angle tiled-grid rendering below) under a NEUTRAL GRAYSCALE ramp
   (`build_evaluation_ramp_for_window`) -- not the final warm palette. A two-stage,
   goal-blind pipeline then picks one window (see "Two-stage goal-blind selection" below).
3. `build_opacity_ramp_for_band` DETERMINISTICALLY derives the FINAL opacity/color transfer
   function from the selected window's own `[low, high]` range and midpoint -- no
   hand-tuned constants, no extra LLM call. Opacity ramps up gradually from `low` to
   `peak_opacity` at the window's midpoint, then holds at a slightly higher
   `sustain_opacity` from `high` onward. This is applied via direct volume rendering
   (`CameraReasoningSession.set_transfer_function`), not a single hard isosurface
   threshold, because a gradual ramp lets THICK, continuous material (many voxels deep
   along the viewing ray) accumulate to full opacity while THIN/isolated noise at the same
   intensity stays comparatively faint -- see the module docstring for the accumulation
   argument in full.

This replaced three earlier, simpler approaches that all turned out to be unreliable in
practice:

- **Histogram-band selection**: segment the volume's own histogram into material bands via
  valley-finding (local minima between peaks), instead of a fixed grid. This worked well
  when a target material formed a genuinely separate histogram peak
  (`data/vis_male_128x256x256_uint8.raw`'s background/soft-tissue/bone), but produced only
  1-2 useless bands for datasets where materials blend as a smooth, unimodal intensity
  gradient with no interior valley at all (`data/skull_256x256x256_uint8.raw`,
  `data/foot_256x256x256_uint8.raw` -- bone never separates from soft tissue there).
- **Self-judged refinement**: to recover from the case above, the LLM was asked to
  self-report whether its selected band still looked like a mix of materials, and if so, a
  second round would subdivide that band and ask again. Verified against the real API: this
  self-judgment rarely fired even when it clearly should have.
- **Direct goal-aware evaluation** (both in one batched call comparing all windows, and
  later one call per window): letting the LLM see the rendered image AND the goal in the
  same call, in either form, was confirmed -- by directly inspecting the actual rendered
  images -- to cause goal-conditioned hallucination. Asked to "show the skull" against
  `data/vis_male_128x256x256_uint8.raw`, the model described a smooth, featureless
  soft-tissue window (visible ear, scalp dome, neck skin folds -- no sutures, no facial
  bones) as "a complete skull... clear anatomical detail including the cranium, facial
  bones, and suture lines", with high self-reported confidence, EVEN when that was the
  only image shown in an isolated single-window call. That ruled out simple
  cross-candidate mislabeling as the cause -- the model was inventing goal-consistent
  detail from a plausible silhouette and color, not confusing one candidate's description
  for another's.

A fixed, dataset-agnostic grid of windows sidesteps the first two failure modes (never
depends on the histogram having any particular shape); the two-stage pipeline below
sidesteps the third.

By default (`multi_angle=True`), `render_window_previews` renders each window from 6 views,
computed RELATIVE to the session's current camera via `camera_actions.apply_action` (not
fixed absolute world axes like `examples/render_head_iso_candidates.ipynb`'s
`six_view_cameras`, which was tuned specifically for a different dataset and wouldn't
transfer) -- since noise can be visible from one angle and hidden from another. Each view is
saved as its own SEPARATE, full-resolution image file (never combined into a tile -- tiling
shrinks each view and can visually compress detail that matters for Stage 1's observation)
and all 6 are sent together in Stage 1's single call per window via `extra_images`, each
preceded by its own opaque `"view_N"` label (see "Opaque view ids" below). Set
`multi_angle=False` to render/send just 1 view per window instead of 6.

Earlier versions of this specialist also tried a blind iterative "should the isovalue go up
or down?" loop, and a batched sweep comparing many individual candidate isovalues directly
-- all replaced outright, not layered on top of each other.

#### Two-stage goal-blind selection

Window selection is split into two LLM calls with a hard information barrier between them,
so nothing that's looking at an image also knows what it's "supposed" to find there:

```
Window renders (6 SEPARATE full-resolution views per window, neutral grayscale, see below)
    |
    v
Stage 1 (per window, BLIND) -- _observe_window_blind
    ONE call per window, showing all 6 of its views TOGETHER (never tiled). Describes only
    visible geometry: shape, continuity, cavities, fragments, symmetry, etc. NEVER shown:
    the goal, the window's real label/intensity range, or any other candidate. Opaque ids
    only -- "candidate_0", "candidate_1", ... for windows, "view_0".."view_5" for views.
    |
    v
Stage 2 (ONE call, TEXT-ONLY) -- _select_candidate_from_observations
    given the goal and every candidate's STORED Stage-1 observation (no images at all),
    selects the best match, or abstains ("no_match") if nothing has sufficient evidence.
    |
    v
Selected window (opaque id mapped back to its real window only AFTER this point) or no_match
```

Stage 2 never sees the images again -- it can only reason from what Stage 1 already wrote
down, so it cannot introduce visual "evidence" beyond what was actually reported.
Abstention (`"no_match"`) is a legitimate, propagated outcome, not a failure to paper over
-- if no candidate's blind observation supports the goal, `run_isovalue_band_selection`
returns `converged=False` / `selected_band_label=None` rather than guessing.

Costs `num_windows + 1` LLM calls (`num_windows` Stage-1 calls, one per window, plus one
Stage-2 call) instead of 1 -- each Stage-1 call itself carries all 6 of that window's views
as separate image attachments, not 6 separate calls. There is currently no cheaper
single-call fallback -- both earlier goal-aware approaches (batched and per-window) are
what this replaced, precisely because they were unreliable regardless of call count.

Neither prompt template
(`BLIND_WINDOW_OBSERVATION_PROMPT_TEMPLATE`/`GOAL_AWARE_WINDOW_SELECTION_PROMPT_TEMPLATE`)
contains any dataset- or domain-specific vocabulary (no "skull", "bone", "anatomy", etc.) --
the observation schema (shape, cavities, fragments, symmetry...) is meant to generalize to
arbitrary future volumetric datasets and goals, not just medical scans.

#### Opaque view ids

Each window's 6 views are labeled only `"view_0"` through `"view_5"` in every prompt and
log -- never `front`/`back`/`left`/`right`/`top`/`bottom`. The camera orientation a sweep
starts from is arbitrary (whatever the session's camera happened to be pointed at before
`render_window_previews` ran), so a semantic direction label would assert a meaning that
doesn't actually exist for an arbitrary starting point. `view_action_map(multi_angle)`
returns the real `view_id -> camera action` mapping (`VIEW_ACTIONS`) for our own
debugging/logging -- it's never included in any prompt.
`run_isovalue_band_selection`'s returned dict includes `"view_images"` (`{window_label:
{view_id: image_path}}`) and `"view_action_map"` so every candidate's six actual image
paths and their real camera transformations stay inspectable, even though the model itself
only ever sees the opaque ids.

#### Neutral evaluation color

Evaluation previews (`render_window_previews`, via `build_evaluation_ramp_for_window`) use a
single fixed grayscale color ramp for EVERY window, regardless of its `[low, high]` range --
unlike the final applied result, which still uses `build_opacity_ramp_for_band`'s normal
warm beige/tan palette. Previously, every window's evaluation preview ALSO used that same
warm palette regardless of what material it actually contained, which is a plausible
contributor to the hallucination described above (pattern-matching on a shared "looks
bone-colored" cue instead of on structure). Every window gets the SAME neutral mapping (not
a unique color per window) -- a per-window-unique color would just create a new
candidate-identity shortcut in its place.

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
