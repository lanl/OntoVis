# Technical Documentation

## 1. Overview

This repository implements an LLM-driven visualization system for volumetric datasets rendered with VTK, built around three components.

`VisualizationOrchestrator` (`visualization_orchestrator/orchestrator.py`) is the entry point. A caller gives it one free-form instruction, such as "Show the skull in a lateral view, with the superior aspect at the top," and it plans which underlying agent or agents the instruction requires, runs them in order, and checks the final rendered image against the instruction before reporting success. It also registers two smaller specialists, one for roll correction and one for applying an already-exact stated value with no LLM call, described in sections 4.4 and 4.5.

The camera reasoning agent (`camera_reasoning/blind_visual_rollout_agent.py`) aligns the camera to a requested viewpoint. Each iteration it renders every candidate camera move, strips their action names so the LLM cannot reason from words like "left" or "right," and resolves the LLM's visual judgment back into a real action only after the judgment is made.

The transfer function agent (`visualization_orchestrator/specialists/isovalue_adapter.py`) chooses which intensity range of the volume to display. It splits the volume's intensity range into fixed windows, has the LLM describe each window's rendered preview without telling it the user's goal, and only then makes a separate, goal-aware selection from those stored descriptions.

Both agents send every LLM call through `ask_chatgpt` in `camera_reasoning/chatgpt_client.py`, which reads `OPENAI_API_KEY` from a `.env` file and defaults to the `gpt-4o` model.

The two runnable entry points are `notebooks/visualization_orchestrator_demo_head.ipynb` and `notebooks/visualization_orchestrator_demo_foot.ipynb`. Each instantiates one `VisualizationOrchestrator` against a different dataset (`data/vis_male_128x256x256_uint8.raw` and `data/foot_256x256x256_uint8.raw`) and issues a sequence of `.run()` calls.

## 2. Repository Structure

```
camera_reasoning/
  session.py                     CameraReasoningSession: VTK scene, rendering, prompt/response loop
  camera_actions.py              named camera actions (VALID_ACTIONS) and apply_action()
  camera_state.py                get/set/save/load camera state as plain dicts
  volume_scene.py                VTK pipeline construction: raw volume loading, isosurface and
                                  volume-rendering actors, screenshot saving
  prompt_writer.py                builds the text prompt used by CameraReasoningSession's own
                                  loop (not the orchestrator's specialists, which build their
                                  own prompts)
  action_parser.py                extracts one action name from an LLM response
  chatgpt_client.py               ask_chatgpt(): the single function that calls the OpenAI API
  spatial_knowledge.py            optional object-space grounding from a JSON spec
  view_description_generator.py   JSON-extraction helper reused by visual_rollout_agent.py
  visual_rollout_agent.py         render_candidate_rollouts() (candidate rendering, reused by
                                  the blind agent) plus an older, non-blind selection loop
  blind_visual_rollout_agent.py   the three-pass, label-blind camera agent used by CameraSpecialist
  medical_reference_views.py      generates rotated reference images and landmark descriptions
  simple_reference_labels.py      writes the flat label mapping actually used at run time
visualization_orchestrator/
  orchestrator.py                 VisualizationOrchestrator, the single entry point
  planner.py / replanner.py       turns an instruction into a VisualizationPlan
  executor.py                     runs a plan's tasks and calls the final verifier
  registry.py                     resolves a capability name to a concrete specialist
  capabilities.py                 AgentSpec / AgentCapability dataclasses
  state.py                        VisualizationState and patch validation
  models.py                       pydantic models for the plan, plus result dataclasses
  verifier.py                     FinalVerifier: checks the rendered image against the instruction
  plan_validator.py               deterministic checks on a planner-produced plan
  render_log.py                   records every committed render's parameters and image
  prompts/                        planner and verifier prompt templates
  specialists/
    camera_adapter.py             CameraSpecialist, wraps blind_visual_rollout_agent
    isovalue_adapter.py           IsovalueSpecialist, the transfer-function agent
    orientation_adapter.py        OrientationSpecialist, roll correction only
    direct_adapter.py             ApplyExactParametersSpecialist, no LLM call
notebooks/
  visualization_orchestrator_demo_head.ipynb
  visualization_orchestrator_demo_foot.ipynb
examples/
  generate_medical_reference_views.py   full landmark/relations pipeline (reference_views.json)
  generate_simple_reference_labels.py   flat label pipeline (reference_views_simple.json)
reference_views_medical/
  skull/, foot/                   seed photographs plus generated rotations and labels
data/
  vis_male_128x256x256_uint8.raw, foot_256x256x256_uint8.raw, and a transfer-function JSON
  for the vis_male dataset
tests/                             pytest suite for the orchestrator layer
```

## 3. System Workflow

A call to `VisualizationOrchestrator.run(instruction)` proceeds as follows.

`Planner.plan()` sends the instruction, a summary of the current `VisualizationState`, and the registry's capability catalog to the LLM, and parses the response into a `VisualizationPlan`. The plan is a list of `PlannedTask` objects. Each names one `required_capability` and lists its dependencies on other tasks. `plan_validator.validate_plan()` then checks the plan for duplicate task IDs, unknown capabilities, missing or self-referential dependencies, dependency cycles, and empty success criteria, and the planner retries on failure up to `max_attempts`.

`VisualizationExecutor.execute()` then runs the plan. For each task whose dependencies are already satisfied, it asks `AgentRegistry.select_agent()` for a specialist that provides the task's capability, calls that specialist's `run_until_complete()`, and validates the returned `state_patch` against the fields that specialist is declared to own (`AgentSpec.owned_state_fields`). A valid patch is applied to a new `VisualizationState` with `apply_patch()`, which never mutates the previous state. If a task fails and its attempts are exhausted, or if the whole plan finishes but leaves the goal unmet, the executor hands the failure to `Replanner`, which asks the planner for a new plan given that failure context. Replanning is bounded by `max_replans`, and total work is bounded by `max_total_tasks` and `max_total_agent_calls`.

Once every task in the plan has completed, `FinalVerifier.verify()` checks the rendered image against the original instruction and the plan's `final_success_criteria`, unless every task in the plan came from a `deterministic` specialist (see `ApplyExactParametersSpecialist` below), in which case the executor skips the vision call and trusts the deterministic result directly.

Underneath all of this, every specialist operates on the same `CameraReasoningSession`. A specialist reads the camera or transfer-function fields it needs from `VisualizationState`, applies its own internal render, evaluate, and act loop against the live VTK scene, and returns a patch describing only the state it changed.

## 4. Main Components

### 4.1 VisualizationOrchestrator

`visualization_orchestrator/orchestrator.py` is the class both demo notebooks actually instantiate, and the facade the rest of this section's components sit behind. `VisualizationOrchestrator.__init__()` takes a dataset path, its dimensions, scalar type, and spacing, an initial isovalue, and an `output_dir`. It builds one `CameraReasoningSession` and calls `initialize()` on it, builds an `AgentRegistry` and registers `CameraSpecialist`, `IsovalueSpecialist`, `OrientationSpecialist`, and `ApplyExactParametersSpecialist` against it, and builds a `Planner`, a `VisualizationExecutor`, and a `RenderLog`.

`run(instruction)` is the only method a caller needs for normal use. It calls `Planner.plan()` and `VisualizationExecutor.execute()` (see section 3 for what happens inside those two calls), stores the resulting state on `self.state`, appends the `ExecutionResult` to `self.history`, and returns that result. State carries forward from one `run()` call to the next, so a sequence of instructions behaves like a multi-turn conversation against the same visualization.

`image_path` is a property returning `self.state.rendered_image_path`, the most recently rendered image. `render_from_log(index)` and `render_from_params(params)` reproduce a past `render_log` entry's exact camera and transfer-function parameters and adopt it as the current state, which is useful for grounding a relative instruction such as "rotate left 60 degrees" against a specific earlier render rather than wherever the live camera last ended up.

`camera_reference_image_paths` and `camera_node_descriptions`, if given, are passed straight through to `CameraSpecialist`'s reference-view bank (see section 4.2). `on_iteration`, if given, is called once per internal specialist iteration, not just once per `run()` call, with a normalized dict describing that iteration's candidates and selection, the same shape every specialist reports through (see `CameraSpecialist._handle_camera_iteration`, `IsovalueSpecialist._handle_band_result`, and `OrientationSpecialist._handle_iteration`).

### 4.2 Camera Reasoning Agent

The camera reasoning agent is treated here as one component, though it is built from two files. `camera_reasoning/session.py` defines `CameraReasoningSession`, which wraps one VTK render window and executes camera actions but has no logic of its own for deciding which action to take next. `VisualizationOrchestrator` builds and owns exactly one of these, as `self.session`. `CameraReasoningSession.__init__()` takes a raw volume path, its dimensions, scalar type, and voxel spacing, plus rendering options (`use_volume_rendering` for direct volume rendering versus a fixed isosurface, an isovalue, and optional opacity and color transfer-function points). `initialize()` loads the volume with `load_raw_volume()`, builds either `build_isosurface_pipeline()` or `build_volume_rendering_pipeline()`, and resets the camera. `render_and_save()` renders the current scene and writes a screenshot. `set_isovalue()` and `set_transfer_function()` rebuild the isosurface actor or the volume-rendering prop in place and swap it into the existing renderer, so a specialist can change what is displayed without disturbing the camera position set by another specialist. `process_chatgpt_response()` extracts an action name from a pasted or returned LLM response with `extract_action()` and applies it with `camera_actions.apply_action()`.

The decision of which action to take is made separately, in `camera_reasoning/blind_visual_rollout_agent.py`, wrapped by `CameraSpecialist` in `visualization_orchestrator/specialists/camera_adapter.py`. This part renders candidates and applies its final choice through the same `CameraReasoningSession` instance, so it never talks to VTK directly. It runs a three-pass loop, coordinated by `run_blind_visual_rollout_alignment_loop()`.

Each iteration, `render_candidate_rollouts()` (defined in `visual_rollout_agent.py` and reused unmodified) restores the camera to its current state, applies one candidate action, renders and saves the result, and restores the camera again, for every action in `BLIND_CANDIDATE_ACTION_SUBSET`. That subset covers the four scalable movement families (`AZIMUTH_LEFT`, `AZIMUTH_RIGHT`, `ELEVATION_UP`, `ELEVATION_DOWN`, rendered only at their COARSE magnitude), the two fixed 180-degree turns, and `STOP` (rendered as the current image unchanged).

`prepare_blind_candidates()` then assigns each rendered candidate a random opaque ID such as `CANDIDATE_K7P4Q`, regenerated and reshuffled every iteration, and copies each image to a neutral filename. The mapping from opaque ID to real action is kept in Python only and is never sent to the LLM.

Pass 1, `diagnose_blind_candidates()`, sends the current render, the target description, an optional reference-view bank, and every blind candidate to the LLM in one call, and asks for a structured JSON diagnosis: a view description and reference-bank match for the target and the current render, and the same plus a `similarity_score` and `confidence` for every candidate. This pass never selects a candidate and never outputs `STOP`. If `sequential_diagnosis` is set, `diagnose_blind_candidates_sequentially()` runs this same call once per candidate instead of once for the whole batch. This costs more LLM calls but eliminates cross-candidate attribution errors.

Pass 2, `select_candidate_from_diagnosis()`, is plain Python with no LLM call. It ranks candidates by `similarity_score`, then `confidence`, then a fixed numeric value derived from `reference_match_quality`, and returns the top-ranked one. It returns `STOP` instead when the current render's own score already clears `stop_similarity_threshold` and no candidate improves on it by at least `stop_improvement_margin`. If every candidate's reference match is `"unclear"`, ranking falls back to either a standout self-reported score (the escape hatch described in the function's docstring) or to continuing the same movement direction as the last applied action.

Pass 3, `estimate_selected_candidate_magnitude()`, is called only when the selected candidate belongs to a scalable movement family. It shows the LLM only the current render, the target, and the already-selected candidate's render, and asks for one of `full`, `reduced`, or `minimal`. It cannot change Pass 2's selection, and `_validate_magnitude_result()` rejects any response that echoes a different candidate ID.

`_construct_final_action()` then combines the selected candidate's movement family with the Pass 3 magnitude into a real action name such as `AZIMUTH_LEFT_MEDIUM`, checks it against `camera_actions.VALID_ACTIONS`, and the loop applies it through `session.process_chatgpt_response()`.

`CameraSpecialist.run_until_complete()` syncs the session's camera to the incoming `VisualizationState`, builds a target description from the task's goal and success criteria, runs the loop, and returns a `state_patch` covering `camera_position`, `focal_point`, `view_up`, `current_view_description`, and `rendered_image_path`. The task counts as satisfied only if the loop's last applied action was `STOP`.

An optional reference-view bank (`reference_image_paths`, a node ID to image path mapping, and `node_descriptions`) grounds Pass 1's judgment against known views of the same object. `CameraSpecialist` does not load a bank on its own. The caller loads one with `load_simple_reference_bank()` (see section 5) and passes it in at construction time.

### 4.3 Transfer Function Agent (Isovalue Specialist)

The isovalue agent lives in `visualization_orchestrator/specialists/isovalue_adapter.py` and is wrapped by `IsovalueSpecialist`. It does not search individual isovalue numbers. It splits the volume's full intensity range into `DEFAULT_NUM_WINDOWS` (8) equal-width windows with `compute_fixed_windows()`, picks one window with a two-stage, goal-blind selection process, and then locally refines the chosen window's range and opacity.

`render_window_previews()` renders each window from up to 6 camera angles, computed relative to the session's current camera through `camera_actions.apply_action()`, under a neutral grayscale ramp built by `build_evaluation_ramp_for_window()`. This grayscale ramp is used only for evaluation. It is deliberately different from the warm palette used for the eventual result, so that color cannot act as a shortcut for identifying which window contains the target material.

Stage 1, `_observe_window_blind()`, sends one window's views together in a single call and asks the LLM to identify what the candidate looks like (a `label`, `supporting_evidence`, and `confidence`) and to rate noise, fragmentation, and occlusion. This call receives no goal, no window intensity range, and no other candidate's information, only an optional one-line `dataset_description` describing what kind of data this is.

Stage 2 runs in two steps against the stored Stage-1 observations, with no images attached. `_evaluate_candidate_criteria()` asks the LLM to judge every candidate against every success criterion independently, without comparing candidates to each other. Each pair gets a status of `met`, `not_met`, or `unknown`. `_select_coarse_baseline()` then deterministically picks a starting window in Python: a candidate is viable only if it is not `not_met` on every criterion, and the viable candidate with the best criterion and artifact profile wins.

From that coarse baseline, `_run_local_refinement()` runs a range-search phase followed by an opacity-search phase. Each phase generates a small set of nearby transfer-function states (`_generate_range_candidates()` or `_generate_opacity_candidates()`), evaluates each one the same way as the coarse windows, and accepts a candidate only if `_refinement_improvement()` shows it never lowers any criterion's status and strictly improves at least one criterion or artifact dimension. When no candidate improves safely, the phase halves its step size and tries again. It stops once the step size reaches a fixed minimum.

The selected window or refined state is converted into an opacity and color transfer function by `build_opacity_ramp_for_band()`, which ramps opacity up from the window's low bound to its midpoint and holds a slightly higher opacity from the midpoint outward, and applies it through `session.set_transfer_function()`.

`IsovalueSpecialist.run_until_complete()` reads `state.dataset_description` and passes it through to every Stage-1 call, and returns a `state_patch` with the new `transfer_function` and `rendered_image_path`. The task counts as satisfied only if the final refined state meets every success criterion.

### 4.4 Orientation Specialist

`visualization_orchestrator/specialists/orientation_adapter.py` handles camera roll, which the camera agent above never touches (`ROLL` is not part of its candidate action subset). Each iteration, `OrientationSpecialist` renders the current image, sends it to the LLM with a prompt asking whether the image looks upright and, if not, how many degrees to roll it, and applies `camera.Roll()` directly if a nonzero value is returned. It stops once the LLM reports `is_correctly_oriented: true`, or after `default_max_iterations`, or if a response gives no actionable roll value.

### 4.5 Direct Parameter Specialist

`visualization_orchestrator/specialists/direct_adapter.py` handles instructions that already state an exact value, such as "set isovalue to 60" or "rotate left 60 degrees," with no LLM call at all. `ApplyExactParametersSpecialist.run_until_complete()` looks up each recognized constraint key (`isovalue`, `reset_camera`, `action`, `roll_degrees`, `azimuth_degrees`, `elevation_degrees`) in `_PARAMETER_HANDLERS` and applies it directly through the matching `CameraReasoningSession` or VTK call, in that fixed order. `AgentSpec.deterministic` is set to `True` for this specialist, which is what lets the executor skip the vision-LLM `FinalVerifier` when a plan consists entirely of direct tasks.

### 4.6 Planning, Execution, and Verification

`Planner` (`planner.py`) builds its prompt from `prompts/planner_prompt.py`. The prompt is assembled dynamically from `AgentRegistry.capability_catalog()`, so a newly registered specialist's capabilities appear in the prompt automatically. `VisualizationPlan` and `PlannedTask` (`models.py`) are pydantic models, so a malformed LLM response raises a validation error that the planner catches and retries against, rather than reaching the executor as a broken plan.

`VisualizationState` (`state.py`) is a single dataclass holding camera position, focal point, view up, isovalue, transfer function, clipping planes, and a few descriptive fields, including `dataset_description`. Every specialist may read the whole state but may only write the fields listed in its own `AgentSpec.owned_state_fields`. `validate_patch()` enforces this before a patch is applied, and rejects a patch that names an unowned or nonexistent field.

`dataset_description` is set once by `VisualizationOrchestrator.run()`, from the planner's own extraction of what the dataset itself is (not the user's goal), and is applied to state before the executor runs so that goal-blind specialists such as `IsovalueSpecialist` can read it without ever seeing what the user is trying to find.

`FinalVerifier` (`verifier.py`) judges only from the rendered image and the instruction. It does not receive the numeric camera state or any task history, because an earlier version that did so inferred orientation directly from `view_up`'s sign and overrode a correct visual judgment with an incorrect numeric one.

## 5. Reference-View Generation

Some datasets benefit from a bank of reference images the camera agent can compare candidates against (see `reference_image_paths` and `node_descriptions` in section 4.2). `reference_views_medical/skull/` and `reference_views_medical/foot/` hold this kind of bank, built from real photographs rather than VTK renders.

`camera_reasoning/medical_reference_views.py` implements the underlying pipeline, run through `examples/generate_medical_reference_views.py --config reference_views_medical/<object>/config.json`. A dataset's `config.json` lists its seed images (a node ID and a `view_label` for each, such as `"right"` or `"dorsal"`) and a list of rotation angles. `generate_reference_views()` builds or loads a frozen `landmark_profile.json` for the object, generates a rotated image for every seed and angle combination (`<node_id>_rot<angle>.png`), sends every image to the LLM to diagnose against the same landmark profile, and writes the result to `reference_views.json`.

`camera_reasoning/simple_reference_labels.py` writes the format the orchestrator's specialists actually load: a flat `{node_id: label}` mapping. `write_simple_reference_labels()` builds this purely from `config.json` and a `base_labels` dict edited directly in `examples/generate_simple_reference_labels.py`, with no LLM call, and writes it to `reference_views_simple.json` in the dataset's directory. It requires the rotated images from the pipeline above to already exist on disk, since it only writes labels, not images.

`visualization_orchestrator/specialists/camera_adapter.py`'s `load_simple_reference_bank()` reads `reference_views_simple.json` and returns the `(reference_image_paths, node_descriptions)` pair a specialist or `VisualizationOrchestrator` expects. It assumes each node's image sits next to the JSON file as `<node_id>.png`.

## 6. Configuration

`.env` (see `.env.example`) holds `OPENAI_API_KEY`, and optionally `AI_URL` to point at a different API-compatible endpoint and `AI_MODEL` to override the default `gpt-4o`. `chatgpt_client.ask_chatgpt()` raises a `RuntimeError` if `OPENAI_API_KEY` is not set.

`requirements.txt` lists the Python dependencies: `vtk`, `numpy`, `jupyter`, `openai`, `python-dotenv`, `pydantic`, `pillow`, `transformers`, `networkx`, `matplotlib`, and `pandas`.

`data/` holds the raw volumes used by the two demo notebooks (`vis_male_128x256x256_uint8.raw` and `foot_256x256x256_uint8.raw`) and `vis_male_transfer_function.json`, a worked example of the opacity and color point format `load_transfer_function_json()` reads.

## 7. Running the Code

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env`, then open either demo notebook:

```bash
jupyter notebook notebooks/visualization_orchestrator_demo_head.ipynb
jupyter notebook notebooks/visualization_orchestrator_demo_foot.ipynb
```

Both notebooks follow the same pattern: construct one `VisualizationOrchestrator` against a dataset, then issue instructions against it. For example, against the `vis_male` dataset used by `tests/test_isovalue_band_selection.py`:

```python
orchestrator = VisualizationOrchestrator(
    dataset_path="../data/vis_male_128x256x256_uint8.raw",
    dimensions=(128, 256, 256),
    spacing=(1.57774, 0.995861, 1.00797),
    isovalue=40,
    output_dir="../output/orchestrator_notebook",
)
result = orchestrator.run("Show the skull in a lateral view, with the superior aspect at the top.")
```

`spacing` here is this dataset's real per-axis voxel spacing. Passing the default isotropic `(1.0, 1.0, 1.0)` instead would silently squeeze the 128-slice axis to half the physical width of the 256-slice axes.

`result` is an `ExecutionResult`, with `success`, `task_records`, `replans_used`, `total_agent_calls`, and `final_verification`. `orchestrator.image_path` points at the most recently rendered image. `orchestrator.history` accumulates every past `ExecutionResult` from the same orchestrator instance, since state carries forward between `.run()` calls.

Tests run with:

```bash
python -m pytest tests/
```

`tests/test_orchestrator_*.py` use fake specialists and a fake planner and verifier (`tests/orchestrator_fakes.py`), so they exercise plan validation, registry selection, state-patch ownership, and executor scheduling without a real LLM or VTK call. `tests/test_isovalue_band_selection.py` runs the real VTK pipeline against `data/vis_male_128x256x256_uint8.raw` and mocks only the LLM call.

## 8. Inputs and Outputs

The main input to `VisualizationOrchestrator` is a raw volume file plus its dimensions, scalar type, and voxel spacing, and an isovalue to start from. A second, optional input is a reference-view bank for the camera specialist, loaded separately and passed in.

`output_dir` (default `output/orchestrator_session`, set to `output/orchestrator_notebook` by both demo notebooks) collects everything a run produces. `screenshots/` holds `latest.png` and per-step screenshots from `CameraReasoningSession`, plus the rendered candidates each agent evaluated under `screenshots/isovalue_windows/` and `screenshots/blind_candidates/`. `camera_states/` holds JSON camera-state snapshots. `blind_rollout_traces/` holds one JSON file per camera-agent iteration, written by `save_blind_rollout_trace()`. Each trace file holds the private candidate mapping alongside all three passes' results, for later inspection. `render_log/` holds a copy of every image `RenderLog.append()` recorded. `output/` is generated at runtime and is excluded from version control by `.gitignore`.

## 9. Extending the Code

A new specialist implements `VisualizationSpecialist.run_until_complete()` (`specialists/base.py`), declares an `AgentSpec` listing its capability names and which `VisualizationState` fields it owns and requires, and registers both with `AgentRegistry.register()`. The planner prompt and the executor's scheduling both read from the registry, so no other file needs to change for the new capability to become available.

A new exact, non-visual parameter (alongside `isovalue`, `roll_degrees`, and the others already handled) is added to `visualization_orchestrator/specialists/direct_adapter.py` by writing one function matching the existing handlers' signature and adding it to `_PARAMETER_HANDLERS`, rather than by writing a new specialist.
