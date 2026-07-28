# Problem: vision LLM confidently misreads a smooth soft-tissue head as "a skull with visible sutures"

## System context

`visualization_orchestrator/specialists/isovalue_adapter.py` implements an isovalue/opacity
specialist for VTK volume rendering. Given a CT-like volume (uint8 scalar intensities, 0-255)
and a natural-language goal (e.g. "show the skull"), it:

1. Splits the full intensity range `[0, 256)` into `num_windows` (default 8) equal-width
   windows via `compute_fixed_windows` -- pure arithmetic, e.g. `[0,32), [32,64), [64,96),
   [96,128), [128,160), [160,192), [192,224), [224,256)`.
2. Renders ONE preview image per window (`render_window_previews`) using direct volume
   rendering under an opacity/color ramp derived deterministically from that window's own
   `[low, high]` range (`build_opacity_ramp_for_band`). Each preview is a 6-angle tiled grid
   (front/right/back/left/top/bottom).
3. Asks a vision LLM (OpenAI `gpt-4o` via `camera_reasoning/chatgpt_client.py`) which window
   best matches the goal, and applies that window's ramp as the final result.

**Important detail**: every window's color ramp uses the *same* fixed palette regardless of
what material it actually contains -- `build_opacity_ramp_for_band`'s `color_points` always
go from near-black -> a warm beige/tan (`(0.85, 0.75, 0.65)`) by the window's midpoint, held
through to max intensity. So a window showing soft tissue and a window showing actual bone
render in visually similar warm tan/beige tones -- only the *geometry* differs, not the
color scheme.

## The problem

For `data/vis_male_128x256x256_uint8.raw` (dimensions `(128,256,256)`, spacing
`(1.57774, 0.995861, 1.00797)`), with goal `"show the skull"`, window `WINDOW_32_64`
(intensity range `[32, 64)`) renders as a **smooth head with a clearly visible ear, scalp
dome, and neck skin folds** -- no cranium, no facial bones, no suture lines, nothing
bone-like about the geometry at all. See the described image below (I verified this by
directly viewing the rendered PNG, not just trusting the model's text).

Despite this, when shown **only this one image** (isolated, single-image LLM call, nothing
else to compare against or mislabel), the model wrote:

> "The image shows a complete skull/bone structure rendered from 6 different angles (front,
> right, back, left, top, bottom). The skull is clearly visible with good anatomical detail
> including the cranium, facial bones, and suture lines. The rendering shows smooth, clean
> surfaces with a beige/tan coloration typical of bone tissue. All views show consistent
> bone structure with minimal artifacts or noise."

...`satisfies_goal: true`, `match_quality: 95`. This became the FINAL selected window
(the real skull window it should have picked instead, `WINDOW_96_128`, was also correctly
identified as a skull elsewhere in the same run, but `WINDOW_32_64`'s inflated
`match_quality=95` won).

For comparison, the actual skull window (`WINDOW_96_128`, intensity `[96,128)`) really does
show a skull (visible cranial sutures, mandible, teeth, orbital sockets, cervical vertebrae)
and the model's description of THAT window is accurate.

## What's been tried

1. **Batched comparison** (`WINDOW_SELECTION_PROMPT_TEMPLATE`, `sequential_diagnosis=False`
   in `run_isovalue_band_selection`): all 8 window images shown in one LLM call, model picks
   one. In one run against this same dataset/goal, this batched call ALSO wrote
   near-identical "clean skull, minimal noise" text for both `WINDOW_32_64` (wrong) and
   `WINDOW_96_128` (right) -- but that time the final pick happened to land on the correct
   window anyway. In a separate live run through the full orchestrator (different goal,
   "reduce the noise"), the batched call's final selection was confirmed wrong (picked a
   soft-tissue window) via direct visual inspection of the rendered output.

2. **Sequential per-window evaluation** (`WINDOW_EVALUATION_PROMPT_TEMPLATE`,
   `_evaluate_window_sequentially`, `sequential_diagnosis=True`, now the default): one LLM
   call per window, each shown ONLY that window's own image (`screenshot_path`, not a
   multi-image `reference_items` list), so there is nothing for the model to mislabel
   against another candidate. This was implemented specifically to rule out
   cross-candidate/cross-image mislabeling as the cause. **It did not fix this case** --
   the isolated single-image call for `WINDOW_32_64` still confidently hallucinated
   "cranium, facial bones, and suture lines" that are not present in the image at all. This
   means the error is NOT (at least not only) a batching/cross-attribution bug; it's the
   model genuinely misreading a single image's content, with high stated confidence.

## Hypotheses (untested / partially tested)

- **Shared color palette as a false cue**: since every window renders in the same warm
  beige/tan regardless of actual material (see "Important detail" above), the model may be
  partly pattern-matching on color/texture ("looks like the bone color from other renders I
  associate with skulls") rather than on structure (presence/absence of sutures, orbital
  sockets, mandible, teeth vs. an ear and smooth scalp). Not yet tested: would giving windows
  visually distinct colors (e.g. keyed to intensity) change the result?
- **Generic prompt framing**: the current prompt (`WINDOW_EVALUATION_PROMPT_TEMPLATE` in
  `isovalue_adapter.py`) just asks "does this satisfy the goal, observe what's shown" without
  telling the model what SPECIFIC features distinguish a skull from a smooth head (sutures,
  orbital sockets, mandible/teeth vs. an ear, scalp dome, skin folds). Not yet tested: would
  explicit feature-based instructions change the result?
- **General vision-LLM overconfidence/sycophancy toward the stated goal**: given a goal like
  "show the skull," the model may be biased toward reporting success rather than genuinely
  checking, especially for a plausible-looking rounded/head-shaped render in a "bone-like"
  color.


## What would help

A fix (prompt redesign, rendering change, verification step, or something else) that makes
the per-window (or per-comparison) judgment actually grounded in the image's real content --
specifically for cases like this one, where a smooth, featureless, head-shaped render in a
bone-colored palette gets confidently misidentified as a detailed skull despite having none
of a skull's actual distinguishing structure.
