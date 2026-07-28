# What we learned about the LLM, building the isovalue specialist

Documentation of concrete, empirically-observed LLM strengths/weaknesses from building and
iterating on `visualization_orchestrator/specialists/isovalue_adapter.py`'s window-selection
pipeline (histogram bands -> fixed windows -> batched evaluation -> sequential per-window
evaluation -> two-stage goal-blind pipeline -> multi-view-per-candidate). Not general claims
about LLMs -- specifically what we observed, with this model (`gpt-4o` via
`camera_reasoning/chatgpt_client.py`), on this task.

## What the LLM is bad at

**Goal-conditioned hallucination.** When a call is shown an image AND the user's goal at
the same time, it can confidently invent goal-consistent structure that isn't actually in
the image. Confirmed directly: asked to "show the skull," the model described a smooth
soft-tissue window (visible ear, scalp dome, neck skin folds -- no bone-like structure at
all) as "a complete skull... clear anatomical detail including the cranium, facial bones,
and suture lines," with high self-reported confidence. This happened even in a fully
isolated single-image call (nothing to cross-attribute against), which ruled out simple
cross-candidate mixups as the explanation -- the goal itself was priming the interpretation
of the image. This is the reason the pipeline now hard-splits into a BLIND stage (Stage 1,
never sees the goal) and a goal-aware stage (Stage 2, never sees the images).

**Cross-candidate mislabeling in batched comparisons.** Separately from goal-conditioned
hallucination: when several similar-looking images are compared in ONE call, the model can
write near-identical descriptions for genuinely different images, or attribute one
candidate's judgment to another. Confirmed on both the camera specialist (motivated its
existing `sequential_diagnosis` option) and the isovalue specialist's earlier batched
window-comparison call.

**Unreliable self-judgment / self-critique.** An earlier design asked the model to
self-report whether its own selection still looked like a mix of materials ("should this be
refined further?"). Verified against the real API: across 4 consecutive live calls with the
same goal/dataset, this self-report never fired -- even though the selected result was
confirmed, by directly looking at the image, to include material well beyond the target.
The model reliably judged its pick as "good enough relative to the other option shown," not
"good enough in an absolute sense."

**Overconfident numeric self-scoring.** Even after fixing cross-image mislabeling (moving
to one-image-per-call evaluation), the model still assigned a high self-reported match
score (95/100) to an outright wrong, hallucinated answer. A model's own confidence number
was not a reliable signal of correctness.

**Likely sensitivity to superficial visual cues over structure.** Suspected (not fully
isolated) contributor to the hallucination above: every window originally rendered in the
same warm beige/tan palette regardless of what material it contained, so "looks
bone-colored" may have been acting as a false identity cue independent of actual geometry.
Evaluation previews now render in a fixed neutral grayscale specifically to remove this
variable.

**Non-determinism / inconsistency across repeated identical calls.** Re-running the same
goal against the same data does not reliably produce the same answer. Observed in a live
orchestrator run: after the verifier said "the isovalue needs to move higher" and
triggered a replan, the isovalue specialist's next independent evaluation picked a *lower*
window than its first attempt -- moving the opposite direction from the feedback. This
connects to the next point.

**No incremental/progressive state across repeated invocations.** Each call to the
specialist re-evaluates every candidate from scratch with no memory of prior attempts. When
a verifier asks for a specific directional correction ("go higher," "reduce noise
further"), the specialist has no mechanism to actually move in that direction -- it just
re-samples independently, which (combined with the non-determinism above) can move the
result in an unhelpful or even opposite direction.

## What the LLM is good at

**Blind, generic visual/geometric description.** When isolated from the goal entirely
(Stage 1), the model produced genuinely grounded, specific geometric observations --
"bulbous protrusions arranged in a row... deep grooves separating them... continuous
supporting body" for a window that (goal-aware Stage 2 confirmed separately) actually
showed toe-bone joints -- without ever hallucinating "bone" or any domain vocabulary,
because it wasn't told what to look for. This generalized correctly across arbitrary,
non-medical synthetic test cases too (branching/tubular network descriptions).

**Multi-view synthesis with citation.** Given several images of the same object from
different (opaquely-labeled) viewpoints, the model reliably produced both per-view
observations and a cross-view summary, correctly citing exactly which view ids supported
each claim.

**Text-only reasoning over structured data it's given.** Stage 2 (goal-aware selection from
stored text observations, no images at all) reliably picked the candidate whose *reported*
observations best matched the goal, and correctly used contradicting evidence (fragmented
continuity, exposed internal structure) to rule out competing candidates -- reasoning
purely over text it was handed, not re-perceiving anything.

**Following structured JSON schema instructions.** Across many real API calls with fairly
elaborate nested schemas (`per_view_observations`, `cross_view_summary`,
`decision`/`selected_candidate`/`explanation`), responses reliably parsed as valid,
schema-conforming JSON -- this was never the source of a failure in this project; every
failure mode above was a *content* problem, not a *format* problem.

## Next step to try: batching instead of one-shot-per-candidate

Current design: Stage 1 makes **one call per window/candidate** (never batched across
candidates -- all 6 of that candidate's views go in together, but only that one
candidate's views). Total cost is `num_windows + 1` calls.

Idea to try next: batch **multiple candidates** into fewer Stage-1 calls (e.g. compare
several candidates' view sets in one call, or all of them at once), instead of one
call per candidate.

Why this might work now, when the original batched approach didn't: the original batched
failure mode (near-identical hallucinated descriptions for different candidates) was
observed when the SAME call also carried the goal -- i.e. goal-conditioned hallucination
and cross-candidate mislabeling were entangled together. Since Stage 1 is now goal-blind by
construction (the goal never enters Stage 1 regardless of batching), batching candidates
back together might now mostly reintroduce just the narrower "which description belongs to
which candidate" bookkeeping risk -- not the goal-hallucination risk, which lives entirely
in Stage 2 and is unaffected by how Stage 1 is batched. That's a real hypothesis, not a
confirmed result -- worth testing empirically (fewer calls/cost/latency vs. re-checking
whether cross-candidate mislabeling comes back) before deciding whether to keep one-call-
per-candidate as the default.
