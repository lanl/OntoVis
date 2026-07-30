# What we learned about the LLM, building the camera / reference-matching specialist

Documentation of concrete, empirically-observed LLM strengths/weaknesses from running the
reference-grounded blind rollout (`camera_reasoning/blind_visual_rollout_agent.py`) against
real photo banks (e.g. `reference_views_medical/skull`, `reference_views_medical/foot`).
Companion to `isovalue_llm_findings_and_next_steps.md`, which covers the isovalue specialist
instead. Not general claims about LLMs -- specifically what we observed, with this model, on
this task.

## Findings from 2026-07-29

**Batching too many images into one call degrades reasoning quality.** When a single call is
given many images at once (e.g. every candidate view plus the full reference bank together),
response quality drops noticeably. Next step: send one representative image per
window/candidate at a time (sequential per-candidate calls) instead of batching everything
into one shot -- mirrors the isovalue specialist's own one-call-per-candidate design, which
was adopted for a related reason (see `isovalue_llm_findings_and_next_steps.md`'s
"cross-candidate mislabeling in batched comparisons").

**Occasionally picks the wrong image/candidate outright.** The model's own match judgment is
sometimes just incorrect, independent of batching. Next step: add a verifier pass with a
little extra context (not a full re-diagnosis) to catch and correct a bad pick before it's
acted on, rather than trusting Pass 1/Pass 2's selection unconditionally.

**Reference images must actually resemble the target object, or reasoning quality collapses.**
If the candidate render doesn't closely resemble the object the reference bank depicts (e.g.
an isolated/partial foot structure being compared against whole-foot reference photos), the
model's matching reasoning gets visibly worse. This surfaces as the model reporting that no
reference image matches at all (`reference_match_quality: "unclear"` for every candidate) --
a legitimate abstention, but a signal that the reference bank's coverage doesn't fit what's
actually being rendered, not that the model is malfunctioning.

**The loop can spin without making progress.** Current stopping heuristic: stop once the same
candidate gets picked twice in a row. This doesn't reliably terminate -- it can keep
oscillating or re-selecting without ever satisfying that condition. Next steps to try:
- Cap the number of iterations and let the user decide whether to keep going past that cap,
  instead of relying solely on the twice-in-a-row heuristic.
- Try varying the transfer function mid-loop as an escape hatch when stuck, rather than only
  varying camera actions.
