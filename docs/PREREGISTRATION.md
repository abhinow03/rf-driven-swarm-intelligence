# v5-a Phase 4 evaluation — preregistration

**Committed 2026-08-11, before any v5-a training run has started.** No training or evaluation
numbers exist yet for v5-a (or v5-b/c/d) at the time this document was written. The point of
committing it now, before step 1 of the real `train_sft_v5.py` run, is so the protocol and
success bars below cannot be adjusted after the fact to match whatever result comes out.

## Protocol

- **Population**: 1,000 held-out real STGT trajectories (fresh seed, disjoint from every seed
  already used elsewhere in this project's history — training=42, ceiling=999, retired
  dev=1, mining=2024, generator diagnostics=7/10/11, coverage measurement=0). Real
  `swarm_intent.stgt.inference.sliding_window_inference` output on real generated geometry —
  not the templated `synth_context()` text battery every earlier eval in this repo also used,
  and not the Phase 1 training corpus itself (v5-a must not be evaluated on data it was
  trained on, or any close paraphrase of it).
- **Ground truth**: derived ONLY from the data generator's own true formation chain
  (`true_chain`), independently of any model's own bridge/reduction output — the same
  discipline established in sec AF/AG (`ground_truth_from_true_chain` /
  `ground_truth_pair`), after this project directly found and fixed a case where ground truth
  was circularly derived from the system under test's own output. `len(true_chain) <= 2` is
  the determinable/pair-eligible population; `len(true_chain) >= 3` has no defensible expected
  answer and correct behavior is abstention, scored separately (see the abstention/
  over-abstention convention in `src/swarm_intent/llm/evaluate.py`).
- **Repetition**: `n_runs=20` on volatile strata (any stratum whose per-run variance was
  non-trivial in a pilot check — historically this has meant `high`/`critical` and any
  ambiguity-guarded cell), with Wilson or t-distribution 95% CIs reported alongside every
  headline number, not point estimates alone.
- **Decoding**: greedy (temperature=0, `do_sample=False`) for every HEADLINE number. Sampled/
  temperature>0 runs may be used for supplementary robustness checks but are never the number
  a bar below is checked against — this keeps headline numbers exactly reproducible run to
  run.
- **Metrics**: the full set established in `llm_finetuning/eval_sft_v5.py` — per-class threat
  accuracy (Wilson 95% CI, low-n caveat below n=10), `accuracy_when_answerable` /
  `abstention_rate_when_unanswerable` / `over_abstention_rate` as three separate fields,
  under-escalation and over-escalation reported separately and directionally, JSON
  schema-validity rate, critical-pair tactical accuracy (same low-n caveat — critical has
  historically been n=2 on synthetic batteries; expect a similarly thin critical-pair count on
  1,000 real trajectories and report it with the same caveat, not a bare percentage).

## Success bars

The confirmed upstream ceiling is **83.0% threat accuracy / 77.3% pair accuracy**
(`docs/V5_LOG.md` Phase 0 close, `swarm_data/best_model.pt` sha256 `18fc201d...`, full
1000-trajectory measurement, seed=999, `robust=True`, pair-eligible-pooled). This is the swarm
classifier's own upper bound — no downstream LLM layer can exceed information the classifier
already lost, so a bar at or above this ceiling would be unachievable by construction and is
explicitly rejected.

The other available reference point is this project's best PRIOR real-STGT-output measurement
from any fine-tuned adapter: **v2 scored 51.8% `accuracy_when_answerable` (pooled) / 68.6%
per-class on `low` threat** on the n=249 GT-determinable real-STGT battery (`AUDIT.md` sec AF).
v2 was trained on a materially smaller, lower-quality corpus than Phase 1's (no comparable
teacher-prose rate, no stratified 49-pair coverage) — v5-a should be expected to do at least as
well, plausibly better, given ~10x more rows, 93.3% teacher-authored prose, full 49-pair
coverage, and a larger LoRA (r=32 vs v2's r=16). This is a reasoned expectation, not a
guarantee, and is stated as such.

| metric | ceiling (unreachable, rejected as a bar) | prior best (v2, real-STGT) | **v5-a preregistered success bar** |
|---|---|---|---|
| pair accuracy, pooled, when-answerable | 77.3% | 51.8% (threat-accuracy proxy, see note) | **>= 55.0%** |
| threat accuracy, `low` (per-class) | — | 68.6% | **>= 65.0%** |
| threat accuracy, pooled, when-answerable | 83.0% | ~51.8% | **>= 55.0%** |
| over-abstention rate | — | — (v2 measured 0.0%) | **<= 15.0%** |
| under-escalation (direction, not magnitude) | — | dominant failure mode across every system measured so far | must remain the LARGER of the two escalation directions, or this itself is a finding worth reporting explicitly, not silently absorbed into one pooled number |
| schema-validity rate | — | — | **>= 95.0%** |

*Note: v2's 51.8% figure in `AUDIT.md` sec AF is reported as pooled `correct` (threat-level
match) in the escalation-direction table, used here as the nearest available reference for
both pair and threat accuracy since a directly comparable pair-accuracy number for v2 on real
STGT output was not separately reported in that session.*

**These bars are deliberately modest relative to the ceiling** — meaningful headroom is
reserved because (a) real STGT output has historically been much harder than the synthetic
`synth_context()` battery every earlier headline number in this project was measured against,
and (b) this is v5-a, the plain-SFT baseline arm of a 4-way ablation — it is not expected to
be the best-performing arm (v5-b/c/d, not yet built, specifically target failure modes v5-a is
not expected to fix: context distillation, class-imbalance-aware loss, and an unbalanced
control for comparison). Clearing these bars is "worth reporting as real progress," not
"the project's final target."

## Explicitly not blocking

This document does not gate whether v5-a training is ALLOWED to run — it gates how the
resulting numbers will be judged once training completes. Nothing here should be edited after
v5-a's real Phase 4 numbers exist; if the protocol needs revision going forward, that is a
new decision for v5-b/c/d, recorded as a new entry, not a silent edit of this one.
