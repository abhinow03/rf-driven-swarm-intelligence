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

## Memorization-vs-generalization check

**The bars above cannot by themselves distinguish "v5-a generalized" from "v5-a memorized its
larger corpus the same way v2 memorized its smaller one."** v2's own historical numbers (100%
in-distribution, 0% abstention, never declines) are consistent with either story — clearing
the bars above alone would not settle which one is true for v5-a, and that ambiguity is the
exact thing this training program exists to resolve. One additional preregistered metric,
decided and committed before any real number exists:

**Metric: verbatim/near-verbatim output-vs-training-target overlap rate.** For every
Phase 4 battery case v5-a answers (non-abstained), compute the TF-IDF cosine similarity
(word 1-2 grams — identical method to `llm_finetuning/report_distinctness_similarity.py`)
between v5-a's generated `situation_summary` and every TRAINING-split row's
`situation_summary` sharing the same `(form_a, form_b)` RULES pair. Take the maximum. The
**overlap rate** is the fraction of answered cases whose max similarity to some same-pair
training target is `>= 0.90`. A model producing genuinely scenario-tailored prose (varying
with the specific velocity/stability/approach-rate/role details of each eval case) should
rarely hit this threshold by chance alone; a model that has memorized associates-pair-with-
canned-prose regardless of the eval scenario's actual details would hit it often.

**Why this metric over the mining-split novel-combination alternative**: both options from
the original instruction were evaluated before committing to one. The alternative
(accuracy specifically on mining-split narrative combinations absent from training for the
same pair) was computed directly on the real corpus: of 600 mining rows, only **27 (4.5%)**
have a `(velocity_trend, stability_trend, spread_trend, role_differentiation)` combination
not already present in training for their pair — and critical-tier contributes only **1**
such row. That subgroup is too thin to support a reliable comparison (the same low-n problem
this project has hit before with the critical tier specifically) and was rejected in favor of
the overlap-rate metric, which uses the FULL answered-case sample, not a ~4.5% subgroup, and
needs no new held-out split or model retraining to compute.

**Preregistered null-hypothesis baseline (measured now, on the real corpus, before v5-a
exists)**: among TEACHER-authored training rows sharing the same pair — text that was
independently generated for genuinely different scenarios, never trained on anything, so any
similarity between them is pure narrative/vocabulary coincidence, not memorization — the
same-pair near-duplicate rate at `>= 0.90` cosine similarity is **0.6% (56/10,080)**. This is
the "chance" rate two unrelated same-pair descriptions hit this threshold.

**Preregistered interpretation bar**: v5-a's overlap rate should be compared against this
0.6% chance baseline, not judged in isolation.
- **Overlap rate not meaningfully above ~0.6-2%**: consistent with generalization — the model
  is producing scenario-specific prose, matching training text only as often as two
  independently-written same-pair descriptions coincidentally would.
- **Overlap rate substantially higher (the specific threshold this document commits to:
  `>= 15%`)**: treated as a positive memorization signal, reported explicitly as such
  regardless of whether the accuracy bars above are also cleared — clearing the accuracy bars
  while failing this check would mean v5-a is accurate BECAUSE it memorized, not because it
  generalized, and that distinction must be reported, not absorbed into a single pass/fail
  headline.

## Explicitly not blocking

This document does not gate whether v5-a training is ALLOWED to run — it gates how the
resulting numbers will be judged once training completes. Nothing here should be edited after
v5-a's real Phase 4 numbers exist; if the protocol needs revision going forward, that is a
new decision for v5-b/c/d, recorded as a new entry, not a silent edit of this one.

## Erratum — 2026-08-12, post-hoc hardening audit

Per this document's own rule above ("nothing here should be edited after real numbers exist"),
this is an APPEND, not a rewrite — the original bars and their PASS results stand unedited.
Two weaknesses in the original bars were identified after v5-a's real numbers existed, during
a dedicated hardening audit. Recorded here with the reasoning, dated, as a provenance record.

**(a) `over_abstention_rate` is unfalsifiable at 0% abstention.** The bar (`<=15.0%`) only
measures incorrectly abstaining on ANSWERABLE cases. v5-a scored 0.2% — an excellent number by
the letter of the bar — but this bar structurally cannot detect the OPPOSITE failure: v5-a's
real `abstention_rate_when_unanswerable` came back at **0.0%** (n=502) — it never correctly
abstains on genuinely unanswerable multi-hop cases either. A system that NEVER abstains,
under any circumstance, clears `over_abstention_rate` trivially and perfectly every time,
regardless of whether its abstention behavior is good, bad, or entirely absent. **This bar
needs a paired under-abstention bar (e.g. `abstention_rate_when_unanswerable` held to some
floor, not just a ceiling on the other direction) to be meaningful** — as written, a system
that always answers and one that abstains exactly when it should are indistinguishable by this
bar alone.

**(b) `escalation_direction` as written encodes a pre-existing finding and cannot fail.** The
original bar text: *"must remain the LARGER of the two escalation directions"* — phrased
relative to under-escalation's own already-documented status as *"the dominant failure mode
across every system measured so far"* (a fact established well before this document was
written, e.g. `AUDIT.md` sec AC/AD). A bar defined as "matches the pattern we already know
holds" is not a real constraint; it is very hard to fail given the pattern's history is
consistent and strong. v5-a's real result (under=18.7%, over=5.6%) did clear this bar, and
that PASS stands — but the bar's design should be acknowledged as weak evidence rather than
a strong pass.

**Correction to a specific claim considered during this audit**: an earlier draft of this
erratum was asked to record that *"the earlier draft's numeric under-escalation `<=10%` bar
would have FAILED at 18.7%."* **Checked directly against `docs/PREREGISTRATION.md`'s full git
history (both commits: creation and the memorization-check amendment) — no numeric
under-escalation bar was ever drafted or committed.** The escalation bar has only ever been
the qualitative "must remain the larger direction" framing above. Not recording a correction
for a bar that was never written — flagging the discrepancy instead, since the underlying
critique in (b) (the bar is weak/tautological) stands on its own without needing an
invented numeric bar to illustrate it.
