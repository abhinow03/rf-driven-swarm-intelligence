# Model / data lineage

The real chain this project's LLM layer went through, one or two lines per step. This
summarizes; it does not duplicate `AUDIT.md`'s full derivations — follow the cited section
letters there for the actual measurements and reasoning behind each entry.

| step | what it was | what changed / what was found | `AUDIT.md` |
|---|---|---|---|
| base Qwen2.5-7B-Instruct | zero-shot, no fine-tuning | establishes the floor; ordinal shrinkage toward `medium` observed even with no fine-tuning at all | sec AD |
| `qwen-swarm-v2` | first QLoRA fine-tune attempt | scored ~100% on its own 55-case battery but the number was recall, not generalization — the model memorized a highly repetitive prompt template rather than learning to reason, and had 0% abstention capability | secs A, B, J |
| `rules_in_prompt` | in-context baseline: give the base model the RULES table directly, no fine-tuning | outperformed early fine-tuned checkpoints on `threat_level` — a real, surprising architecture finding, not a bug: in-context RULES beat fine-tuning until masking/training issues were fixed | secs S, AB |
| `qwen-swarm-v3b` / `v3b-fix` | assistant-only loss masking introduced, then corrected | `assistant_only_loss=True` was found to matter enormously (+13.5pt in-distribution / +21.8pt mean-under-perturbation over the unmasked run) — this became a locked, non-negotiable training setting from here on | secs AA, CC |
| v5-a | first real fine-tune on the full 12,001-row Phase 1 corpus, plain SFT | generalizes (not just recall) — real pair accuracy 63.6%/threat accuracy 78.9% on the current seed=999 population, vs. an 83.0%/77.3% same-population STGT+bridge ceiling. **Abstention gap discovered**: 0.0% correct-abstention on genuinely unanswerable (multi-hop/oscillation) inputs — it always guesses | secs AI, AJ, AK |
| (no-retrain attempt, reverted) | tried routing structurally-ambiguous cases to deterministic abstention without touching the model | failed its own false-positive gate (51/498, 10.2%, would have misrouted genuinely-answerable cases) — reverted before any full eval ran; confirmed a retrain with real abstention examples is required | sec AN |
| Phase 3a abstention corpus | 900 new rows, ground-truth-classifier-labeled (not STGT's own noisy read) `multi_hop`/`oscillation` examples, merged into v5-a's corpus | corpus-only work, no training — v5-a stays the shipped baseline throughout; safety copy made read-only before any of this touched the working checkpoint | secs AO, AP |
| **v5a2** (in progress) | fresh full-corpus QLoRA retrain (not continued-training — see `PREREGISTRATION_V5A2.md` sec 4 for why) on the 12,901-row merged pool, same hyperparameters as v5-a | **training now / not yet scored.** Bars locked before this run started in `docs/PREREGISTRATION_V5A2.md`; do not cite a v5a2 number until that script has actually run against a completed checkpoint | sec AQ |

## A note on the abstention gap specifically

Every step above `qwen-swarm-v2` through v5-a shares one property: **zero abstention
capability**, because none of their training data ever demonstrated declining to answer.
v5a2 is the first attempt in this lineage's history to train on abstention examples at all —
whether it actually closes the gap (vs. e.g. over-correcting into excessive abstention on
answerable cases) is exactly what `docs/PREREGISTRATION_V5A2.md`'s bars d/e/f are designed to
catch, in either direction.
