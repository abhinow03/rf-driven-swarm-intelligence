# evaluation/

Raw JSON/text output from the evaluation scripts in `scripts/` and `llm_finetuning/` — every
number quoted in `docs/evaluation.md`, `docs/CEILING.md`, `docs/V5_LOG.md`, and `AUDIT.md`
traces back to a specific file here. Filenames generally match the script that produced them
(e.g. `phase0_ceiling_v5_guardfix.json` is `scripts/phase0_ceiling.py`'s output at the point
in the V5 program right after the ambiguity-guard fix landed).

These are regenerable, not hand-edited — if a number in the docs looks wrong, re-run the
corresponding script (same seed, stated in the script's own docstring) rather than editing
these files directly.
