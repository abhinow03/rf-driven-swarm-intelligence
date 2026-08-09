# Archive

Files here are kept for historical provenance, not because they reflect current project
state. See `docs/experiments.md`'s "Obsolete" section for why each one is here.

- `PROJECT_HANDOFF.md` — an early project-state snapshot, predating the coverage-routing
  (AUDIT.md sec AE onward) and V5 retraining work. Several of its claims about what
  exists/works are stale.
- `handoff_audit_report.md` — a verification report checking `PROJECT_HANDOFF.md`'s claims
  against on-disk artifacts at the time; inherits the same staleness.
- `matrix_early_rules_draft.py` (originally `matrix.py`) — an unreferenced early duplicate of
  the RULES dict, from the initial notebook-to-package migration commit. The canonical RULES
  dict lives in `llm_finetuning/build_sft_dataset.py`.
