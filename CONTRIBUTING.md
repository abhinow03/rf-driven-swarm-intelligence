# Contributing

This started as a university capstone project; contributions, questions, and issues are
welcome via GitHub issues/PRs.

## Before you start

- Read `CLAUDE.md` first — it's written as agent instructions but doubles as the most
  accurate, current architecture/convention summary in the repo, and states plainly where the
  README and code disagree (trust the code).
- `docs/methodology.md` documents this project's evaluation discipline (independent ground
  truth, independent judge model, dev/mining split hygiene). Any new evaluation code should
  follow it — the project has been burned once by an LLM grading its own output.
- There is no CI configured yet. Run `python -m unittest discover -s tests -q` (140 tests, no
  GPU required) before submitting a change.

## Conventions

- No hidden global state: pass `cfg`, `device`, thresholds, and normalization stats
  explicitly. Do not reintroduce module-level globals.
- A single seeded `np.random.Generator` (from `cfg.seed`) is threaded through all sampling in
  `data.py` — do not call `np.random.*` or a fresh `default_rng()` without it.
- Secrets come from the environment only (`GROQ_API_KEY`). Never commit a key or a
  hardcoded fallback.
- If you tune a threshold against data, tune it on a dedicated, disjoint, single-use dev
  split — never the same split twice, never the eval battery you're reporting results on.
- If a fix doesn't help, or helps less than hoped, or makes something else worse, that's a
  result worth recording, not a reason to omit the attempt. See `docs/experiments.md`'s
  "Abandoned" section for the standard this project tries to hold itself to.

## Reporting issues

Please include: what you ran, the exact command, and (if applicable) which seed/checkpoint —
most of this project's own diagnostic work depended on being able to reproduce a specific
population exactly.
