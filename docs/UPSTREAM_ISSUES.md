# Upstream issues — requests against components outside `src/swarm_intent/`

Two changes to code owned by other parts of the project, each with a measured downstream cost
and a specific, minimal requested fix. Full derivation for both is in `AUDIT.md` (sections
referenced inline); this document is the standalone ask.

Note on naming: earlier `AUDIT.md` sections (AF/AG) refer to this defect informally as living in
"`data_gen.py`". The actual file, confirmed against the current tree, is
`src/swarm_intent/formations.py` (`get_formation_offsets`), with the time-dependent shrink in
`src/swarm_intent/data.py` (`generate_swarm_sequence`). Cited precisely below.

## 1. `dispersed` and `converging` are geometrically indistinguishable at the offset level

**File / function:** `src/swarm_intent/formations.py`, `get_formation_offsets()`, lines 68–73:

```python
elif formation_type in ("dispersed", "converging"):
    # converging returns the (spread-out) STARTING offsets; the shrink
    # over time is handled in generate_swarm_sequence.
    offsets = rng.uniform(
        low=[-20, -20, -10], high=[20, 20, 10], size=(6, 3)
    ).astype(float)
```

Both formation types draw from the identical `rng.uniform` call — same distribution, same
shape. The only thing that distinguishes them is a per-timestep shrink multiplier applied
downstream in `src/swarm_intent/data.py`, `generate_swarm_sequence()`, lines 54–56:

```python
if formation_type == "converging":
    shrink = 1.0 - (0.9 * t / (n_timesteps - 1))
    offsets = base_offsets * shrink
else:
    offsets = base_offsets
```

Every other formation (`v_shape`, `encirclement`, `column`, `diamond`, `shield`) has a fixed,
deterministic offset template — a genuinely distinct spatial pattern a classifier can learn from
a single snapshot. `dispersed`/`converging` do not: they are the same random point cloud, and
`converging`'s only distinguishing signal is a *temporal derivative* (the cloud shrinking across
timesteps), which is far weaker and noisier than a static shape, especially from any single
50-timestep window that doesn't span enough of the shrink to make the trend clear against
per-timestep positional noise (`noise_std`).

**Measured downstream cost** (`AUDIT.md` sec AF step 3, sec AG step 1):
- STGT's per-window class probabilities for `dispersed` vs `converging` are frequently near-tied
  (the `dispersed_converging_ambiguity` guard in `src/swarm_intent/coverage.py`), which the LLM
  pipeline (`pipeline_v2.py`) correctly refuses to resolve on its own.
- Of 249 real STGT sequences whose generator ground truth is a clean, resolvable `(a, b)`
  formation pair, **115 (46.2%) are blocked from automatic resolution specifically by this
  ambiguity** — the single largest of five diagnosed failure categories (`AUDIT.md` sec AG
  step 1).
- A majority-vote reduction fix was built and tested to route around it (`AUDIT.md` sec AG
  step 2); it does not help, because the ambiguity is a genuine per-window classification
  problem, not a reduction-logic problem — recovered-pair precision never exceeds **49%** at any
  threshold tested, and of the pairs it gets wrong, **63.3% trace directly to this
  dispersed/converging confusion** (the remaining 36.7% is unrelated model error).
- Projected payoff of fixing this alone, no other pipeline change (`AUDIT.md` sec AH step 3,
  single stated assumption — see that section for the full derivation): Layer-1 (deterministic,
  rule-table) firing rate rises from **1.8% to ~12.0%** of real observations, and pipeline
  over-abstention drops from **69.8% to ~49.3%**. This is a conservative floor: it only counts
  the 51/115 sequences where ambiguity is the *sole* blocker, and doesn't include the separate,
  larger, currently-unquantified upside of combining this with the already-implemented (but
  unshipped, for this exact reason) robust reduction fix.

**Requested change:** give `converging` its own base geometry in `get_formation_offsets()`,
distinct from `dispersed`'s wide uniform scatter — a compact cluster (e.g. small-radius points
around the centroid, roughly the same style as `diamond`/`shield`'s fixed templates) that is
visually and statistically separable from `dispersed`'s spread-out scatter *at a single
timestep*, independent of the shrink trend. The shrink-over-time behavior in
`generate_swarm_sequence()` can stay as an additional signal on top of that, but the model
shouldn't have to rely on it alone to tell the two formations apart.

## 2. `max_seq_len=50` is baked into the positional encoding at training time

**File / function:** `src/swarm_intent/config.py` line 58 (`max_seq_len: int = 50`), consumed by
`src/swarm_intent/model.py` lines 62–69 (`PositionalEncoding.__init__`, which allocates a fixed
`pe = torch.zeros(max_len, d_model)` buffer) and line 87
(`self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_seq_len, cfg.dropout)`). This buffer is
saved into `swarm_data/best_model.pt` at training time and is not a runtime-adjustable inference
parameter.

**Measured downstream cost** (`AUDIT.md` sec AF step 4): a sliding-window observation ending
before a formation transition's blend has fully settled reads as `"transitioning"` for its final
window(s), which breaks the current unanimity-based reduction (and, per sec AG step 1, is the
**second-largest** failure category on its own — `trailing_transitioning_run`, **25.7%** of the
249 GT-clean reduction failures). Mechanistically confirmed, not inferred: of 183
`terminal_unknown` bucket-C cases, **159 (86.9%)** have less than a full window's worth of
settled post-transition geometry before the sequence ends — the final window is *structurally
guaranteed* to contain transition geometry, not a classifier failure.

Two cheap fixes were tried and ruled out (sec AF step 4):
- `window_size=100` (a longer observation window) **crashes** — the exact shape mismatch this
  constraint predicts (`x + self.pe[:, :x.size(1), :]`), confirming it is not adjustable without
  retraining.
- `stride=5` (finer sampling) resolves only **18/183 (9.8%)** of `terminal_unknown` cases, since
  a finer stride adds overlapping windows earlier in the sequence but doesn't change the content
  of the *last* window, which is what determines this failure mode.

**Requested change:** not urgent on its own — this is not asking for a retrain solely for this.
But if/when STGT is retrained for any other reason (e.g. after issue #1's geometry fix, which
would itself justify a retrain), please consider increasing `max_seq_len` past 50 at that point,
since `trailing_transitioning_run` at 25.7% of reduction failures is large enough that a longer
window would likely recover a meaningful share of it for free, given the retrain is already
happening. Not requesting a stand-alone retrain for this reason alone.
