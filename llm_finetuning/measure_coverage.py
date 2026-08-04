"""
AUDIT.md sec AE step 2: the mathematical defense for "why not just a dict."

Generates ~500 long, varied swarm trajectories directly from data_gen's
generate_transition_sequence (NOT the templated synth_context() narrative
generator every other eval script in this project uses -- this measurement is
about what the REAL trained STGT (swarm_data/best_model.pt) actually outputs
on real geometry, not about a templated text proxy for it), runs the real
swarm_intent.stgt.inference.sliding_window_inference over each, bridges the
per-window predictions through stgt_bridge.bridge_predictions, and classifies
the result into exactly one of three buckets via src/swarm_intent/coverage.py:

  A. RESOLVABLE  -- reduces to one unambiguous (from, to) pair in RULES
  B. GUARDABLE   -- OOV name / dominant-history contradiction / dispersed-vs-
                    converging near-tie / all windows low-confidence
  C. UNRESOLVABLE -- multi-hop chain, terminal transitioning, oscillation, or
                     any pattern with no RULES key

Bucket A is what a 49-entry dict can answer on its own; B+C is what it
structurally cannot (deliberately not-guessing on B, no key to look up at all
on C). This single measured split is the Q1 defense for why pipeline_v2 keeps
an LLM layer at all instead of shipping the dict.

Sampling regime (fixed up front, NOT tuned against the resulting split --
see the "do not tune" instruction this script exists under):
  - num_formations ~ Uniform{1,2,3,4} (chain length: 1=steady single
    formation, 2=a single resolvable transition, 3-4=multi-hop/oscillation
    by construction of an unconstrained random walk, not injected)
  - each next formation ~ Uniform(BASE_FORMATIONS \\ {previous}) -- a random
    walk that only forbids an immediate self-repeat (a genuine "no change"
    is instead expressed by num_formations=1); oscillation (A->B->A) and
    multi-hop (A->B->C) both emerge naturally from this walk, not from a
    hand-built axis
  - per-hop segment length ~ Uniform{50..100} timesteps
  - spread ~ Uniform(0.6, 1.8), noise_std ~ Uniform(0.15, 1.4) per sequence
  - consecutive hop segments are RIGID-TRANSLATED (not regenerated) so each
    hop's starting centroid exactly matches the previous hop's ending
    centroid -- generate_transition_sequence always starts a fresh call at
    centroid=[0,0,100] with a fresh random velocity, so naive concatenation
    would teleport the swarm at every hop boundary and inject spurious
    boundary-window noise unrelated to genuine model uncertainty; the rigid
    translation preserves every hop's own internally-generated geometry/
    velocity/noise exactly, it only repositions it in world space.

Usage (run inside tmux):
    python llm_finetuning/measure_coverage.py --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from swarm_intent.data import generate_transition_sequence  # noqa: E402
from swarm_intent.coverage import classify_observation, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

DATA_DIR = REPO / "swarm_data"
CHECKPOINT = DATA_DIR / "best_model.pt"


def wilson_ci95(k: int, n: int):
    """Wilson score interval, more reliable than a normal approximation for
    proportions near 0/1 or with modest n (some C sub-types will have small
    counts)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def sample_chain(rng: np.random.Generator) -> list[str]:
    num_formations = int(rng.integers(1, 5))  # {1,2,3,4}, uniform
    chain = [rng.choice(BASE_FORMATIONS)]
    for _ in range(num_formations - 1):
        pool = [f for f in BASE_FORMATIONS if f != chain[-1]]
        chain.append(rng.choice(pool))
    return chain


def build_long_sequence(chain: list[str], rng: np.random.Generator, spread: float, noise_std: float) -> np.ndarray:
    """Concatenate one generate_transition_sequence() call per hop (or one
    steady segment if len(chain)==1), rigid-translating segment i>0 so its
    first-timestep centroid matches segment i-1's last-timestep centroid --
    see module docstring."""
    segments = []
    if len(chain) == 1:
        seg_len = int(rng.integers(50, 101))
        seg = generate_transition_sequence(chain[0], chain[0], n_timesteps=seg_len,
                                           spread=spread, noise_std=noise_std, rng=rng)
        segments.append(seg)
    else:
        for i in range(len(chain) - 1):
            seg_len = int(rng.integers(50, 101))
            blend_start = int(seg_len * rng.uniform(0.3, 0.5))
            blend_end = int(seg_len * rng.uniform(0.55, 0.75))
            seg = generate_transition_sequence(chain[i], chain[i + 1], n_timesteps=seg_len,
                                               spread=spread, noise_std=noise_std,
                                               blend_start=blend_start, blend_end=blend_end, rng=rng)
            segments.append(seg)

    stitched = [segments[0]]
    for seg in segments[1:]:
        prev_last_centroid = stitched[-1][-1].mean(axis=0)
        this_first_centroid = seg[0].mean(axis=0)
        delta = prev_last_centroid - this_first_centroid
        stitched.append(seg + delta[None, None, :])
    return np.concatenate(stitched, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "coverage_measurement.json"))
    args = ap.parse_args()

    import torch
    from swarm_intent.stgt.config import device
    from swarm_intent.stgt.model import STGTModel
    from swarm_intent.stgt.inference import sliding_window_inference

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = STGTModel(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_mean = np.load(DATA_DIR / "train_mean.npy")
    train_std = np.load(DATA_DIR / "train_std.npy")
    reg_mean = np.load(DATA_DIR / "reg_mean.npy")
    reg_std = np.load(DATA_DIR / "reg_std.npy")

    rng = np.random.default_rng(args.seed)
    reporter = Reporter("measure_coverage", args.n, rate_hint=2.0)

    records = []
    for i in range(args.n):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq = build_long_sequence(chain, rng, spread, noise_std)

        predictions = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=50, stride=10, dt=0.5)
        result = classify_observation(predictions)
        records.append({
            "i": i, "true_chain": chain, "spread": spread, "noise_std": noise_std,
            "n_timesteps": int(long_seq.shape[0]), "n_windows": len(predictions),
            "bucket": result["bucket"], "subtype": result["subtype"],
            "rules_key": result["rules_key"], "guard_reasons": result["guard_reasons"],
        })
        reporter.update(1, item=f"sample {i}")

    reporter.status = "done"
    reporter._write()

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"n": args.n, "seed": args.seed, "records": records}, indent=2))
    print(f"\nsaved {out_path}")

    n = len(records)
    bucket_counts = Counter(r["bucket"] for r in records)
    print(f"\n=== step 2: bucket split, n={n} (Wilson 95% CI) ===")
    print("| bucket | n | % | 95% CI |")
    print("|---|---|---|---|")
    for b in (BUCKET_A, BUCKET_B, BUCKET_C):
        k = bucket_counts.get(b, 0)
        p, lo, hi = wilson_ci95(k, n)
        print(f"| {b} | {k} | {p:.1%} | [{lo:.1%}, {hi:.1%}] |")

    c_records = [r for r in records if r["bucket"] == BUCKET_C]
    print(f"\n=== bucket C sub-type breakdown (n={len(c_records)}) ===")
    print("| subtype | n | % of C | % of total (95% CI) |")
    print("|---|---|---|---|")
    subtype_counts = Counter(r["subtype"] for r in c_records)
    for subtype, k in subtype_counts.most_common():
        pct_of_c = k / len(c_records) if c_records else 0.0
        p, lo, hi = wilson_ci95(k, n)
        print(f"| {subtype} | {k} | {pct_of_c:.1%} | {p:.1%} [{lo:.1%}, {hi:.1%}] |")

    b_records = [r for r in records if r["bucket"] == BUCKET_B]
    print(f"\n=== bucket B guard-reason breakdown (n={len(b_records)}, reasons may co-occur) ===")
    reason_counts = Counter(reason for r in b_records for reason in r["guard_reasons"])
    for reason, k in reason_counts.most_common():
        print(f"  {reason}: {k}/{len(b_records)} ({k/len(b_records):.1%} of B)" if b_records else f"  {reason}: 0")

    print(f"\nchain-length distribution (sanity check on the sampling regime): "
         f"{Counter(len(r['true_chain']) for r in records)}")


if __name__ == "__main__":
    main()
