"""
AUDIT.md sec AF step 4: sub-type breakdown of bucket C with counts, plus a
test of whether a longer observation window or finer stride would resolve a
meaningful share of `terminal_unknown` (61.0% of C, sec AE step 2) -- a cheap
real fix, if so, rather than a permanent structural limit.

Regenerates the IDENTICAL 500 sequences from sec AE step 2 (same seed=0,
same sample_chain draws in the same order), but with an INSTRUMENTED copy of
measure_coverage.build_long_sequence that consumes the RNG in EXACTLY the
same order (same calls, same arguments, nothing added or removed before it)
so the sequences it returns are bit-for-bit identical to last session's --
this instrumentation only additionally RECORDS each hop's (seg_len,
blend_start, blend_end), it does not change what's generated.

Mechanistic sub-classification of `terminal_unknown` cases: for each, compute
the LAST hop's settled_tail = seg_len - blend_end (timesteps of fully-settled
target-formation geometry after the blend completes). If settled_tail <
window_size (50), the final 50-step window is STRUCTURALLY GUARANTEED to
straddle real transition geometry -- "windowing_artifact". If settled_tail >=
window_size, there was plenty of settled time and the model still read
unknown -- "model_uncertainty" (not a windowing artifact; a longer window
would not obviously fix this one).

Then re-runs sliding_window_inference on the SAME in-memory sequences at an
alternative stride (5, half the original) and reports how many ORIGINAL
terminal_unknown cases leave that subtype.

window_size=100 was ALSO attempted and is NOT reported as a result: it
crashes. STGTModel's PositionalEncoding buffer (src/swarm_intent/stgt/model.py)
is registered at construction with `max_len=cfg["max_seq_len"]` (50, baked
into the checkpoint) -- `x + self.pe[:, :x.size(1), :]` breaks the moment a
window has more than 50 timesteps (shape mismatch, not a soft degradation).
window_size is therefore NOT a free inference-time knob for this checkpoint
the way stride is: "a longer observation window" is not a cheap fix, it
requires retraining STGT with a larger `max_seq_len` -- a materially bigger
undertaking than a stride change, and itself a finding worth reporting, not
silently worked around.

Usage (run inside tmux):
    python llm_finetuning/analyze_bucket_c_windowing.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.coverage import classify_observation, BUCKET_C  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
from swarm_intent.data import generate_transition_sequence  # noqa: E402

from measure_coverage import sample_chain, DATA_DIR, CHECKPOINT  # noqa: E402

N_SEQUENCES = 500
SEED = 0
ORIGINAL_WINDOW_SIZE, ORIGINAL_STRIDE = 50, 10


def build_long_sequence_instrumented(chain, rng, spread, noise_std):
    """Byte-for-byte the same RNG-consuming calls, in the same order, as
    measure_coverage.build_long_sequence -- see module docstring. Returns
    (sequence, hop_metadata) instead of just sequence."""
    segments, hop_meta = [], []
    if len(chain) == 1:
        seg_len = int(rng.integers(50, 101))
        seg = generate_transition_sequence(chain[0], chain[0], n_timesteps=seg_len,
                                           spread=spread, noise_std=noise_std, rng=rng)
        segments.append(seg)
        hop_meta.append({"seg_len": seg_len, "blend_start": None, "blend_end": None, "steady": True})
    else:
        for i in range(len(chain) - 1):
            seg_len = int(rng.integers(50, 101))
            blend_start = int(seg_len * rng.uniform(0.3, 0.5))
            blend_end = int(seg_len * rng.uniform(0.55, 0.75))
            seg = generate_transition_sequence(chain[i], chain[i + 1], n_timesteps=seg_len,
                                               spread=spread, noise_std=noise_std,
                                               blend_start=blend_start, blend_end=blend_end, rng=rng)
            segments.append(seg)
            hop_meta.append({"seg_len": seg_len, "blend_start": blend_start, "blend_end": blend_end,
                             "steady": False})

    stitched = [segments[0]]
    for seg in segments[1:]:
        prev_last_centroid = stitched[-1][-1].mean(axis=0)
        this_first_centroid = seg[0].mean(axis=0)
        delta = prev_last_centroid - this_first_centroid
        stitched.append(seg + delta[None, None, :])
    return np.concatenate(stitched, axis=0), hop_meta


def main():
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

    rng = np.random.default_rng(SEED)
    reporter = Reporter("analyze_bucket_c_windowing", N_SEQUENCES, rate_hint=6.0)

    records = []
    for i in range(N_SEQUENCES):
        chain = sample_chain(rng)
        spread = float(rng.uniform(0.6, 1.8))
        noise_std = float(rng.uniform(0.15, 1.4))
        long_seq, hop_meta = build_long_sequence_instrumented(chain, rng, spread, noise_std)

        preds_orig = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                              train_mean, train_std, window_size=ORIGINAL_WINDOW_SIZE,
                                              stride=ORIGINAL_STRIDE, dt=0.5)
        result_orig = classify_observation(preds_orig)

        record = {"i": i, "true_chain": [str(f) for f in chain], "n_timesteps": int(long_seq.shape[0]),
                  "bucket_orig": result_orig["bucket"], "subtype_orig": result_orig["subtype"],
                  "last_hop": hop_meta[-1]}

        if result_orig["bucket"] == BUCKET_C and result_orig["subtype"] == "terminal_unknown":
            last = hop_meta[-1]
            if last["steady"]:
                record["mechanism"] = "model_uncertainty"  # no blend at all involved
            else:
                settled_tail = last["seg_len"] - last["blend_end"]
                record["settled_tail"] = settled_tail
                record["mechanism"] = ("windowing_artifact" if settled_tail < ORIGINAL_WINDOW_SIZE
                                       else "model_uncertainty")

            # window_size=100 is NOT tested here -- STGTModel's PositionalEncoding
            # buffer is fixed at max_len=50 (baked into the checkpoint); feeding it
            # a >50-timestep window is an architecture-level shape mismatch, not a
            # runtime knob. See module docstring.

            # alternative config: stride=5 (always valid, window_size unchanged)
            preds_c = sliding_window_inference(model, long_seq, ckpt["cfg"], reg_mean, reg_std,
                                               train_mean, train_std, window_size=ORIGINAL_WINDOW_SIZE,
                                               stride=5, dt=0.5)
            result_c = classify_observation(preds_c)
            record["alt_stride5"] = {"bucket": result_c["bucket"], "subtype": result_c["subtype"]}

        records.append(record)
        reporter.update(1, item=f"seq {i}")

    reporter.status = "done"
    reporter._write()

    out_path = REPO / "evaluation" / "bucket_c_windowing_analysis.json"
    out_path.write_text(json.dumps(records, indent=2))
    print(f"\nsaved {out_path}")

    # ================= REPORTING =================
    bucket_c = [r for r in records if r["bucket_orig"] == BUCKET_C]
    print(f"\n=== bucket C sub-type breakdown, n={len(bucket_c)} (cross-check vs sec AE step 2) ===")
    subtype_counts = Counter(r["subtype_orig"] for r in bucket_c)
    for subtype, k in subtype_counts.most_common():
        print(f"  {subtype}: {k}/{len(bucket_c)} ({k/len(bucket_c):.1%} of C, {k/N_SEQUENCES:.1%} of total)")

    tu = [r for r in records if r["subtype_orig"] == "terminal_unknown"]
    print(f"\n=== terminal_unknown mechanistic breakdown, n={len(tu)} ===")
    mech_counts = Counter(r["mechanism"] for r in tu)
    for mech, k in mech_counts.most_common():
        print(f"  {mech}: {k}/{len(tu)} ({k/len(tu):.1%})")

    print(f"\n=== window_size=100 was NOT tested: architecturally infeasible for this checkpoint ===")
    print(f"  STGTModel's PositionalEncoding buffer is fixed at max_len=50 (baked into "
         f"swarm_data/best_model.pt at training time). A >50-timestep window is a shape "
         f"mismatch, not a soft degradation -- confirmed by an actual crash on first attempt. "
         f"A longer observation window is NOT a cheap inference-time fix; it requires "
         f"retraining STGT with a larger max_seq_len.")

    print(f"\n=== alternative stride=5 (window_size unchanged at 50) ===")
    resolved_stride = [r for r in tu if not (r["alt_stride5"]["bucket"] == "C"
                                             and r["alt_stride5"]["subtype"] == "terminal_unknown")]
    print(f"  no longer terminal_unknown under stride=5: {len(resolved_stride)}/{len(tu)} "
         f"({len(resolved_stride)/len(tu):.1%})")
    new_bucket_counts_stride = Counter(r["alt_stride5"]["bucket"] for r in tu)
    print(f"  new bucket distribution: {dict(new_bucket_counts_stride)}")


if __name__ == "__main__":
    main()
