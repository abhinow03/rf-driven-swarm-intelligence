"""
Phase 3a step 3 (docs/PHASE3A_ABSTENTION_CORPUS.md): generates the new abstention-training
corpus. For each mechanism (multi_hop / oscillation / terminal_transitioning):

  1. Sample a TRUE formation chain for that mechanism (this IS the ground truth -- the
     chain is chosen, not inferred, so there is nothing to get wrong about it).
  2. Run it through the REAL simulator (swarm_intent.eval_trajectories /
     swarm_intent.data.generate_transition_sequence) to get true_labels -- genuine
     simulator output, never an STGT read.
  3. Verify unanswerability with step 1's classify_trajectory_ground_truth() (never STGT).
  4. Render a tactical-context string in the SAME narrative vocabulary/grammar Phase 1's
     synth_context() and llm_finetuning/degradation.py's _render_lines() use (reused
     structural rendering, real-population-calibrated narrative field sampling), so the
     student sees in-distribution input shape.
  5. Build the row's target: threat_level="unknown"/likely_intent="unknown"/
     recommended_action="monitor" (schema-identical to pipeline_v2._layer2_abstain),
     stating the ground-truth mechanism reason in threat_reasoning -- optionally polished
     by the NVIDIA teacher (--no-teacher runs with rule-clean templated prose only, same
     fallback convention build_sft_dataset.py already supports). The ANSWERABILITY LABEL
     itself never comes from the teacher.

Usage:
    python llm_finetuning/build_abstention_corpus.py --out data/abstention_corpus.jsonl
    python llm_finetuning/build_abstention_corpus.py --no-teacher --out data/abstention_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from swarm_intent.config import BASE_FORMATIONS, TRANSITION_CLASS
from swarm_intent.inference import build_llm_prompt
from swarm_intent import context_spec as spec
from swarm_intent.eval_trajectories import build_long_sequence_labeled
from swarm_intent.ground_truth_abstention import (
    classify_trajectory_ground_truth, MULTI_HOP, OSCILLATION, TERMINAL_TRANSITIONING,
)

from build_sft_dataset import _sample_real  # noqa: E402  (real-population-calibrated narrative sampler)

STRATA_TARGETS = {MULTI_HOP: 780, OSCILLATION: 120, TERMINAL_TRANSITIONING: 100}

REASON_TEXT = {
    MULTI_HOP: "the observation structurally spans {n} distinct formations ({chain}), which "
              "no single (from, to) rule-table entry can represent",
    OSCILLATION: "the observation returns to an earlier formation ({chain}), an oscillation "
                "pattern no single (from, to) rule-table entry can represent",
    TERMINAL_TRANSITIONING: "the observation window ended while the swarm was still mid-"
                           "transition ({chain}), so the destination formation was never "
                           "confirmed within the observed window",
}


# ---------------------------------------------------------------------------
# Chain samplers -- these DEFINE the ground truth (the chain is chosen, not inferred),
# step 1's classifier is run afterward purely as a build-then-verify check.
# ---------------------------------------------------------------------------

def _extend_chain(chain: list[str], rng: random.Random, avoid_last_repeat: bool = True) -> str:
    pool = [f for f in BASE_FORMATIONS if f != chain[-1]] if avoid_last_repeat else list(BASE_FORMATIONS)
    return rng.choice(pool)


def sample_multi_hop_chain(rng: random.Random) -> list[str]:
    length = rng.choice([3, 3, 3, 4])  # weight toward the dominant 3-hop shape
    chain = [rng.choice(BASE_FORMATIONS)]
    while len(chain) < length:
        chain.append(_extend_chain(chain, rng))
    if chain[0] == chain[-1]:  # would accidentally be an oscillation -- resample the endpoint
        pool = [f for f in BASE_FORMATIONS if f not in (chain[-2], chain[0])]
        chain[-1] = rng.choice(pool)
    return chain


def sample_oscillation_chain(rng: random.Random) -> list[str]:
    length = rng.choice([3, 3, 3, 4])
    chain = [rng.choice(BASE_FORMATIONS)]
    while len(chain) < length - 1:
        chain.append(_extend_chain(chain, rng))
    chain.append(chain[0])  # force the return to the starting formation
    return chain


def sample_terminal_transitioning_known(rng: random.Random) -> list[str]:
    """The known (fully resolved) prefix before the truncated, unresolved final hop."""
    severity = rng.choice([1, 1, 2])  # weight toward the simpler single-formation case
    known = [rng.choice(BASE_FORMATIONS)]
    if severity == 2:
        known.append(_extend_chain(known, rng))
    return known


# ---------------------------------------------------------------------------
# Real-simulator grounding
# ---------------------------------------------------------------------------

def simulate_multi_hop_or_oscillation(chain: list[str], np_rng: np.random.Generator):
    spread = float(np_rng.uniform(0.6, 1.8))
    noise_std = float(np_rng.uniform(0.15, 1.4))
    long_seq, true_labels = build_long_sequence_labeled(chain, np_rng, spread, noise_std)
    return true_labels


def simulate_terminal_transitioning(known: list[str], np_rng: np.random.Generator):
    """Appends one more (unobserved-destination) hop, then truncates strictly inside its
    blend region -- the destination formation genuinely never gets confirmed."""
    pool = [f for f in BASE_FORMATIONS if f != known[-1]]
    next_formation = str(np_rng.choice(pool))
    full_chain = known + [next_formation]
    spread = float(np_rng.uniform(0.6, 1.8))
    noise_std = float(np_rng.uniform(0.15, 1.4))
    long_seq, true_labels = build_long_sequence_labeled(full_chain, np_rng, spread, noise_std)

    runs, start = [], None
    for i, lab in enumerate(true_labels):
        if lab == TRANSITION_CLASS:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(true_labels)))
    if not runs:
        return None  # defensive -- a 2+-hop chain always has at least one blend region
    last_start, last_end = runs[-1]
    cut = last_start + 1 if last_end - last_start < 2 else int(np_rng.integers(last_start + 1, last_end))
    return true_labels[:cut]


# ---------------------------------------------------------------------------
# Context rendering -- same structural grammar as degradation.py's _render_lines() /
# build_sft_dataset.py's synth_context(), real-population-calibrated narrative sampling.
# ---------------------------------------------------------------------------

def render_abstention_context(chain: list[str], mechanism: str, py_rng: random.Random) -> tuple[str, list]:
    mean_conf = round(py_rng.uniform(0.55, 0.9), 2)  # lower ceiling than answerable rows --
    # genuinely harder/noisier observations, not an artificial abstention "tell"
    mean_stab_draw = _sample_real(py_rng, "stability")
    delta_stab_draw = _sample_real(py_rng, "delta_stability")
    stab_early = round(min(1.0, max(0.0, mean_stab_draw - delta_stab_draw / 2)), 2)
    stab_late = round(min(1.0, max(0.0, mean_stab_draw + delta_stab_draw / 2)), 2)
    mean_stab = round((stab_early + stab_late) / 2, 2)
    approach = round(_sample_real(py_rng, "approach_rate"), 3)
    delta_v = round(_sample_real(py_rng, "delta_v_physical"), 2)
    vel_word = (spec.VELOCITY_ACCELERATING if delta_v > 0.5
               else spec.VELOCITY_DECELERATING if delta_v < -0.5 else spec.VELOCITY_STEADY)
    stab_word = (spec.STABILITY_DEGRADING if stab_late < stab_early - 0.1
                else spec.STABILITY_IMPROVING if stab_late > stab_early + 0.1 else spec.STABILITY_HOLDING)
    spread_word = (spec.SPREAD_CONVERGING if approach < -0.1
                  else spec.SPREAD_DISPERSING if approach > 0.1 else spec.SPREAD_STABLE)

    dominant = chain[0]
    if mechanism == TERMINAL_TRANSITIONING:
        history = (" -> transitioning -> ".join(chain) if len(chain) > 1 else chain[0]) + " -> transitioning"
        n_hops = len(chain) - 1 + 1
    else:
        history = " -> transitioning -> ".join(chain)
        n_hops = len(chain) - 1

    lines = [
        "Observation window: 0.0s - 60.0s (9 overlapping windows)",
        f"Dominant formation: {dominant}",
        f"Formation history: {history}",
        f"Multiple formation changes detected across the observation window "
        f"({n_hops} transitions; see Formation history).",
        f"Velocity trend: {vel_word} (delta_v={delta_v:+.2f})",
        f"Formation stability: {stab_word} (mean={mean_stab:.2f})",
        f"Spread dynamics: {spread_word} (mean approach_rate={approach:.3f})",
        f"Role differentiation: {spec.ROLE_DIFFERENTIATION_NOT_PROMINENT}",
        spec.CONFIDENCE_LINE_TEMPLATE.format(mean_conf=mean_conf, low_conf=py_rng.randint(1, 3)),
    ]
    ctx = "\n".join(lines)
    key_windows = [
        {"t": "0.0-25.0s", "formation": chain[0], "confidence": mean_conf,
         "velocity": round(_sample_real(py_rng, "velocity_physical"), 3), "approach": approach,
         "stability": stab_early, "from": None, "to": None, "role_differentiation": False},
        {"t": "35.0-60.0s", "formation": TRANSITION_CLASS,
         "confidence": mean_conf, "velocity": round(_sample_real(py_rng, "velocity_physical"), 3),
         "approach": approach, "stability": stab_late,
         "from": chain[-2] if len(chain) > 1 else chain[0], "to": chain[-1], "role_differentiation": False},
    ]
    return ctx, key_windows


def build_abstention_teacher_prompt(mechanism: str, chain: list[str], ctx: str, prompt: str) -> str:
    reason = REASON_TEXT[mechanism].format(n=len(chain), chain=" -> ".join(chain))
    return (
        f"{prompt}\n\n"
        f"GROUND TRUTH from the canonical rule engine — this observation is NOT resolvable "
        f'to a single rule-table entry: {reason}. Use exactly these values, do not deviate: '
        f'threat_level="unknown", likely_intent="unknown", recommended_action="monitor". '
        f"Write situation_summary, threat_reasoning, key_indicators and follow_up_watch that "
        f"genuinely explain WHY this observation cannot be resolved -- do not guess a specific "
        f"threat level or intent."
    )


def finalize_abstention_assessment(mechanism: str, chain: list[str], draft) -> tuple[dict, bool]:
    reason = REASON_TEXT[mechanism].format(n=len(chain), chain=" -> ".join(chain))
    if not isinstance(draft, dict) or "error" in draft:
        draft = {}
    used_teacher = bool(draft.get("situation_summary"))
    return {
        "situation_summary": draft.get("situation_summary",
            f"The swarm's observed formation history ({' -> '.join(chain)}) cannot be "
            f"resolved to a single tactical assessment."),
        "threat_level": "unknown",
        "threat_reasoning": draft.get("threat_reasoning", f"Insufficient evidence: {reason}."),
        "likely_intent": "unknown",
        "recommended_action": "monitor",
        "confidence_in_assessment": "low",
        "key_indicators": draft.get("key_indicators", [reason]),
        "follow_up_watch": draft.get("follow_up_watch",
            "Re-acquire a higher-confidence, unambiguous observation window."),
    }, used_teacher


def generate_mechanism_rows(mechanism: str, n: int, py_rng: random.Random, np_rng: np.random.Generator,
                            teacher, batch_size: int) -> list[dict]:
    chains, ctxs, prompts = [], [], []
    n_verify_fail = 0
    attempts = 0
    while len(chains) < n:
        attempts += 1
        if attempts > n * 20:
            raise RuntimeError(f"{mechanism}: giving up after {attempts} attempts, only "
                              f"{len(chains)}/{n} verified -- sampler or verifier is broken")
        if mechanism == MULTI_HOP:
            chain = sample_multi_hop_chain(py_rng)
            true_labels = simulate_multi_hop_or_oscillation(chain, np_rng)
        elif mechanism == OSCILLATION:
            chain = sample_oscillation_chain(py_rng)
            true_labels = simulate_multi_hop_or_oscillation(chain, np_rng)
        else:
            known = sample_terminal_transitioning_known(py_rng)
            true_labels = simulate_terminal_transitioning(known, np_rng)
            chain = known
            if true_labels is None:
                n_verify_fail += 1
                continue

        gt = classify_trajectory_ground_truth(chain, true_labels)
        if gt != mechanism:
            n_verify_fail += 1
            continue  # step 1's verifier rejected this sample -- resample, never force-label

        ctx, key_windows = render_abstention_context(chain, mechanism, py_rng)
        prompt = build_llm_prompt(
            predictions=[{**kw, "time_start_s": 0, "time_end_s": 0,
                          "formation_type": kw["formation"], "centroid_velocity": kw["velocity"],
                          "approach_rate": kw["approach"], "formation_stability": kw["stability"],
                          "formation_confidence": kw["confidence"],
                          "role_differentiation": kw["role_differentiation"],
                          "transition_from": kw["from"], "transition_to": kw["to"]} for kw in key_windows],
            tactical_context=ctx, summary={})
        chains.append(chain)
        ctxs.append(ctx)
        prompts.append(prompt)

    print(f"  {mechanism}: {len(chains)}/{n} generated ({n_verify_fail} resampled after failing "
         f"step 1's verifier)")

    rows = []
    for chunk_start in range(0, len(chains), batch_size):
        chunk_chains = chains[chunk_start:chunk_start + batch_size]
        chunk_ctxs = ctxs[chunk_start:chunk_start + batch_size]
        chunk_prompts = prompts[chunk_start:chunk_start + batch_size]
        if teacher is not None:
            teacher_prompts = [build_abstention_teacher_prompt(mechanism, c, ctx, p)
                              for c, ctx, p in zip(chunk_chains, chunk_ctxs, chunk_prompts)]
            drafts = teacher.complete_batch(teacher_prompts, batch_size=batch_size)
        else:
            drafts = [None] * len(chunk_chains)
        for chain, prompt, draft in zip(chunk_chains, chunk_prompts, drafts):
            gold, used_teacher = finalize_abstention_assessment(mechanism, chain, draft)
            rows.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(gold, indent=2)},
            ], "_mechanism": mechanism, "_chain": chain, "_used_teacher": used_teacher})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/abstention_corpus.jsonl")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4242)  # disjoint from Phase 1's seed=42
    args = ap.parse_args()

    teacher = None
    if not args.no_teacher:
        from swarm_intent.llm.client import NvidiaClient
        teacher = (NvidiaClient(model=args.teacher_model, max_tokens=3072) if args.teacher_model
                  else NvidiaClient(max_tokens=3072))

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    all_rows = []
    t0 = time.monotonic()
    for mechanism, n in STRATA_TARGETS.items():
        rows = generate_mechanism_rows(mechanism, n, py_rng, np_rng, teacher, args.concurrency)
        all_rows.extend(rows)
        print(f"  elapsed {(time.monotonic()-t0)/60:.1f} min")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")

    meta_path = args.out.replace(".jsonl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_total": len(all_rows),
            "per_mechanism": {m: sum(1 for r in all_rows if r["_mechanism"] == m) for m in STRATA_TARGETS},
            "n_used_teacher": sum(1 for r in all_rows if r["_used_teacher"]),
            "teacher_enabled": teacher is not None,
            "seed": args.seed,
            "rows_detail": [{"mechanism": r["_mechanism"], "chain": r["_chain"],
                             "used_teacher": r["_used_teacher"]} for r in all_rows],
        }, f, indent=2)

    print(f"\nsaved {len(all_rows)} rows to {args.out}")
    print(f"saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
