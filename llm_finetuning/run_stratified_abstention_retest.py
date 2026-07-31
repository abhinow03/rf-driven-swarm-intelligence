"""
Step 2 of the "is the 55-case battery in-distribution?" session (AUDIT.md sec J/K).

The ~100% abstention v3b showed on multi_hop(sev>=3)/terminal_transitioning was
measured on perturbations of only the original 6 base cases (run_degradation_eval.py).
This is the cheap follow-up: stratify a 15-case sample out of the full 55-case
TEST_CASES pool -- across threat level (both critical cases forced in) and formation
family (formation_a diversity) -- run ONLY the multi_hop and terminal_transitioning
axes from degradation.py against those 15 base cases, and see whether the abstention
rate holds on a more diverse sample than the original 6.

NOT the full ~2600-generation battery (all 5 axes x 55 base cases) -- deliberately
scoped down per the user's "cheap version" instruction.

Usage:
    python llm_finetuning/run_stratified_abstention_retest.py --n-runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import GroqClient, LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from degradation import axis_multi_hop, axis_terminal_transitioning, make_llm_battery_run_case  # noqa: E402

SELECTION_SEED = 42
N_TARGET = 15


def stratified_sample(test_cases: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by expected_threat (forcing in every 'critical' case first, then
    filling the remainder proportional to each stratum's share of the non-critical
    pool), and within each stratum prefer formation_a diversity (round-robin across
    distinct formation_a values) so the 15-case sample doesn't collapse onto a
    handful of formation families."""
    rng = random.Random(seed)
    by_threat: dict[str, list[dict]] = {}
    for c in test_cases:
        by_threat.setdefault(c["expected_threat"], []).append(c)
    for cases in by_threat.values():
        rng.shuffle(cases)

    selected: list[dict] = []
    critical = by_threat.pop("critical", [])
    selected.extend(critical)  # "include both critical cases"

    remaining_slots = n - len(selected)
    other_strata = list(by_threat.items())
    total_other = sum(len(cases) for _, cases in other_strata)
    quota = {threat: round(len(cases) / total_other * remaining_slots) for threat, cases in other_strata}
    # rounding can over/under-shoot by 1-2; fix up against the largest stratum
    while sum(quota.values()) != remaining_slots:
        threat = max(quota, key=lambda t: len(by_threat[t]))
        quota[threat] += 1 if sum(quota.values()) < remaining_slots else -1

    def by_formation_a_diversity(cases: list[dict], k: int) -> list[dict]:
        buckets: dict[str, list[dict]] = {}
        for c in cases:
            buckets.setdefault(c["formation_a"], []).append(c)
        for b in buckets.values():
            rng.shuffle(b)
        order = list(buckets.keys())
        rng.shuffle(order)
        picked = []
        i = 0
        while len(picked) < k and any(buckets.values()):
            fam = order[i % len(order)]
            if buckets[fam]:
                picked.append(buckets[fam].pop())
            i += 1
        return picked

    for threat, k in quota.items():
        selected.extend(by_formation_a_diversity(by_threat[threat], k))

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-swarm-v3b")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "evaluation" / "stratified_abstention_v3b.json"))
    ap.add_argument("--rate-hint", type=float, default=0.25,
                    help="generations/sec used for the upfront runtime estimate")
    args = ap.parse_args()

    sample = stratified_sample(TEST_CASES, N_TARGET, SELECTION_SEED)
    assert len(sample) == N_TARGET
    threat_dist = Counter(c["expected_threat"] for c in sample)
    family_dist = Counter(c["formation_a"] for c in sample)
    print(f"=== stratified 15-case sample ===")
    print(f"threat distribution: {dict(threat_dist)} (both critical cases included: "
          f"{[c['name'] for c in sample if c['expected_threat'] == 'critical']})")
    print(f"formation_a family coverage: {len(family_dist)}/7 distinct "
          f"({dict(family_dist)})")
    print(f"cases: {[c['name'] for c in sample]}")

    battery = {
        "multi_hop": axis_multi_hop(sample),
        "terminal_transitioning": axis_terminal_transitioning(sample),
    }
    n_cases = sum(len(cases) for cases in battery.values())
    total_generations = n_cases * args.n_runs
    print(f"\n=== multi_hop: {len(battery['multi_hop'])} cases "
          f"(3 severities x {N_TARGET} base), "
          f"terminal_transitioning: {len(battery['terminal_transitioning'])} cases "
          f"(3 severities x {N_TARGET} base) ===")
    print(f"total generations: {n_cases} cases x {args.n_runs} runs = {total_generations} "
          f"(single system: {args.adapter})")
    print(f"estimated runtime at --rate-hint={args.rate_hint}/s: "
          f"~{total_generations / args.rate_hint / 60:.1f} minutes")

    reporter = Reporter("run_stratified_abstention_retest", total_generations, rate_hint=args.rate_hint)

    judge = None
    if os.environ.get("GROQ_API_KEY"):
        judge = GroqClient(model="llama-3.3-70b-versatile")
        print("judge: llama-3.3-70b-versatile (advisory only)")
    else:
        print("GROQ_API_KEY not set — running WITHOUT a judge; objective metrics unaffected")

    client = LocalHFClient(args.base, adapter_path=str(REPO / args.adapter), temperature=0.3)
    run_case = make_llm_battery_run_case(client)

    results = {}
    for axis, cases in battery.items():
        res = evaluate_llm(run_case, cases, judge_client=judge, n_runs=args.n_runs,
                           progress_reporter=reporter)
        results[axis] = res

    reporter.status = "done"
    reporter._write()

    out = {
        "n_runs": args.n_runs,
        "sample_names": [c["name"] for c in sample],
        "threat_distribution": dict(threat_dist),
        "formation_a_family_coverage": dict(family_dist),
        "axes": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n\n=== STRATIFIED ABSTENTION RE-TEST (qwen-swarm-v3b, n_runs={args.n_runs}, "
          f"{N_TARGET} stratified base cases) ===")
    print("| axis | abstention_rate_when_unanswerable (mean +/- std across runs) | over_abstention_rate |")
    print("|---|---|---|")
    for axis, res in results.items():
        agg = res["aggregate"]
        mean = agg.get("abstention_rate_when_unanswerable_mean_across_runs")
        std = agg.get("abstention_rate_when_unanswerable_std_across_runs")
        abst_str = f"{mean:.2%} +/- {std:.2%}" if mean is not None else "N/A"
        overabst = agg.get("over_abstention_rate")
        overabst_str = f"{overabst:.2%}" if overabst is not None else "N/A"
        print(f"| {axis} | {abst_str} | {overabst_str} |")

    print("\nCompare to the original 6-base-case degradation battery (evaluation/degradation_v3b.json): "
          "multi_hop sev>=3 abstain=100%, terminal_transitioning all severities abstain=100%, "
          "over_abstention=0% throughout.")


if __name__ == "__main__":
    main()
