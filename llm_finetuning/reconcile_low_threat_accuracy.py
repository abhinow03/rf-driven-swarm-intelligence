"""
AUDIT.md sec AC step 1: reconcile three disagreeing rules_in_prompt low-threat
accuracy numbers before trusting any composite comparison built on them --
93.3% (measure_base_rules_prior.py's greedy free-form decode, sec AA step 2),
65.3% (eval_expanded_rules_in_prompt.json, standalone, sampled n_runs=5,
sec AB step 3), 82.3% (eval_expanded_composite.json, same branch, sampled
n_runs=5, sec AB step 3).

Four measurements, same 15 low-threat TEST_CASES, run through the ACTUAL
production code paths (not reimplementations) so results are directly
comparable to what's already on disk:

  a. restricted 4-candidate logit-argmax at the threat_level token, from a
     GREEDY-generated prefix -- reused from evaluation/base_rules_prior.json
     (measure_base_rules_prior.py's `restricted_argmax`), not re-run.
  b. full JSON generation, GREEDY (temperature=0 -> LocalHFClient's
     do_sample=False branch), n_runs=1 -- run fresh through
     baselines.make_rules_in_prompt_run_case + evaluate_llm (the ACTUAL
     production path), not measure_base_rules_prior.py's separate
     reimplementation, specifically to cross-check the two independently.
  c. full JSON generation, SAMPLED (temperature=0.3), n_runs=5 -- the EXACT
     protocol that produced eval_expanded_rules_in_prompt.json.
  d. same as (c) but through composite.make_composite_run_case -- the exact
     protocol that produced eval_expanded_composite.json.

Before running anything, (0) diffs the actual prompt TEXT built by
logit_inspection.py's build_case_prompt (measure_base_rules_prior.py's path)
against baselines.make_rules_in_prompt_run_case's path, byte for byte, for
all 15 low cases, to rule out a prompt-construction divergence as the
explanation before looking anywhere else.

(b)/(c)/(d) all run the FULL 55-case battery (not a skip-optimized subset)
so the shared rng's draw sequence is byte-identical to how the numbers on
disk were produced, and capture every raw generated assessment dict (not
just correct/incorrect) for the 15 low-threat AND all high/critical cases
-- the low-threat data answers step 1's failure-mode question, the
high/critical data answers step 2's confusion-matrix question, both from
the same generations (no extra GPU calls needed for step 2).

Usage (run inside tmux):
    python llm_finetuning/reconcile_low_threat_accuracy.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES, is_abstention, match_threat  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case, load_rules_txt  # noqa: E402
from build_sft_dataset import synth_context  # noqa: E402
from composite import make_composite_run_case  # noqa: E402
from logit_inspection import build_case_prompt  # noqa: E402
from swarm_intent.inference import build_llm_prompt  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V3B_FIX_ADAPTER = "adapters/qwen-swarm-v3b-fix"

LOW_CASES = [c for c in TEST_CASES if c["expected_threat"] == "low"]
HIGH_CRIT_CASES = [c for c in TEST_CASES if c["expected_threat"] in ("high", "critical")]
assert len(LOW_CASES) == 15


def step0_prompt_diff():
    """Byte-for-byte diff of the two prompt-construction paths, same rng
    protocol (walk all 55 TEST_CASES in order, one draw per case -- matches
    n_runs=1 evaluate_llm exactly), for the 15 low-threat cases."""
    print("=== step 0: prompt byte-diff, logit_inspection.build_case_prompt vs "
          "baselines-style synth_context+build_llm_prompt ===")
    rng_a, rng_b = Random(0), Random(0)
    mismatches = []
    for case in TEST_CASES:
        prompt_a = build_case_prompt(case, rng_a)

        ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng_b)
        preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                 "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                 "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                 "role_differentiation": False, "transition_from": kw["from"],
                 "transition_to": kw["to"]} for kw in key_windows]
        prompt_b = build_llm_prompt(preds, ctx, {})

        if case["expected_threat"] == "low" and prompt_a != prompt_b:
            mismatches.append(case["name"])

    if mismatches:
        print(f"  MISMATCHES FOUND on {len(mismatches)}/15 low cases: {mismatches}")
    else:
        print("  Byte-identical on all 15 low-threat cases. Prompt construction is NOT "
              "the source of the discrepancy.")
    return mismatches


def _capturing(run_case, log: dict):
    def wrapped(case):
        assessment, ctx = run_case(case)
        log[case["name"]].append(assessment)
        return assessment, ctx
    return wrapped


def _classify_failure(assessment: dict, expected_threat: str) -> str:
    if "error" in assessment:
        return "json_parse_failure"
    intent = assessment.get("likely_intent", "")
    if is_abstention(intent):
        return "abstained"
    threat = assessment.get("threat_level", "")
    if match_threat(threat, expected_threat):
        return "correct"
    return f"threat_level_diverged (predicted={threat!r})"


def run_labeled(label: str, client, run_case_factory, n_runs: int, reporter: Reporter, out_dir: Path):
    print(f"\n=== {label}: full 55-case battery, n_runs={n_runs} ===")
    log: dict = defaultdict(list)
    run_case = _capturing(run_case_factory(client), log)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=None, n_runs=n_runs,
                       progress_reporter=reporter)
    (out_dir / f"reconcile_{label}.json").write_text(json.dumps(
        {"aggregate": res["aggregate"], "per_case": res["per_case"], "raw": log}, indent=2))

    low_case_results = [c for c in res["per_case"] if c["name"] in {lc["name"] for lc in LOW_CASES}]
    accs = [c["threat_accuracy"] for c in low_case_results if c["threat_accuracy"] is not None]
    import numpy as np
    print(f"  low-threat accuracy: mean={np.mean(accs):.1%} across {len(accs)} cases "
          f"(n_runs={n_runs} each)")
    return res, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mismatches = step0_prompt_diff()

    n_greedy = len(TEST_CASES)
    n_sampled = len(TEST_CASES) * args.n_runs
    total = n_greedy + n_sampled * 2  # b (greedy) + c (sampled standalone) + d (sampled composite)
    reporter = Reporter("reconcile_low_threat", total, rate_hint=0.25)

    rules_client = LocalHFClient(args.base, adapter_path=None, temperature=0.0,
                                 system_prompt=load_rules_txt())

    # --- b. full JSON generation, GREEDY, n_runs=1, production code path ---
    res_b, log_b = run_labeled("b_greedy_standalone", rules_client, make_rules_in_prompt_run_case,
                               1, reporter, out_dir)

    # --- c. full JSON generation, SAMPLED, n_runs=5, production code path ---
    rules_client.temperature = 0.3
    res_c, log_c = run_labeled("c_sampled_standalone", rules_client, make_rules_in_prompt_run_case,
                               args.n_runs, reporter, out_dir)

    # --- d. same, via composite ---
    ft_client = LocalHFClient(args.base, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    branch_log: dict = {}

    def composite_factory(_unused_client):
        return make_composite_run_case(rules_client, ft_client, branch_log, seed=0)

    res_d, log_d = run_labeled("d_sampled_composite", None, composite_factory,
                               args.n_runs, reporter, out_dir)

    reporter.status = "done"
    reporter._write()

    del rules_client, ft_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    # ================= step 1: four-way reconciliation =================
    a_data = json.loads((REPO / "evaluation" / "base_rules_prior.json").read_text())["rules_in_prompt"]
    a_correct = sum(1 for r in a_data.values() if r["threat_level"]["restricted_argmax"] == "low")
    a_acc = a_correct / len(a_data)

    def low_acc(res):
        low_names = {c["name"] for c in LOW_CASES}
        cells = [c["threat_accuracy"] for c in res["per_case"]
                if c["name"] in low_names and c["threat_accuracy"] is not None]
        import numpy as np
        return float(np.mean(cells)), float(np.std(cells)), len(cells)

    b_acc, b_std, b_n = low_acc(res_b)
    c_acc, c_std, c_n = low_acc(res_c)
    d_acc, d_std, d_n = low_acc(res_d)

    print("\n\n=== STEP 1: FOUR-WAY RECONCILIATION (low-threat, n=15 cases) ===")
    print("| measurement | protocol | accuracy | std (per-case, across n_runs) |")
    print("|---|---|---|---|")
    print(f"| a. logit-argmax (restricted, greedy prefix) | n_runs=1, deterministic | {a_acc:.1%} | n/a |")
    print(f"| b. full JSON, greedy | n_runs=1, deterministic | {b_acc:.1%} | n/a |")
    print(f"| c. full JSON, sampled, standalone | n_runs={args.n_runs}, temp=0.3 | {c_acc:.1%} | {c_std:.1%} |")
    print(f"| d. full JSON, sampled, composite | n_runs={args.n_runs}, temp=0.3 | {d_acc:.1%} | {d_std:.1%} |")

    # ---- failure-mode breakdown: cases where (b) greedy was right but a sampled run was wrong ----
    print("\n=== failure-mode breakdown: sampled runs wrong where GREEDY (b) was right ===")
    b_correct_names = {c["name"] for c in res_b["per_case"]
                       if c["name"] in {lc["name"] for lc in LOW_CASES} and c["threat_accuracy"] == 1.0}
    print(f"greedy (b) got {len(b_correct_names)}/15 low cases right: {sorted(b_correct_names)}")

    for label, log in (("c_sampled_standalone", log_c), ("d_sampled_composite", log_d)):
        print(f"\n--- {label} ---")
        counts = Counter()
        for name in sorted(b_correct_names):
            for i, assessment in enumerate(log[name]):
                cls = _classify_failure(assessment, "low")
                if cls != "correct":
                    counts[cls] += 1
                    print(f"  {name} run{i}: {cls}  (full assessment: "
                          f"threat={assessment.get('threat_level')!r} "
                          f"intent={assessment.get('likely_intent')!r} "
                          f"error={assessment.get('error')!r})")
        print(f"  totals over {label}: {dict(counts)}")

    # ================= step 2: high/critical confusion matrix =================
    print("\n\n=== STEP 2: high/critical confusion matrix ===")
    for label, log in (("rules_in_prompt (c)", log_c), ("composite (d)", log_d)):
        print(f"\n--- {label} ---")
        for expected in ("high", "critical"):
            names = {c["name"] for c in HIGH_CRIT_CASES if c["expected_threat"] == expected}
            if not names:
                continue
            n_flag = "  <-- n=2, NOT STATISTICALLY MEANINGFUL, reported anyway" if expected == "critical" else ""
            print(f"  expected={expected} (n_cases={len(names)}){n_flag}")
            pred_counts = Counter()
            for name in names:
                for assessment in log.get(name, []):
                    if "error" in assessment:
                        pred_counts["json_parse_failure"] += 1
                    elif is_abstention(assessment.get("likely_intent", "")):
                        pred_counts["abstained"] += 1
                    else:
                        pred_counts[assessment.get("threat_level", "?")] += 1
            total = sum(pred_counts.values())
            for pred, n in sorted(pred_counts.items(), key=lambda kv: -kv[1]):
                print(f"    predicted={pred}: {n}/{total} ({n/total:.1%})")

    print(f"\nmismatches found in step 0 prompt diff: {mismatches}")
    print("\nDone. See evaluation/reconcile_*.json for full raw data.")


if __name__ == "__main__":
    main()
