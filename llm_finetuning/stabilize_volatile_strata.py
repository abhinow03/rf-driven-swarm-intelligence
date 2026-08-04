"""
AUDIT.md sec AD step 2: last session's n_runs=5 threat-accuracy std was as
high as ~9-25pt on the low/high/critical strata (evaluation/
composite_comparison_table.json) -- too volatile to be the number that goes
in a writeup. Re-runs ONLY the low (15) + high (14) + critical (2) = 31
cases -- not the full 55-case battery, medium is already stable -- at
n_runs=20 for rules_in_prompt and composite, and reports mean with a proper
95% CI (t-distribution, since n_runs=20 is not asymptotically large) instead
of just std.

Same production code paths as every other session's measurement
(baselines.make_rules_in_prompt_run_case, composite.make_composite_run_case),
same seed=0 shared-rng protocol -- but walks only the 31 target cases (in
TEST_CASES order, so ctx draws are NOT byte-identical to a full-55-case walk
-- this is a deliberate, disclosed deviation for compute efficiency; every
number in this script's output should be read as "31-case-battery ctx
draws," not literally re-deriving the exact same contexts full-battery runs
used elsewhere in this project).

Usage (run inside tmux):
    python llm_finetuning/stabilize_volatile_strata.py --n-runs 20
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case, load_rules_txt  # noqa: E402
from composite import make_composite_run_case  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V3B_FIX_ADAPTER = "adapters/qwen-swarm-v3b-fix"

TARGET_CASES = [c for c in TEST_CASES if c["expected_threat"] in ("low", "high", "critical")]
assert len(TARGET_CASES) == 31


def _capturing(run_case, log: dict):
    def wrapped(case):
        assessment, ctx = run_case(case)
        log[case["name"]].append(assessment)
        return assessment, ctx
    return wrapped


def t_ci95(values):
    """95% CI half-width via the t-distribution (n_runs=20 -> df=19), not a
    normal-approximation z-interval -- appropriate for small n."""
    from scipy import stats
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return t * se


def run_level_accuracy(raw: dict, names: list, n_runs: int) -> list:
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from swarm_intent.llm.prompts import match_threat, is_abstention
    name_to_case = {c["name"]: c for c in TEST_CASES}
    run_accs = []
    for r in range(n_runs):
        hits, scored = 0, 0
        for name in names:
            case = name_to_case[name]
            a = raw[name][r]
            if is_abstention(a.get("likely_intent", "")):
                continue
            scored += 1
            if match_threat(a.get("threat_level", ""), case["expected_threat"]):
                hits += 1
        run_accs.append(hits / scored if scored else None)
    return [a for a in run_accs if a is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    total = len(TARGET_CASES) * args.n_runs * 2
    reporter = Reporter("stabilize_volatile_strata", total, rate_hint=0.25)

    print(f"=== rules_in_prompt: {len(TARGET_CASES)} cases (low+high+critical), n_runs={args.n_runs} ===")
    rules_client = LocalHFClient(args.base, adapter_path=None, temperature=0.3,
                                 system_prompt=load_rules_txt())
    log_rip: dict = defaultdict(list)
    run_case = _capturing(make_rules_in_prompt_run_case(rules_client, seed=0), log_rip)
    res_rip = evaluate_llm(run_case, TARGET_CASES, judge_client=None, n_runs=args.n_runs,
                           progress_reporter=reporter)
    (out_dir / "stabilize_rules_in_prompt.json").write_text(json.dumps(
        {"aggregate": res_rip["aggregate"], "raw": log_rip}, indent=2))

    print(f"\n=== composite: {len(TARGET_CASES)} cases, n_runs={args.n_runs} ===")
    ft_client = LocalHFClient(args.base, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    branch_log: dict = {}
    log_comp: dict = defaultdict(list)
    run_case = _capturing(make_composite_run_case(rules_client, ft_client, branch_log, seed=0), log_comp)
    res_comp = evaluate_llm(run_case, TARGET_CASES, judge_client=None, n_runs=args.n_runs,
                            progress_reporter=reporter)
    (out_dir / "stabilize_composite.json").write_text(json.dumps(
        {"aggregate": res_comp["aggregate"], "raw": log_comp, "branch_counts":
         {k: list(branch_log.values()).count(k) for k in set(branch_log.values())}}, indent=2))

    reporter.status = "done"
    reporter._write()

    del rules_client, ft_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print(f"\n\n=== STEP 2: stabilized strata, n_runs={args.n_runs}, mean +/- 95% CI (t-dist) ===")
    print("| system | stratum | mean | 95% CI | n_runs_scored |")
    print("|---|---|---|---|---|")
    for label, log in (("rules_in_prompt", log_rip), ("composite", log_comp)):
        for stratum in ("low", "high", "critical"):
            names = [c["name"] for c in TARGET_CASES if c["expected_threat"] == stratum]
            accs = run_level_accuracy(log, names, args.n_runs)
            mean = float(np.mean(accs))
            ci = t_ci95(accs)
            print(f"| {label} | {stratum} | {mean:.1%} | ±{ci:.1%} | {len(accs)} |")


if __name__ == "__main__":
    main()
