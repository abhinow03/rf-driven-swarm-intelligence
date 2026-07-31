"""
Runs all 4 systems (rules_lookup, base, rules_in_prompt, qwen-swarm-v2) through the
degradation battery (llm_finetuning/degradation.py) -- the SAME 4 systems as the
clean 6-case eval (run_4way_eval.py, commit a943617), so this is the direct
before/after comparison: how does each system do on inputs designed to have
headroom, versus the clean battery qwen-swarm-v2 already saturates.

For each system, groups the battery's 108 cases by (axis, severity) and runs
evaluate_llm SEPARATELY per group (so each group gets its own aggregate, not one
number smeared across axes that mean very different things) -- --n-runs 5 per
group. Saves evaluation/degradation_{system}.json per system, and prints a
per-axis/per-severity summary table at the end.

judge (llama-3.3-70b-versatile, GroqClient) is ADVISORY ONLY, same as
run_4way_eval.py -- the headline metrics don't depend on it. If GROQ_API_KEY isn't
set, runs without a judge.

Usage:
    python llm_finetuning/run_degradation_eval.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import GroqClient, LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

from baselines import load_rules_txt  # noqa: E402
from degradation import build_battery, make_rules_lookup_battery_run_case, make_llm_battery_run_case  # noqa: E402


def group_by_severity(cases: list[dict]) -> list[tuple]:
    by_sev = {}
    for c in cases:
        by_sev.setdefault(c["severity"], []).append(c)
    return sorted(by_sev.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))


def run_system(label: str, run_case, battery: dict, judge, n_runs: int) -> dict:
    out = {"system": label, "n_runs": n_runs, "axes": {}}
    for axis, cases in battery.items():
        axis_results = []
        for severity, sev_cases in group_by_severity(cases):
            res = evaluate_llm(run_case, sev_cases, judge_client=judge, n_runs=n_runs)
            agg = res["aggregate"]
            axis_results.append({"severity": severity, "aggregate": agg, "per_case": res["per_case"]})
            n_gt, n_no_gt = agg["n_cases_with_ground_truth"], agg["n_cases_without_ground_truth"]
            gt = f"{n_gt}gt/{n_no_gt}no-gt" if (n_gt and n_no_gt) else ("gt" if n_gt else "no-gt")
            print(f"    {axis} sev={severity} ({gt}): "
                  f"acc_when_answerable={_fmt(agg['accuracy_when_answerable'])} "
                  f"abstain_when_unanswerable={_fmt(agg['abstention_rate_when_unanswerable'])} "
                  f"over_abstain={_fmt(agg['over_abstention_rate'])} "
                  f"halluc={_fmt(agg['mean_hallucination_rate'])}", flush=True)
        out["axes"][axis] = axis_results
    return out


def _fmt(x):
    return f"{x:.2%}" if x is not None else "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    judge = None
    if os.environ.get("GROQ_API_KEY"):
        judge = GroqClient(model="llama-3.3-70b-versatile")
        print("judge: llama-3.3-70b-versatile (advisory only)")
    else:
        print("GROQ_API_KEY not set — running WITHOUT a judge; "
              "objective headline metrics are unaffected")

    battery = build_battery(TEST_CASES)
    print("battery:", {axis: len(cases) for axis, cases in battery.items()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("\n=== rules_lookup (no model call) ===")
    run_case = make_rules_lookup_battery_run_case()
    res = run_system("rules_lookup", run_case, battery, judge, args.n_runs)
    (out_dir / "degradation_rules_lookup.json").write_text(json.dumps(res, indent=2))
    all_results["rules_lookup"] = res

    print("\n=== base (no adapter) ===")
    base_client = LocalHFClient(args.base, adapter_path=None, temperature=0.3)
    run_case = make_llm_battery_run_case(base_client)
    res = run_system("base", run_case, battery, judge, args.n_runs)
    (out_dir / "degradation_base.json").write_text(json.dumps(res, indent=2))
    all_results["base"] = res

    print("\n=== rules_in_prompt (base + RULES.txt system prompt) ===")
    base_client.system_prompt = load_rules_txt()
    run_case = make_llm_battery_run_case(base_client)
    res = run_system("rules_in_prompt", run_case, battery, judge, args.n_runs)
    (out_dir / "degradation_rules_in_prompt.json").write_text(json.dumps(res, indent=2))
    all_results["rules_in_prompt"] = res

    del base_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print("\n=== adapters/qwen-swarm-v2 ===")
    v2_client = LocalHFClient(args.base, adapter_path=str(REPO / "adapters/qwen-swarm-v2"),
                              temperature=0.3)
    run_case = make_llm_battery_run_case(v2_client)
    res = run_system("qwen-swarm-v2", run_case, battery, judge, args.n_runs)
    (out_dir / "degradation_v2.json").write_text(json.dumps(res, indent=2))
    all_results["qwen-swarm-v2"] = res

    print("\n\n=== PER-AXIS SUMMARY (n_runs={}) ===".format(args.n_runs))
    print("NOTE: rules_lookup's abstention on multi_hop/terminal_transitioning is BY")
    print("CONSTRUCTION -- RULES has no key for those inputs, so it cannot do anything")
    print("else. That is not a measured behaviour, it's the baseline's definition.")
    print("Three decomposed metrics per cell (never one blended number -- a cell can")
    print("mix has_ground_truth=True/False cases, e.g. dropped_lines):")
    print("  acc      = accuracy_when_answerable          (N/A if no gt cases in cell)")
    print("  abstain  = abstention_rate_when_unanswerable  (N/A if no non-gt cases in cell)")
    print("  overabst = over_abstention_rate                (abstaining on answerable cases)")
    for axis in battery:
        print(f"\n--- {axis} ---")
        severities = [s for s, _ in group_by_severity(battery[axis])]
        header = "| system | " + " | ".join(f"sev {s}" for s in severities) + " |"
        print(header)
        print("|" + "---|" * (len(severities) + 1))
        for system, res in all_results.items():
            cells = []
            for sev_block in res["axes"][axis]:
                agg = sev_block["aggregate"]
                cells.append(f"acc={_fmt(agg['accuracy_when_answerable'])} "
                            f"abstain={_fmt(agg['abstention_rate_when_unanswerable'])} "
                            f"overabst={_fmt(agg['over_abstention_rate'])}")
            note = "  <- by construction" if system == "rules_lookup" else ""
            print(f"| {system} | " + " | ".join(cells) + f" |{note}")


if __name__ == "__main__":
    main()
