"""
Headline 7-way comparison on the EXPANDED battery (TEST_CASES, now 55 cases --
ORIGINAL_TEST_CASES + full RULES-pair coverage, see src/swarm_intent/llm/prompts.py)
instead of the original 6. Same evaluate_llm() harness, same n_runs, so results are
directly comparable to run_4way_eval.py's numbers in kind (not magnitude -- 55 cases
is a different, larger, more representative sample of RULES than 6).

IN-DISTRIBUTION CAVEAT (AUDIT.md sec J): this battery covers 49/49 RULES pairs, and
so does every SFT training file (sft_train_v2/final/final_abstain.jsonl) -- see
llm_finetuning/check_training_coverage.py. For qwen-swarm-v2/v3a/v3a-nomask/v3b this
is therefore a rule-table RECALL battery, not a generalization test. High
accuracy_when_answerable here should be reported as "recall on the trained rule
table," not "generalization accuracy." The generalization test is
llm_finetuning/holdout_shapes.py.

Systems: rules_lookup, base, rules_in_prompt, qwen-swarm-v2, qwen-swarm-v3a,
qwen-swarm-v3a-nomask, qwen-swarm-v3b -- the full set this session's ablation
produced. rules_lookup is excluded from the shared progress Reporter (its "units"
are sub-millisecond dict lookups, not model calls, and would badly skew the
rate/ETA estimate if mixed with the ~1650 real generations from the other 6
systems); it still gets its own quick before/after print.

Every case here has has_ground_truth=True (it's the clean RULES-pair battery, not
the perturbed degradation battery), so abstention_rate_when_unanswerable is N/A
throughout by construction -- accuracy_when_answerable and over_abstention_rate
are the two metrics that matter. Reports mean +/- std ACROSS RUNS (n_runs
independent replicate measurements, see evaluate_llm's *_across_runs fields) as
error bars, not a single point estimate.

Usage:
    python llm_finetuning/run_headline_eval.py --n-runs 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient, JUDGE_MODEL, default_judge_client  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import (make_rules_lookup_run_case, make_rules_in_prompt_run_case,  # noqa: E402
                       make_batched_run_case, load_rules_txt)

# label -> adapter subdir under adapters/, or None for the base model with no adapter
LLM_SYSTEMS = [
    ("base", None),
    ("rules_in_prompt", None),  # base client, RULES.txt in system prompt -- set below
    ("qwen-swarm-v2", "adapters/qwen-swarm-v2"),
    ("qwen-swarm-v3a", "adapters/qwen-swarm-v3a"),
    ("qwen-swarm-v3a-nomask", "adapters/qwen-swarm-v3a-nomask"),
    ("qwen-swarm-v3b", "adapters/qwen-swarm-v3b"),
]


def pct_or_na(x) -> str:
    return f"{x:.2%}" if x is not None else "N/A"


def mean_std_str(mean, std) -> str:
    if mean is None:
        return "N/A"
    return f"{mean:.2%} +/- {std:.2%}" if std is not None else f"{mean:.2%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    ap.add_argument("--batch-size", type=int, default=1,
                    help="generation batch size for LocalHFClient-backed systems. "
                         "1 = exact reproduction of pre-batching results (sequential "
                         "generate(), byte-identical prompts/ctx). >1 pre-generates "
                         "the whole battery for that system in one batched pass BEFORE "
                         "evaluate_llm's scoring loop runs -- so the progress reporter "
                         "will appear to jump from ~0%% to 100%% quickly once scoring "
                         "starts, since all GPU time was already spent generating.")
    ap.add_argument("--rate-hint", type=float, default=0.25,
                    help="generations/sec used for the upfront runtime estimate "
                         "(observed ~0.2-0.28/s for 4-bit Qwen2.5-7B on this GPU "
                         "across earlier sessions' battery runs)")
    args = ap.parse_args()

    n_cases = len(TEST_CASES)
    n_llm_systems = len(LLM_SYSTEMS)
    total_llm_generations = n_cases * args.n_runs * n_llm_systems
    print(f"=== {n_cases} test cases x {args.n_runs} runs x {n_llm_systems} LLM-backed "
          f"systems = {total_llm_generations} total generations "
          f"(+ rules_lookup: {n_cases * args.n_runs} instant lookups, no model call) ===")
    print(f"estimated runtime at --rate-hint={args.rate_hint}/s: "
          f"~{total_llm_generations / args.rate_hint / 60:.0f} minutes "
          f"({total_llm_generations / args.rate_hint / 3600:.1f} hours)")

    judge = default_judge_client()
    if judge:
        print(f"judge: {JUDGE_MODEL} via NVIDIA NIM (advisory only)")
    else:
        print("NVIDIA_API_KEY not set — running WITHOUT a judge; "
              "objective headline metrics are unaffected")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    t0 = time.time()
    print("\n=== rules_lookup (no model call, not tracked by the progress Reporter) ===")
    run_case = make_rules_lookup_run_case(seed=0)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    (out_dir / "eval_expanded_rules_lookup.json").write_text(json.dumps(res, indent=2))
    all_results["rules_lookup"] = res
    print(f"  done in {time.time() - t0:.1f}s")

    reporter = Reporter("run_headline_eval", total_llm_generations, rate_hint=args.rate_hint)

    base_client = None
    for label, adapter_subdir in LLM_SYSTEMS:
        print(f"\n=== {label} ===")
        if label == "base":
            base_client = LocalHFClient(args.base, adapter_path=None, temperature=0.3)
            client = base_client
        elif label == "rules_in_prompt":
            base_client.system_prompt = load_rules_txt()
            client = base_client
        else:
            if base_client is not None:
                del base_client
                base_client = None
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            client = LocalHFClient(args.base, adapter_path=str(REPO / adapter_subdir), temperature=0.3)

        if args.batch_size > 1:
            run_case = make_batched_run_case(client, TEST_CASES, args.n_runs, args.batch_size, seed=0)
        else:
            run_case = make_rules_in_prompt_run_case(client, seed=0)
        res = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs,
                           progress_reporter=reporter)
        suffix = f"_batch{args.batch_size}" if args.batch_size > 1 else ""
        fname = f"eval_expanded_{label.replace('qwen-swarm-', '')}{suffix}.json"
        (out_dir / fname).write_text(json.dumps(res, indent=2))
        all_results[label] = res
        agg = res["aggregate"]
        print(f"  accuracy_when_answerable={pct_or_na(agg['accuracy_when_answerable'])} "
              f"over_abstention={pct_or_na(agg['over_abstention_rate'])} "
              f"halluc={pct_or_na(agg['mean_hallucination_rate'])}")

        if label not in ("base", "rules_in_prompt"):
            del client
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    reporter.status = "done"
    reporter._write()

    print(f"\n\n=== HEADLINE 7-WAY COMPARISON (n_runs={args.n_runs}, {n_cases} test cases, "
          f"49/49 RULES pairs) ===")
    print("mean +/- std computed ACROSS RUNS (n_runs independent replicate measurements "
          "of the whole-battery aggregate), not point estimates.")
    print("\n| system | accuracy_when_answerable | over_abstention_rate | hallucination_rate |")
    print("|---|---|---|---|")
    for name, res in all_results.items():
        agg = res["aggregate"]
        acc_str = mean_std_str(agg.get("accuracy_when_answerable_mean_across_runs"),
                               agg.get("accuracy_when_answerable_std_across_runs"))
        overabst_str = mean_std_str(agg.get("over_abstention_rate_mean_across_runs"),
                                    agg.get("over_abstention_rate_std_across_runs"))
        print(f"| {name} | {acc_str} | {overabst_str} | {pct_or_na(agg['mean_hallucination_rate'])} |")

    total_elapsed = time.time() - t0
    print(f"\ntotal wall-clock time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")


if __name__ == "__main__":
    main()
