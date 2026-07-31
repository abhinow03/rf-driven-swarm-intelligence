"""
Step 1 of the "sampling vs greedy" session (AUDIT.md sec Z).

Every eval/degradation runner on disk (grep confirms: run_degradation_eval.py,
run_degradation_eval_v3.py, run_masking_ablation.py, evaluate_finetuned.py,
run_stratified_abstention_retest.py, run_v3c_eval.py, run_batching_equivalence_
check.py, run_4way_eval.py, run_headline_eval.py, run_holdout_eval.py) constructs
LocalHFClient(..., temperature=0.3) -- do_sample=True, temperature=0.3 explicit.
top_p/top_k/repetition_penalty are NEVER set explicitly anywhere in this project;
they silently inherit from Qwen2.5-7B-Instruct's own generation_config.json
(top_p=0.8, top_k=20, repetition_penalty=1.05) because HF's model.generate() falls
back to self.generation_config for any kwarg not explicitly passed. So every
result on disk to date was generated with:
    do_sample=True, temperature=0.3, top_p=0.8, top_k=20, repetition_penalty=1.05

This re-runs the 55-case battery for v2/v3a/v3a-nomask/v3b with temperature=0.0
(LocalHFClient's do_sample=False branch -- pure greedy, no top_p/top_k/repetition
penalty effect since sampling is off). Greedy decoding is deterministic given a
fixed prompt, so n_runs=1 (not 5) -- repeating an identical greedy generation five
times would waste GPU time for zero new information. Uses the batch_size=8 batched
path (AUDIT.md sec U, equivalence-checked and confirmed safe) for speed.

Usage:
    python llm_finetuning/run_greedy_eval.py --batch-size 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_rules_in_prompt_run_case, make_batched_run_case  # noqa: E402

SYSTEMS = [
    ("v2", "adapters/qwen-swarm-v2"),
    ("v3a", "adapters/qwen-swarm-v3a"),
    ("v3a-nomask", "adapters/qwen-swarm-v3a-nomask"),
    ("v3b", "adapters/qwen-swarm-v3b"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    n_cases = len(TEST_CASES)
    total = n_cases * len(SYSTEMS)
    print(f"=== greedy re-run: {n_cases} cases x 1 run x {len(SYSTEMS)} systems = {total} generations "
          f"(temperature=0.0, do_sample=False -- deterministic, n_runs=1 by design) ===")

    reporter = Reporter("run_greedy_eval", total, rate_hint=0.3)
    out_dir = Path(args.out_dir)
    all_results = {}

    for label, adapter_subdir in SYSTEMS:
        print(f"\n=== {label} (greedy) ===")
        client = LocalHFClient(args.base, adapter_path=str(REPO / adapter_subdir), temperature=0.0)
        if args.batch_size > 1:
            run_case = make_batched_run_case(client, TEST_CASES, 1, args.batch_size, seed=0)
        else:
            run_case = make_rules_in_prompt_run_case(client, seed=0)
        res = evaluate_llm(run_case, TEST_CASES, judge_client=None, n_runs=1,
                           progress_reporter=reporter)
        (out_dir / f"eval_expanded_{label}_greedy.json").write_text(json.dumps(res, indent=2))
        all_results[label] = res
        agg = res["aggregate"]
        print(f"  accuracy_when_answerable={agg['accuracy_when_answerable']:.2%} "
              f"threat_acc={agg['mean_threat_accuracy']:.2%}")

        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    reporter.status = "done"
    reporter._write()
    print("\n\nDone. Per-class greedy-vs-sampled comparison: see "
          "llm_finetuning/report_greedy_vs_sampled.py")


if __name__ == "__main__":
    main()
