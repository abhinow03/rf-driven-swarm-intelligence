"""
Step 3b of the "settle the entropy confound, then fix the abstention bug"
session (AUDIT.md sec AA step 3).

`build_abstain_rows.py` used to hardcode threat_level="medium" on all 36
abstention rows (sec Z part b) because threat_level had no schema-legal
"unknown" the way likely_intent did. That's now fixed (prompts.py
THREAT_FAMILIES gained an "unknown" family; the abstention rows now say
"unknown"). `qwen-swarm-v3b-fix` is v3b retrained IDENTICALLY (same
hyperparameters: r=16/alpha=32, 3 epochs, lr=2e-4, assistant_only_loss=True,
same val file) on `data/sft_train_final_abstain_fix.jsonl` -- the same
234+36 rows as v3b's `sft_train_final_abstain.jsonl`, differing ONLY in
those 36 rows' threat_level target.

This measures whether the bug was suppressing abstention quality, on the
exact same two protocols v3b's existing on-disk results already used, so
the comparison is apples-to-apples with zero re-running of v3b itself:

  - per-class threat accuracy: greedy (temperature=0.0, n_runs=1) 55-case
    battery, same protocol as run_greedy_eval.py -- writes
    evaluation/eval_expanded_v3b-fix_greedy.json, directly comparable to
    the existing evaluation/eval_expanded_v3b_greedy.json.
  - abstention rate on multi_hop / terminal_transitioning + over-abstention:
    the same degradation battery run_degradation_eval_v3.py used for v3b
    (n_runs=5), writes evaluation/degradation_v3b-fix.json, directly
    comparable to the existing evaluation/degradation_v3b.json.

Usage:
    python llm_finetuning/eval_v3b_fix.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES, ORIGINAL_TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import make_batched_run_case  # noqa: E402
from degradation import build_battery, make_llm_battery_run_case  # noqa: E402
from run_degradation_eval import run_system  # noqa: E402

LABEL = "v3b-fix"
ADAPTER = "adapters/qwen-swarm-v3b-fix"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    import torch

    out_dir = REPO / "evaluation"

    print(f"=== {LABEL}: greedy 55-case battery ===")
    client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / ADAPTER), temperature=0.0)
    run_case = make_batched_run_case(client, TEST_CASES, 1, 8, seed=0)
    reporter = Reporter(f"eval_{LABEL}_greedy", len(TEST_CASES), rate_hint=0.3)
    res = evaluate_llm(run_case, TEST_CASES, judge_client=None, n_runs=1, progress_reporter=reporter)
    reporter.status = "done"
    reporter._write()
    (out_dir / f"eval_expanded_{LABEL}_greedy.json").write_text(json.dumps(res, indent=2))
    agg = res["aggregate"]
    print(f"  accuracy_when_answerable={agg['accuracy_when_answerable']:.2%} "
          f"threat_acc={agg['mean_threat_accuracy']:.2%}")

    del client
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n=== {LABEL}: degradation battery (n_runs=5, matches degradation_v3b.json) ===")
    battery = build_battery(ORIGINAL_TEST_CASES)
    client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / ADAPTER), temperature=0.3)
    run_case = make_llm_battery_run_case(client)
    res = run_system(LABEL, run_case, battery, judge=None, n_runs=5)
    (out_dir / f"degradation_{LABEL}.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out_dir / f'degradation_{LABEL}.json'}")

    del client
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
