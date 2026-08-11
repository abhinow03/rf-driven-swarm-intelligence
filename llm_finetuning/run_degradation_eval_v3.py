"""
Runs adapters/qwen-swarm-v3a and adapters/qwen-swarm-v3b (assistant_only_loss=True,
see train_qlora.py) through the same degradation battery run_degradation_eval.py
ran v2/base/rules_in_prompt/rules_lookup through, then prints the 3-way comparison
v2 vs v3a vs v3b (v3a isolates MASKING, v3b isolates MASKING + ABSTENTION DATA vs
v2's assistant_only_loss=False baseline).

Reuses group_by_severity/run_system/_fmt from run_degradation_eval.py rather than
duplicating them -- same battery, same evaluate_llm, same JSON shape, so this run's
degradation_v3a.json / degradation_v3b.json are directly comparable to the existing
degradation_v2.json without any re-emit step (evaluate_llm itself now returns
accuracy_when_answerable / abstention_rate_when_unanswerable / over_abstention_rate
natively -- see src/swarm_intent/llm/evaluate.py).

Usage:
    python llm_finetuning/run_degradation_eval_v3.py --n-runs 5
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

from swarm_intent.llm.client import LocalHFClient, JUDGE_MODEL, default_judge_client  # noqa: E402
from swarm_intent.llm.prompts import ORIGINAL_TEST_CASES  # noqa: E402

from degradation import build_battery, make_llm_battery_run_case  # noqa: E402
from run_degradation_eval import group_by_severity, run_system, _fmt  # noqa: E402

ADAPTERS = {
    "qwen-swarm-v3a": "adapters/qwen-swarm-v3a",  # MASKING isolated
    "qwen-swarm-v3b": "adapters/qwen-swarm-v3b",  # MASKING + ABSTENTION DATA
}
COMPARE_FILES = {
    "qwen-swarm-v2": "degradation_v2.json",
    "qwen-swarm-v3a": "degradation_v3a.json",
    "qwen-swarm-v3b": "degradation_v3b.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--out-dir", default=str(REPO / "evaluation"))
    args = ap.parse_args()

    judge = default_judge_client()
    if judge:
        print(f"judge: {JUDGE_MODEL} via NVIDIA NIM (advisory only)")
    else:
        print("NVIDIA_API_KEY not set — running WITHOUT a judge; "
              "objective headline metrics are unaffected")

    battery = build_battery(ORIGINAL_TEST_CASES)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, adapter_path in ADAPTERS.items():
        print(f"\n=== {label} (adapter_path={adapter_path}) ===")
        client = LocalHFClient(args.base, adapter_path=str(REPO / adapter_path), temperature=0.3)
        run_case = make_llm_battery_run_case(client)
        res = run_system(label, run_case, battery, judge, args.n_runs)
        fname = f"degradation_{label.replace('qwen-swarm-', '')}.json"
        (out_dir / fname).write_text(json.dumps(res, indent=2))
        print(f"wrote {out_dir / fname}")
        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    print("\n\n=== 3-WAY COMPARISON: v2 (assistant_only_loss=False) vs v3a (masking) "
          "vs v3b (masking + abstention data) ===")
    print("acc = accuracy_when_answerable | abstain = abstention_rate_when_unanswerable "
          "| overabst = over_abstention_rate")
    all_data = {label: json.loads((out_dir / fname).read_text())
                for label, fname in COMPARE_FILES.items()}
    for axis in battery:
        print(f"\n--- {axis} ---")
        severities = [s for s, _ in group_by_severity(battery[axis])]
        print("| system | " + " | ".join(f"sev {s}" for s in severities) + " |")
        print("|" + "---|" * (len(severities) + 1))
        for label, data in all_data.items():
            cells = []
            for block in data["axes"][axis]:
                agg = block["aggregate"]
                cells.append(f"acc={_fmt(agg['accuracy_when_answerable'])} "
                            f"abstain={_fmt(agg['abstention_rate_when_unanswerable'])} "
                            f"overabst={_fmt(agg['over_abstention_rate'])}")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
