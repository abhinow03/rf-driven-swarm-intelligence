"""
Runs adapters/qwen-swarm-v3a-nomask (assistant_only_loss=False, otherwise identical
to v3a: same data/sft_train_final.jsonl, same hyperparameters -- see train_qlora.py)
through the degradation battery, then compares it against the already-saved v3a
result (evaluation/degradation_v3a.json) on accuracy_when_answerable per axis.

This is the ONE cell that isolates the masking effect: v3a and v3a-nomask differ in
exactly one thing (assistant_only_loss), both trained on the same 234 rows with the
same hyperparameters. v2 differs from v3a in BOTH assistant_only_loss AND training
set (810 vs 234 rows, see the prior session's commit) -- the v2-vs-v3a delta is NOT
a clean masking measurement and should not be described as one; this script produces
the delta that is.

Usage:
    python llm_finetuning/run_masking_ablation.py --n-runs 5
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
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

from degradation import build_battery, make_llm_battery_run_case  # noqa: E402
from run_degradation_eval import group_by_severity, run_system, _fmt  # noqa: E402


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== qwen-swarm-v3a-nomask (adapter_path=adapters/qwen-swarm-v3a-nomask) ===")
    client = LocalHFClient(args.base, adapter_path=str(REPO / "adapters/qwen-swarm-v3a-nomask"),
                           temperature=0.3)
    run_case = make_llm_battery_run_case(client)
    res = run_system("qwen-swarm-v3a-nomask", run_case, battery, judge, args.n_runs)
    (out_dir / "degradation_v3a-nomask.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out_dir / 'degradation_v3a-nomask.json'}")
    del client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    v3a = json.loads((out_dir / "degradation_v3a.json").read_text())
    v3a_nomask = json.loads((out_dir / "degradation_v3a-nomask.json").read_text())

    print("\n\n=== MASKING EFFECT (isolated): v3a (assistant_only_loss=True) vs "
          "v3a-nomask (assistant_only_loss=False), SAME data/sft_train_final.jsonl, "
          "SAME hyperparameters ===")
    print("cells show accuracy_when_answerable only (this is what masking should "
          "affect; abstention behaviour is not expected to differ -- neither "
          "adapter saw the abstention-augmented dataset)")
    for axis in battery:
        print(f"\n--- {axis} ---")
        severities = [s for s, _ in group_by_severity(battery[axis])]
        print("| system | " + " | ".join(f"sev {s}" for s in severities) + " |")
        print("|" + "---|" * (len(severities) + 1))
        for label, data in (("qwen-swarm-v3a", v3a), ("qwen-swarm-v3a-nomask", v3a_nomask)):
            cells = [_fmt(block["aggregate"]["accuracy_when_answerable"])
                    for block in data["axes"][axis]]
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
