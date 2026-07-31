"""
Runs v2, v3a and v3b against the held-out unanswerable shapes (holdout_shapes.py) --
none of which appear in v3b's training data (verified by tests/test_holdout_shapes.py
and holdout_shapes.py's own __main__ check). This is the decisive test for whether
v3b's 100% abstention on multi_hop/terminal_transitioning is a learned "insufficient
information -> decline" capability, or memorization of the two training substrings:
  - v3b abstains here too  -> generalization
  - v3b answers confidently -> memorization of the two trained shapes specifically
Do NOT read this script's output as license to tune anything toward either outcome --
the whole point is finding out which one is true.

Since every case has_ground_truth=False, evaluate_llm's abstention_rate_when_
unanswerable IS the correct-behaviour metric (see src/swarm_intent/llm/evaluate.py).

Usage:
    python llm_finetuning/run_holdout_eval.py --n-runs 5
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

from degradation import make_llm_battery_run_case  # noqa: E402
from holdout_shapes import build_holdout_battery  # noqa: E402

SYSTEMS = {
    "qwen-swarm-v2": "adapters/qwen-swarm-v2",
    "qwen-swarm-v3a": "adapters/qwen-swarm-v3a",
    "qwen-swarm-v3b": "adapters/qwen-swarm-v3b",
}


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

    battery = build_holdout_battery(TEST_CASES)
    print("held-out battery:", {shape: len(cases) for shape, cases in battery.items()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for label, adapter_path in SYSTEMS.items():
        print(f"\n=== {label} (adapter_path={adapter_path}) ===")
        client = LocalHFClient(args.base, adapter_path=str(REPO / adapter_path), temperature=0.3)
        run_case = make_llm_battery_run_case(client)
        out = {"system": label, "n_runs": args.n_runs, "shapes": {}}
        for shape, cases in battery.items():
            res = evaluate_llm(run_case, cases, judge_client=judge, n_runs=args.n_runs)
            out["shapes"][shape] = {"aggregate": res["aggregate"], "per_case": res["per_case"]}
            agg = res["aggregate"]
            print(f"    {shape}: abstain={_fmt(agg['abstention_rate_when_unanswerable'])} "
                  f"halluc={_fmt(agg['mean_hallucination_rate'])}", flush=True)
        fname = f"holdout_{label.replace('qwen-swarm-', '')}.json"
        (out_dir / fname).write_text(json.dumps(out, indent=2))
        print(f"wrote {out_dir / fname}")
        all_results[label] = out
        del client
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    print("\n\n=== HELD-OUT SHAPE ABSTENTION RATES (n_runs={}) ===".format(args.n_runs))
    print("None of these shapes appear in v3b's training data (verified by "
          "holdout_shapes.py / tests/test_holdout_shapes.py). v3b abstaining here = "
          "generalization; v3b answering confidently = memorization of the two "
          "trained substrings.")
    shapes = list(battery.keys())
    print("\n| system | " + " | ".join(shapes) + " |")
    print("|" + "---|" * (len(shapes) + 1))
    for label, out in all_results.items():
        cells = [_fmt(out["shapes"][s]["aggregate"]["abstention_rate_when_unanswerable"])
                for s in shapes]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
