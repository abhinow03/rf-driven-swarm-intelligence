"""
Fills the one missing cell in the abstention picture: does `rules_in_prompt`
(base Qwen2.5-7B-Instruct, no adapter, RULES.txt pasted as system prompt --
baselines.py's make_rules_in_prompt_run_case protocol) abstain on genuinely
unanswerable input, the way the fine-tuned adapters do?

`evaluation/degradation_rules_in_prompt.json` (already on disk, commit
e9c9183) already answers this for multi_hop/terminal_transitioning:
abstention_rate_when_unanswerable = 0.0% across every severity of both axes
-- rules_in_prompt NEVER abstains on RULES-uncoverable input, unlike the
100% abstention rate every fine-tuned adapter (v2 does NOT abstain either,
by construction -- see ADAPTER_VERSIONS.md; v3a/v3b/v3d were specifically
trained to). What's missing is the held-out shapes (holdout_shapes.py) --
`evaluation/holdout_{v2,v3a,v3b}.json` exist, this adds rules_in_prompt.

Usage (run inside tmux -- see AUDIT.md sec AB step 2):
    python llm_finetuning/run_holdout_eval_rules_in_prompt.py --n-runs 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import GroqClient, LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from degradation import make_llm_battery_run_case  # noqa: E402
from holdout_shapes import build_holdout_battery  # noqa: E402
from baselines import load_rules_txt  # noqa: E402


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

    battery = build_holdout_battery()
    print("held-out battery:", {shape: len(cases) for shape, cases in battery.items()})
    total = sum(len(cases) for cases in battery.values()) * args.n_runs
    reporter = Reporter("holdout_rules_in_prompt", total, rate_hint=0.3)

    client = LocalHFClient(args.base, adapter_path=None, temperature=0.3,
                           system_prompt=load_rules_txt())
    run_case = make_llm_battery_run_case(client)

    out = {"system": "rules_in_prompt", "n_runs": args.n_runs, "shapes": {}}
    for shape, cases in battery.items():
        res = evaluate_llm(run_case, cases, judge_client=judge, n_runs=args.n_runs,
                           progress_reporter=reporter)
        out["shapes"][shape] = {"aggregate": res["aggregate"], "per_case": res["per_case"]}
        agg = res["aggregate"]
        print(f"    {shape}: abstain={_fmt(agg['abstention_rate_when_unanswerable'])} "
              f"halluc={_fmt(agg['mean_hallucination_rate'])}", flush=True)

    reporter.status = "done"
    reporter._write()

    out_dir = Path(args.out_dir)
    (out_dir / "holdout_rules_in_prompt.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'holdout_rules_in_prompt.json'}")

    print("\n=== rules_in_prompt: held-out shape abstention rates (n_runs={}) ===".format(args.n_runs))
    shapes = list(battery.keys())
    print("| system | " + " | ".join(shapes) + " |")
    print("|" + "---|" * (len(shapes) + 1))
    cells = [_fmt(out["shapes"][s]["aggregate"]["abstention_rate_when_unanswerable"]) for s in shapes]
    print(f"| rules_in_prompt | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
