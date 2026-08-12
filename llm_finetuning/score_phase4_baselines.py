"""
Scores the 4 non-v5-a systems' raw output (evaluation/phase4_baselines_results.json,
produced by prerun_baselines_phase4.py) using the EXACT SAME scoring path as
v5-a (eval_sft_v5.py's evaluate()), against the same locked
evaluation/phase4_eval_set.json ground truth. Deliberately reuses that one
scorer rather than building a second one, so v5-a and the baselines are never
scored by two divergent code paths.

Usage:
    python llm_finetuning/score_phase4_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))

from eval_sft_v5 import evaluate, build_case_from_phase4_item, load_phase4_items  # noqa: E402

BASELINES_PATH = REPO / "evaluation" / "phase4_baselines_results.json"
OUT_PATH = REPO / "evaluation" / "phase4_baselines_scored.json"


def main():
    items = load_phase4_items()
    cases = [build_case_from_phase4_item(it) for it in items]

    with open(BASELINES_PATH) as f:
        payload = json.load(f)
    raw_results = payload["results"]

    scored = {}
    for label, per_case_raw in raw_results.items():
        scored[label] = evaluate(cases, lambda c, _label=label, _raw=per_case_raw: _raw[c["name"]])
        print(f"scored {label}")

    OUT_PATH.write_text(json.dumps(scored, indent=2))
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
