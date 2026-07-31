"""
Re-emits evaluation/degradation_{system}.json in place under the decomposed
metric contract (see src/swarm_intent/llm/evaluate.py's module docstring):
accuracy_when_answerable, abstention_rate_when_unanswerable, over_abstention_rate
-- computed instead of a single blended number per (axis, severity) cell.

No model calls, no re-run: every per_case block already stored in these files
carries has_ground_truth, abstention_rate and intent_accuracy per case (they
are evaluate_llm's own per-case output from the original run), so the three
decomposed aggregate fields are recomputed directly from data already on disk.
This makes the BEFORE column (v2, base, rules_in_prompt, rules_lookup) directly
comparable to whatever v3a/v3b produce later, without re-running ~1620
generations.

Usage:
    python llm_finetuning/reemit_degradation_decomposition.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "evaluation"

FILES = {
    "rules_lookup": "degradation_rules_lookup.json",
    "base": "degradation_base.json",
    "rules_in_prompt": "degradation_rules_in_prompt.json",
    "qwen-swarm-v2": "degradation_v2.json",
}

BY_CONSTRUCTION_NOTE = (
    "rules_lookup's abstention on multi_hop (severity>=3) and terminal_transitioning "
    "(all severities) is BY CONSTRUCTION: RULES (build_sft_dataset.py) has no key for "
    "those inputs, so it structurally cannot do anything else. This is the baseline's "
    "definition, not a measured behaviour -- see degradation.py's module docstring."
)


def decompose(per_case: list[dict]) -> dict:
    gt = [c for c in per_case if c["has_ground_truth"]]
    no_gt = [c for c in per_case if not c["has_ground_truth"]]
    gt_acc = [c["intent_accuracy"] for c in gt if c["intent_accuracy"] is not None]
    no_gt_abstain = [c["abstention_rate"] for c in no_gt]
    gt_abstain = [c["abstention_rate"] for c in gt]
    return {
        "accuracy_when_answerable": sum(gt_acc) / len(gt_acc) if gt_acc else None,
        "abstention_rate_when_unanswerable": (sum(no_gt_abstain) / len(no_gt_abstain)
                                              if no_gt_abstain else None),
        "over_abstention_rate": sum(gt_abstain) / len(gt_abstain) if gt_abstain else None,
    }


def main():
    all_data = {}
    for system, fname in FILES.items():
        path = EVAL_DIR / fname
        data = json.loads(path.read_text())
        for axis, blocks in data["axes"].items():
            for block in blocks:
                block["aggregate"].update(decompose(block["per_case"]))
        if system == "rules_lookup":
            data["note"] = BY_CONSTRUCTION_NOTE
        path.write_text(json.dumps(data, indent=2))
        print(f"re-emitted {path} ({sum(len(b) for b in data['axes'].values())} severity blocks)")
        all_data[system] = data

    print(f"\n{BY_CONSTRUCTION_NOTE}\n")
    print("=== DECOMPOSED PER-AXIS SUMMARY (before column) ===")
    for axis in all_data["rules_lookup"]["axes"]:
        print(f"\n--- {axis} ---")
        severities = [b["severity"] for b in all_data["rules_lookup"]["axes"][axis]]
        print("| system | " + " | ".join(f"sev {s}" for s in severities) + " |")
        print("|" + "---|" * (len(severities) + 1))
        for system, data in all_data.items():
            cells = []
            for block in data["axes"][axis]:
                agg = block["aggregate"]

                def fmt(x):
                    return f"{x:.0%}" if x is not None else "N/A"

                cells.append(f"acc={fmt(agg['accuracy_when_answerable'])} "
                            f"abstain={fmt(agg['abstention_rate_when_unanswerable'])} "
                            f"overabst={fmt(agg['over_abstention_rate'])}")
            note = "  <- by construction" if system == "rules_lookup" else ""
            print(f"| {system} | " + " | ".join(cells) + f" |{note}")


if __name__ == "__main__":
    main()
