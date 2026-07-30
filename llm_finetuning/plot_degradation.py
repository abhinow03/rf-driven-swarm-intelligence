"""
Plots evaluation/degradation_curves.png from evaluation/degradation_{system}.json --
one panel per axis, four curves (one per system), severity on x.

Y-axis is a blended "success rate": for each (system, axis, severity) cell,
weight-averages mean_intent_accuracy (over cases that HAD a ground-truth answer)
with mean_correct_abstention_rate (over cases that did NOT), weighted by how many
cases of each kind were in that cell. This is the single number that answers "did
the system do the objectively correct thing" whether "correct" means matching the
expected intent or correctly abstaining -- computed from evaluate_llm's own
aggregate fields, not a new scoring rule.

Colors: fixed categorical assignment (not re-ordered by rank/performance), from the
project's validated palette -- same color for the same system in every panel.

Usage:
    python llm_finetuning/plot_degradation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "evaluation"

SYSTEMS = ["rules_lookup", "base", "rules_in_prompt", "qwen-swarm-v2"]
FILES = {
    "rules_lookup": "degradation_rules_lookup.json",
    "base": "degradation_base.json",
    "rules_in_prompt": "degradation_rules_in_prompt.json",
    "qwen-swarm-v2": "degradation_v2.json",
}
# Fixed categorical order (project palette) -- same color per system in every panel.
COLORS = {
    "rules_lookup": "#2a78d6",      # blue
    "base": "#e34948",              # red
    "rules_in_prompt": "#eda100",   # yellow
    "qwen-swarm-v2": "#1baf7a",     # aqua
}
AXIS_TITLES = {
    "multi_hop": "multi-hop chains",
    "terminal_transitioning": "terminal 'transitioning'",
    "confidence_decay": "confidence decay",
    "dropped_lines": "dropped context lines",
    "contradictory_cues": "contradictory cues",
}


def blended_success(agg: dict):
    n_gt = agg["n_cases_with_ground_truth"]
    n_no_gt = agg["n_cases_without_ground_truth"]
    total = n_gt + n_no_gt
    if total == 0:
        return None
    gt_part = (agg["mean_intent_accuracy"] or 0.0) * n_gt
    no_gt_part = (agg["mean_correct_abstention_rate"] or 0.0) * n_no_gt
    return 100.0 * (gt_part + no_gt_part) / total


def main():
    data = {s: json.loads((EVAL_DIR / FILES[s]).read_text()) for s in SYSTEMS}
    axes_names = list(data["rules_lookup"]["axes"].keys())

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888",
                         "axes.labelcolor": "#333333", "text.color": "#333333"})
    fig, axs = plt.subplots(1, len(axes_names), figsize=(4.6 * len(axes_names), 4.2), sharey=True)

    for ax, axis in zip(axs, axes_names):
        for system in SYSTEMS:
            blocks = sorted(data[system]["axes"][axis], key=lambda b: float(b["severity"]))
            xs = [b["severity"] for b in blocks]
            ys = [blended_success(b["aggregate"]) for b in blocks]
            positions = list(range(len(xs)))
            ax.plot(positions, ys, marker="o", markersize=6, linewidth=2,
                   color=COLORS[system], label=system)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(x) for x in xs], fontsize=9)
        ax.set_title(AXIS_TITLES.get(axis, axis), fontsize=11)
        ax.set_xlabel("severity")
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[0].set_ylabel("success rate (%)\n[intent accuracy where answerable,\ncorrect-abstention where not]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=10)
    fig.suptitle("Degradation battery: success rate vs. severity, by system and axis", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = EVAL_DIR / "degradation_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
