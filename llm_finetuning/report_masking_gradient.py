"""
Step 4 of the "is the 55-case battery in-distribution?" session (AUDIT.md sec J/K/L).

Puts v3a-vs-v3a-nomask (the clean masking 2x2, assistant_only_loss=True/False,
otherwise identical training data/hyperparameters -- AUDIT.md sec H) side by side
on two batteries:
  (a) the clean 55-case TEST_CASES battery [in-distribution, sec J]
  (b) the perturbed degradation battery [out-of-distribution shift, secs C/H]
to test whether the masking effect is bigger under distribution shift.

Pure post-processing of existing JSON (evaluation/eval_expanded_v3a*.json,
evaluation/degradation_v3a*.json) -- no model calls.

Std caveat: (a)'s std is across RUNS (n_runs independent replicate whole-battery
measurements -- accuracy_when_answerable_std_across_runs, added to evaluate.py after
the degradation battery in (b) was generated). (b)'s std here is across CASES within
an axis (the per-case intent_accuracy spread), because degradation_v3a*.json predates
that run-level instrumentation and re-running the full battery to get run-level std
retroactively is a further ~70min job not undertaken without being asked. The two
stds are not the same statistic -- do not treat them as directly comparable error bars,
only as "is there dispersion" indicators.

Usage:
    python llm_finetuning/report_masking_gradient.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def clean_battery_acc(system: str):
    d = json.loads((REPO / "evaluation" / f"eval_expanded_{system}.json").read_text())
    agg = d["aggregate"]
    return agg["accuracy_when_answerable_mean_across_runs"], agg["accuracy_when_answerable_std_across_runs"]


def degradation_axis_acc(system: str, axis: str):
    """Pool all severities of `axis` that have ground-truth cells: weighted mean by
    n_cases_with_ground_truth, and case-level std from the pooled per_case
    intent_accuracy values (not run-level std -- see module docstring)."""
    d = json.loads((REPO / "evaluation" / f"degradation_{system}.json").read_text())
    accs, weights = [], []
    per_case_vals = []
    for sev_block in d["axes"][axis]:
        agg = sev_block["aggregate"]
        if agg["accuracy_when_answerable"] is None:
            continue
        accs.append(agg["accuracy_when_answerable"])
        weights.append(agg["n_cases_with_ground_truth"])
        for c in sev_block["per_case"]:
            if c["has_ground_truth"]:
                per_case_vals.append(c["intent_accuracy"])
    if not accs:
        return None, None
    weighted_mean = float(np.average(accs, weights=weights))
    case_std = float(np.std(per_case_vals)) if per_case_vals else None
    return weighted_mean, case_std


def main():
    print("=== masking effect as a gradient: v3a vs v3a-nomask ===\n")
    print("| battery | v3a acc (mean±std) | v3a-nomask acc (mean±std) | delta (pts) |")
    print("|---|---|---|---|")

    rows = []
    clean_v3a = clean_battery_acc("v3a")
    clean_nomask = clean_battery_acc("v3a-nomask")
    delta_clean = (clean_v3a[0] - clean_nomask[0]) * 100
    print(f"| clean 55-case (in-distribution, std=across-runs) | "
          f"{clean_v3a[0]:.1%}±{clean_v3a[1]:.1%} | "
          f"{clean_nomask[0]:.1%}±{clean_nomask[1]:.1%} | {delta_clean:+.1f} |")
    rows.append(("clean_55case", delta_clean))

    for axis in ("multi_hop", "confidence_decay", "dropped_lines", "contradictory_cues"):
        v3a_m, v3a_s = degradation_axis_acc("v3a", axis)
        nomask_m, nomask_s = degradation_axis_acc("v3a-nomask", axis)
        if v3a_m is None or nomask_m is None:
            continue
        delta = (v3a_m - nomask_m) * 100
        print(f"| degradation: {axis} (perturbed, std=across-cases) | "
              f"{v3a_m:.1%}±{v3a_s:.1%} | {nomask_m:.1%}±{nomask_s:.1%} | {delta:+.1f} |")
        rows.append((f"degradation_{axis}", delta))

    degradation_deltas = [d for name, d in rows if name != "clean_55case"]
    mean_degradation_delta = sum(degradation_deltas) / len(degradation_deltas)
    print(f"\nclean-battery delta: {delta_clean:+.1f} pts")
    print(f"mean degradation-battery delta (4 axes): {mean_degradation_delta:+.1f} pts")
    if mean_degradation_delta > delta_clean:
        print("FINDING: the masking effect is LARGER under perturbation than on the "
              "clean in-distribution battery -- masking's benefit scales with "
              "distribution shift, it is not a fixed constant.")
    else:
        print("FINDING: the masking effect is NOT larger under perturbation -- does not "
              "support 'masking's benefit scales with distribution shift.'")

    (REPO / "evaluation" / "masking_gradient.json").write_text(
        json.dumps({"clean_delta_pts": delta_clean,
                    "degradation_deltas_pts": dict((n, d) for n, d in rows if n != "clean_55case"),
                    "mean_degradation_delta_pts": mean_degradation_delta}, indent=2))


if __name__ == "__main__":
    main()
