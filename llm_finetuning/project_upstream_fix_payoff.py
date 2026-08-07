"""
AUDIT.md sec AH step 3: project the payoff of fixing the upstream dispersed/
converging geometry defect (data_gen.py's get_formation_offsets(), diagnosed
sec AF step 3, quantified sec AG step 1), using ONLY the failure-category
breakdown already on disk (evaluation/reduction_failure_diagnosis.json,
sec AG step 1) plus the before-fix bucket counts (evaluation/
robust_reduction_firing_rate.json, sec AG step 2). No new sequences are
generated, no model is loaded, no new experiment is run -- this is
post-processing of two existing JSON files.

Primary projection (single assumption, stated explicitly): if dispersed and
converging had distinct base geometry, the ambiguity guard would stop firing
on sequences where it is CURRENTLY the only thing blocking bucket A under
the CURRENT DEFAULT (robust=False) reduction. Sequences that ALSO carry an
independent guard trigger (oov_name from a stray unknown window,
dominant_history_contradiction from a tie) stay guarded -- fixing the
geometry defect does not touch those.

Usage:
    python llm_finetuning/project_upstream_fix_payoff.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

N_TOTAL_SEQUENCES = 500  # sec AE/AF/AG's held-out set


def main():
    diagnosis = json.loads((REPO / "evaluation" / "reduction_failure_diagnosis.json").read_text())
    firing = json.loads((REPO / "evaluation" / "robust_reduction_firing_rate.json").read_text())

    n_gt = len(diagnosis)  # 249, the GT-clean subset
    cat_counts = Counter(r["category"] for r in diagnosis)
    already_resolved = cat_counts["already_resolved"]
    ambiguity_cases = [r for r in diagnosis if r["category"] == "dispersed_converging_ambiguity"]
    assert len(ambiguity_cases) == cat_counts["dispersed_converging_ambiguity"]

    # split the ambiguity-blocked cases by whether ambiguity is the ONLY guard reason
    pure_ambiguity = [r for r in ambiguity_cases if r["guard_reasons"] == ["dispersed_converging_ambiguity"]]
    mixed_ambiguity = [r for r in ambiguity_cases if r["guard_reasons"] != ["dispersed_converging_ambiguity"]]
    assert len(pure_ambiguity) + len(mixed_ambiguity) == len(ambiguity_cases)

    print(f"GT-clean subset: n={n_gt} (of {N_TOTAL_SEQUENCES} total held-out sequences)")
    print(f"already bucket A: {already_resolved}")
    print(f"dispersed_converging_ambiguity category: {len(ambiguity_cases)} total")
    print(f"  -- ambiguity is the ONLY guard reason (cleanly recoverable if geometry fixed): "
         f"{len(pure_ambiguity)}")
    print(f"  -- ambiguity co-occurs with another guard (oov_name / dominant_history_contradiction, "
         f"stays guarded even if ambiguity is fixed): {len(mixed_ambiguity)}")
    mix_breakdown = Counter(tuple(sorted(r["guard_reasons"])) for r in mixed_ambiguity)
    for reasons, n in mix_breakdown.most_common():
        print(f"       {reasons}: {n}")

    # -------- primary projection: geometry fix alone, current default (robust=False) --------
    projected_a = already_resolved + len(pure_ambiguity)
    projected_b_among_gt = n_gt - projected_a - (cat_counts["all_windows_transitioning"]
                                                  + cat_counts["trailing_transitioning_run"]
                                                  + cat_counts["formation_name_mismatch"])
    # sanity: projected_b_among_gt should equal len(mixed_ambiguity)
    assert projected_b_among_gt == len(mixed_ambiguity), (projected_b_among_gt, len(mixed_ambiguity))

    # scale to the full 500-sequence denominator sec AG's before/after table used --
    # the other (500-249)=251 sequences have no clean (a,b) ground truth (3+ true
    # hops) and are NOT modeled here: the guard may fire on them too, but there is
    # no ground truth to check whether a recovered pair would even be correct, so
    # this projection is deliberately silent on that portion, not assuming zero effect.
    before_a_500 = 9  # sec AG step 2's measured "before" row
    before_b_500 = 191
    before_c_500 = 300
    projected_a_500 = before_a_500 + len(pure_ambiguity)
    projected_b_500 = before_b_500 - len(pure_ambiguity)
    projected_c_500 = before_c_500  # untouched by this projection

    print("\n" + "=" * 90)
    print("PRIMARY PROJECTION -- geometry fix alone, reduction logic unchanged (robust=False, the shipped default)")
    print("=" * 90)
    print(f"assumption: retraining STGT on corrected data_gen.py geometry eliminates near-tied dispersed/")
    print(f"converging window predictions on TRUE instances of either class, so the ambiguity guard stops")
    print(f"firing on sequences where it is currently the SOLE guard trigger. No other code change assumed.")
    print(f"\n| bucket | before (measured) | projected (geometry fix only) |")
    print(f"|---|---|---|")
    print(f"| A (Layer 1) | {before_a_500}/500 (1.8%) | {projected_a_500}/500 ({projected_a_500/500:.1%}) |")
    print(f"| B (Layer 2, guard) | {before_b_500}/500 (38.2%) | {projected_b_500}/500 ({projected_b_500/500:.1%}) |")
    print(f"| C (Layer 3, LLM) | {before_c_500}/500 (60.0%) | {projected_c_500}/500 ({projected_c_500/500:.1%}) (unchanged) |")

    # -------- projected over-abstention (approximate) --------
    # sec AG step 3 measured pipeline_v2 (robust=False) run-level over-abstention at
    # 69.8% among the 249 GT-determinable sequences. Bucket A never abstains (RULES
    # lookup always answers); bucket B always abstains (guard); bucket C (v3b-fix)
    # abstains at some rate x we don't have directly, but can back out approximately
    # by treating the 69.8% as (roughly) a sequence-count-weighted mix, ignoring the
    # fact that different sequences carry different n_runs -- an approximation,
    # stated as such, not a re-derivation from the raw per-run data.
    measured_over_abstention = 0.698
    b_frac = len(ambiguity_cases) / n_gt  # 115/249, current bucket B share among GT sequences
    # note: bucket B among GT sequences = dispersed_converging_ambiguity category only,
    # per sec AG step 1's mutually-exclusive priority-ordered categorization
    c_frac = (cat_counts["all_windows_transitioning"] + cat_counts["trailing_transitioning_run"]
             + cat_counts["formation_name_mismatch"]) / n_gt
    a_frac = already_resolved / n_gt
    assert abs(a_frac + b_frac + c_frac - 1.0) < 1e-9

    implied_c_abstention = (measured_over_abstention - b_frac) / c_frac
    print(f"\nback-derived (approximate) bucket-C-internal abstention rate x, solving "
         f"{measured_over_abstention:.1%} = {a_frac:.3f}*0 + {b_frac:.3f}*1 + {c_frac:.3f}*x: "
         f"x = {implied_c_abstention:.1%}")

    projected_a_frac = projected_a / n_gt
    projected_b_frac = len(mixed_ambiguity) / n_gt
    projected_over_abstention = projected_a_frac * 0 + projected_b_frac * 1 + c_frac * implied_c_abstention
    print(f"\nprojected over-abstention (geometry fix only, x held constant): "
         f"{projected_a_frac:.3f}*0 + {projected_b_frac:.3f}*1 + {c_frac:.3f}*{implied_c_abstention:.1%} "
         f"= {projected_over_abstention:.1%} (vs measured {measured_over_abstention:.1%})")

    print("\nEXPLICIT ASSUMPTIONS / SCOPE LIMITS:")
    print(f"  1. Only the {len(pure_ambiguity)}/249 sequences where ambiguity is the SOLE guard trigger are")
    print(f"     projected to move; the {len(mixed_ambiguity)}/249 with an additional independent guard")
    print(f"     (oov_name / dominant_history_contradiction) stay guarded -- the geometry fix does not")
    print(f"     address those triggers.")
    print(f"  2. Scope is the 249 GT-clean sequences only; the other {N_TOTAL_SEQUENCES - n_gt}/500 sequences")
    print(f"     (true 3+-hop chains, no clean (a,b) ground truth) are NOT modeled -- not assumed zero effect,")
    print(f"     simply outside what this evaluation protocol can check for correctness.")
    print(f"  3. Precision on the {len(pure_ambiguity)} newly-recovered pairs is assumed high because they")
    print(f"     already passed the CURRENT unanimity-based reduction (the conservative, already-tested path)")
    print(f"     -- unlike sec AG step 2's robust-reduction recoveries, which showed weak 16.7-25% precision.")
    print(f"  4. Does NOT include the compounding effect of also shipping robust=True (sec AG step 2's fix");
    print(f"     for trailing_transitioning_run / all_windows_transitioning, {cat_counts['trailing_transitioning_run']}")
    print(f"     + {cat_counts['all_windows_transitioning']} = "
         f"{cat_counts['trailing_transitioning_run'] + cat_counts['all_windows_transitioning']}/249 sequences,")
    print(f"     25.7%+22.5%=48.2%). That fix's own recovery precision is separately capped at ~49% by the SAME")
    print(f"     dispersed/converging confusion (sec AG step 2), so a combined number would compound two")
    print(f"     unmeasured assumptions and is deliberately not given a point estimate here.")
    print(f"  5. low_confidence (unconditional, orthogonal to this defect) could still block a small,")
    print(f"     unmeasured fraction of the {len(pure_ambiguity)} projected recoveries.")

    out = {
        "n_gt_clean": n_gt, "already_resolved": already_resolved,
        "pure_ambiguity_n": len(pure_ambiguity), "mixed_ambiguity_n": len(mixed_ambiguity),
        "mixed_ambiguity_breakdown": {str(k): v for k, v in mix_breakdown.items()},
        "before_500": {"A": before_a_500, "B": before_b_500, "C": before_c_500},
        "projected_500_geometry_fix_only": {"A": projected_a_500, "B": projected_b_500, "C": projected_c_500},
        "measured_over_abstention": measured_over_abstention,
        "implied_bucket_c_abstention_rate": implied_c_abstention,
        "projected_over_abstention_geometry_fix_only": projected_over_abstention,
    }
    out_path = REPO / "evaluation" / "upstream_fix_payoff_projection.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
