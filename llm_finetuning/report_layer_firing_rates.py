"""
AUDIT.md sec AF step 1: sec AE's headline table reports pipeline_v2 at 100.0%
threat accuracy on all four classes of the 55-case clean battery. That number
is meaningless on its own without knowing what fraction of that battery even
reaches a layer where a mistake is POSSIBLE -- if Layer 1 (deterministic RULES
dict lookup) fires on ~100% of it, the 100% accuracy figure is scoring a
dictionary against dictionary-derived ground truth, a tautology, not a
generalization result.

Bucket assignment (src/swarm_intent/coverage.classify_ctx) is a PURE FUNCTION
of ctx text -- it does not depend on any LLM's sampled output at all, so this
is computed directly and deterministically, with no GPU and no dependence on
the actual eval run's sampling (this is not a re-measurement subject to
sampling noise; it is the exact same routing decision that eval_pipeline_v2.py
produced, recomputed from first principles as a standalone check).

Usage:
    python llm_finetuning/report_layer_firing_rates.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.prompts import TEST_CASES, ORIGINAL_TEST_CASES  # noqa: E402
from swarm_intent.coverage import classify_ctx, BUCKET_A, BUCKET_B, BUCKET_C  # noqa: E402

from build_sft_dataset import synth_context  # noqa: E402
from degradation import build_battery  # noqa: E402


def firing_rates(cases_ctx_pairs):
    counts = Counter()
    for ctx, key_windows in cases_ctx_pairs:
        bucket_info = classify_ctx(ctx, key_windows)
        counts[bucket_info["bucket"]] += 1
    total = sum(counts.values())
    return counts, total


def main():
    from random import Random

    # Clean battery: same shared-rng protocol (seed=0) as every headline eval
    # in this project -- ctx text is byte-identical to what eval_pipeline_v2.py
    # actually fed classify_ctx (bucket routing doesn't depend on the run index,
    # only on the case's own ctx draw at its position in the walk).
    rng = Random(0)
    clean_pairs = [synth_context(c["formation_a"], c["formation_b"], rng) for c in TEST_CASES]
    clean_counts, clean_total = firing_rates(clean_pairs)

    degradation_battery = build_battery(ORIGINAL_TEST_CASES)
    degradation_cases = [c for axis_cases in degradation_battery.values() for c in axis_cases]
    deg_pairs = [(c["ctx"], c["key_windows"]) for c in degradation_cases]
    deg_counts, deg_total = firing_rates(deg_pairs)

    print(f"=== clean battery (n={clean_total}) ===")
    for b in (BUCKET_A, BUCKET_B, BUCKET_C):
        k = clean_counts.get(b, 0)
        print(f"  {b}: {k}/{clean_total} ({k/clean_total:.1%})")

    print(f"\n=== degradation battery (n={deg_total}) ===")
    for b in (BUCKET_A, BUCKET_B, BUCKET_C):
        k = deg_counts.get(b, 0)
        print(f"  {b}: {k}/{deg_total} ({k/deg_total:.1%})")

    print("\n=== verdict ===")
    clean_a_pct = clean_counts.get(BUCKET_A, 0) / clean_total
    if clean_a_pct > 0.95:
        print(f"Layer 1 fires on {clean_a_pct:.1%} of the 55-case clean battery. "
             f"sec AE's 100.0% pipeline_v2 accuracy figure on this battery is a "
             f"CONSTRUCTION ARTEFACT (dictionary scored against dictionary-derived "
             f"ground truth on a battery designed to be entirely rule-table-resolvable), "
             f"NOT a generalization result. It confirms Layer 1's decision-field overwrite "
             f"logic is bug-free -- nothing more.")
    else:
        print(f"Layer 1 fires on {clean_a_pct:.1%} -- not a near-total-tautology; "
             f"the 100% figure has some genuine mixed-layer support.")


if __name__ == "__main__":
    main()
