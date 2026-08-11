"""
AUDIT.md V5 Phase 1 step 3, step 3: full coverage report on the final
regenerated corpus (data/sft_train_v5_phase1*.jsonl, 12,001 rows).

Two things measured, both from the tactical-context TEXT embedded in each
row's user message (no per-row structured metadata exists -- see step 0's
provenance audit) via regex extraction matching synth_context()'s exact
phrasing:

  1. (pair, threat) coverage matrix vs. the fallback stratification target
     (STRATA_TARGETS in build_sft_dataset.py: low/medium/high 3600 each,
     critical 1200 across its 2 pairs).
  2. Narrative-combination coverage per pair: distinct
     (velocity_trend, stability_trend, spread_trend, role_differentiation)
     tuples actually observed per (form_a, form_b) pair, against a >=8
     distinct-combinations target (docs/RULES_EXTENSION_PROPOSAL.md's own
     "~150 rows per narrative-combination cell at 8 combos/pair" figure for
     the critical stratum). The full narrative space is 3(velocity) x
     3(stability) x 3(spread) x 2(role) = 54 possible combinations per pair;
     8 is a floor, not the ceiling.

Also reports source-model distribution -- see the finding below: this
diverges from what a prior task's instructions assumed.

Usage:
    python llm_finetuning/report_phase1_coverage.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from swarm_intent.coverage import _extract_pair_from_ctx  # noqa: E402

sys.path.insert(0, str(REPO / "llm_finetuning"))
from build_sft_dataset import RULES, STRATA_TARGETS  # noqa: E402

FILES = {
    "train": REPO / "data" / "sft_train_v5_phase1.jsonl",
    "val": REPO / "data" / "sft_train_v5_phase1_val.jsonl",
    "mining": REPO / "data" / "sft_train_v5_phase1_mining.jsonl",
}

_VEL_RE = re.compile(r"Velocity trend:\s*(\w+)")
_STAB_RE = re.compile(r"Formation stability:\s*(\w+)")
_SPREAD_RE = re.compile(r"Spread dynamics:\s*([\w\s()]+?)\s*\(mean")
_ROLE_RE = re.compile(r"Role differentiation:\s*([\w\s]+?)\n")

CRITICAL_PAIRS = {p for p, (t, *_r) in RULES.items() if t == "critical"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def narrative_combo(ctx: str):
    vel = _VEL_RE.search(ctx)
    stab = _STAB_RE.search(ctx)
    spread = _SPREAD_RE.search(ctx)
    role = _ROLE_RE.search(ctx)
    return (vel.group(1) if vel else None,
           stab.group(1) if stab else None,
           spread.group(1).strip() if spread else None,
           role.group(1).strip() if role else None)


def main():
    all_rows = []
    for split, path in FILES.items():
        all_rows.extend((split, r) for r in load_jsonl(path))

    pair_rows = defaultdict(list)   # pair -> list of ctx
    unresolved = 0
    for split, r in all_rows:
        ctx = r["messages"][0]["content"]
        pair = _extract_pair_from_ctx(ctx)
        if pair is None:
            unresolved += 1
            continue
        pair_rows[pair].append(ctx)

    print(f"total rows: {len(all_rows)}  (pair-extractable: {sum(len(v) for v in pair_rows.values())}, "
         f"unresolved: {unresolved})")

    # ================= 1. (pair, threat) coverage matrix =================
    print("\n" + "=" * 100)
    print("STEP 3a: (pair, threat) coverage vs. fallback stratification target")
    print("=" * 100)
    tier_counts = Counter()
    tier_targets = STRATA_TARGETS
    rows_per_pair_in_rules = Counter()
    pairs_not_in_rules = Counter()

    for pair, ctxs in pair_rows.items():
        if pair in RULES:
            threat = RULES[pair][0]
            tier_counts[threat] += len(ctxs)
            rows_per_pair_in_rules[pair] += len(ctxs)
        else:
            pairs_not_in_rules[pair] += len(ctxs)

    print("| tier | target | actual | delta | % of target |")
    print("|---|---|---|---|---|")
    for tier in ("low", "medium", "high", "critical"):
        target = tier_targets[tier]
        actual = tier_counts.get(tier, 0)
        print(f"| {tier} | {target} | {actual} | {actual - target:+d} | {actual/target:.1%} |")

    if pairs_not_in_rules:
        print(f"\n{sum(pairs_not_in_rules.values())} rows extracted a (form_a,form_b) pair NOT in "
             f"RULES ({len(pairs_not_in_rules)} distinct pairs) -- unexpected, since "
             f"build_stratified_pairs() only ever draws from RULES.keys(). Top 5:")
        for pair, n in pairs_not_in_rules.most_common(5):
            print(f"  {pair}: {n}")

    n_pairs_in_rules = len(RULES)
    n_pairs_seen = len(rows_per_pair_in_rules)
    print(f"\nRULES has {n_pairs_in_rules} total pairs; {n_pairs_seen} appear at least once in "
         f"this corpus ({n_pairs_seen/n_pairs_in_rules:.1%} pair coverage).")
    zero_cov = [p for p in RULES if p not in rows_per_pair_in_rules]
    if zero_cov:
        print(f"Pairs with ZERO rows: {zero_cov}")

    # ================= 2. narrative-combination coverage =================
    print("\n" + "=" * 100)
    print("STEP 3b: narrative-combination coverage per pair (target >=8 distinct combos)")
    print("=" * 100)

    combo_counts = {}
    for pair, ctxs in pair_rows.items():
        if pair not in RULES:
            continue
        combos = set(narrative_combo(ctx) for ctx in ctxs)
        combo_counts[pair] = (len(combos), len(ctxs))

    below_target = [(p, c, n) for p, (c, n) in combo_counts.items() if c < 8]
    print(f"{len(combo_counts)} pairs measured. {len(below_target)} pairs fall below the "
         f">=8 distinct-combination target.")
    if below_target:
        print("\n| pair | distinct combos | rows | threat tier |")
        print("|---|---|---|---|")
        for pair, combos, n in sorted(below_target, key=lambda x: x[1]):
            tier = RULES[pair][0]
            print(f"| {pair} | {combos} | {n} | {tier} |")

    print("\n--- critical stratum specifically (flagged for extra attention) ---")
    print("| pair | rows | distinct combos | rows/combo | vs >=8 target |")
    print("|---|---|---|---|---|")
    for pair in sorted(CRITICAL_PAIRS):
        combos, n = combo_counts.get(pair, (0, 0))
        status = "PASS" if combos >= 8 else "BELOW TARGET"
        rows_per_combo = n / combos if combos else float("nan")
        print(f"| {pair} | {n} | {combos} | {rows_per_combo:.1f} | {status} |")

    all_combo_counts = [c for c, n in combo_counts.values()]
    print(f"\nAcross all {len(combo_counts)} pairs: mean {sum(all_combo_counts)/len(all_combo_counts):.1f}, "
         f"min {min(all_combo_counts)}, max {max(all_combo_counts)} distinct combos "
         f"(54 = full 3x3x3x2 space per pair).")

    # ================= 3. source-model distribution =================
    print("\n" + "=" * 100)
    print("STEP 3c: source-model / teacher distribution")
    print("=" * 100)
    print("No per-row source_model field exists (step 0's provenance audit already established "
         "this -- rows are exactly {messages:[user,assistant]}). Reconstructed from V5_LOG.md's "
         "documented history instead of measured per-row:")
    print("  - Phase 1 step 0: Groq RETIRED entirely as the teacher provider before any Phase 1 "
         "generation ran; replaced with a single NVIDIA NIM-hosted teacher "
         "(nvidia/nemotron-3-super-120b-a12b, NvidiaClient).")
    print("  - No teacher-model switch occurred at any point after that -- every generation cycle "
         "(the original run + all 6 resume/regeneration cycles) used the same single teacher model.")
    print("  - FINDING: this diverges from an instruction elsewhere assuming '>=3 distinct Groq "
         "families were actually used' -- that assumption does not match this corpus's actual "
         "build history. Exactly 1 distinct teacher model family (NVIDIA Nemotron) was ever used; "
         "Groq was never called during Phase 1 at all. Reporting this plainly rather than silently "
         "reconciling it.")


if __name__ == "__main__":
    main()
