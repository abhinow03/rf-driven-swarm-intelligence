"""
AUDIT.md sec AE step 4: evaluate pipeline_v2 against v2, rules_in_prompt,
v3b-fix, and the old composite, on both the 55-case clean battery and the
degradation battery.

Clean battery: low/high/critical run at n_runs=20 (sec AD's variance finding
-- n_runs=5 std on these strata was 20-25pt, too volatile to report), medium
at n_runs=5 (already stable, sec AD/AC). Degradation battery (108 cases,
degradation.build_battery(ORIGINAL_TEST_CASES)): run at a uniform n_runs=5
for ALL systems -- sec AD's n_runs=20 finding was specifically measured on
the CLEAN battery's low/high/critical strata; extending it to every
degradation-battery stratum was never separately established as similarly
volatile, and doing so would push this already-large job (6 systems x 2
batteries) well past a tractable overnight run. Disclosed scope decision,
not a silent cut.

Only THREE LocalHFClient instances are loaded for all 5 systems (not
5-7): a base-model client with RULES.txt as its default system_prompt
(serves rules_in_prompt directly, composite's rules branch, AND
pipeline_v2's Layer 1 narrator -- Layer 1 always passes an explicit
system_prompt override per call, so the RULES.txt default never leaks into
its narration), a v3b-fix adapter client (serves v3b-fix standalone,
composite's finetuned branch, and pipeline_v2's Layer 3), and a v2 adapter
client (v2 standalone only, no sharing possible -- different weights).

Batching (sec U, validated ~2.78x speedup, equivalence-checked) is used
throughout via baselines.make_batched_run_case (v2/rules_in_prompt/v3b-fix)
and the composite-/pipeline_v2-specific batched factories defined below and
in pipeline_v2.py, or this job would not finish in a reasonable window.

Usage (run inside tmux):
    python llm_finetuning/eval_pipeline_v2.py
"""
from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES, ORIGINAL_TEST_CASES, is_abstention, match_threat  # noqa: E402
from swarm_intent.inference import build_llm_prompt  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402
from swarm_intent import pipeline_v2  # noqa: E402

from baselines import make_batched_run_case, load_rules_txt  # noqa: E402
from composite import _route, _preds_from_key_windows, RULES_BRANCH, FINETUNED_BRANCH  # noqa: E402
from degradation import build_battery  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
V2_ADAPTER = "adapters/qwen-swarm-v2"
V3B_FIX_ADAPTER = "adapters/qwen-swarm-v3b-fix"
BATCH_SIZE = 8

THREAT_ORDER = ("low", "medium", "high", "critical")
STRATIFIED_N_RUNS = 20
DEFAULT_N_RUNS = 5

NAME_TO_CASE = {c["name"]: c for c in TEST_CASES}


def t_ci95(values):
    from scipy import stats
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    return float(stats.t.ppf(0.975, df=n - 1) * se)


def normalize_threat(raw: str) -> str:
    raw = (raw or "").strip().lower()
    for level in ("low", "medium", "high", "critical"):
        if level in raw:
            return level
    return "unparsed"


def _capturing(run_case, log: dict):
    def wrapped(case):
        assessment, ctx = run_case(case)
        log[case["name"]].append(assessment)
        return assessment, ctx
    return wrapped


# --- batched run_case factories not already provided elsewhere ---

def make_battery_batched_run_case(client, battery_cases: list, n_runs: int, batch_size: int = BATCH_SIZE):
    """Single-client batched battery run_case -- v2/rules_in_prompt/v3b-fix on the
    degradation battery, whose cases carry precomputed ctx/key_windows (not
    formation_a/formation_b), so baselines.make_batched_run_case (which calls
    synth_context()) does not apply."""
    items, prompts = [], []
    for case in battery_cases:
        for _ in range(n_runs):
            items.append(case)
            prompts.append(build_llm_prompt(_preds_from_key_windows(case["key_windows"]), case["ctx"], {}))
    outs = client.complete_batch(prompts, batch_size=batch_size)
    call_idx = [0]

    def run_case(case):
        i = call_idx[0]
        call_idx[0] += 1
        return outs[i], items[i]["ctx"]

    return run_case


def _composite_resolve(items, rules_client, ft_client, batch_size):
    for branch, client in ((RULES_BRANCH, rules_client), (FINETUNED_BRANCH, ft_client)):
        idx = [i for i, it in enumerate(items) if it["branch"] == branch]
        if not idx:
            continue
        prompts = [build_llm_prompt(_preds_from_key_windows(items[i]["key_windows"]), items[i]["ctx"], {})
                  for i in idx]
        outs = client.complete_batch(prompts, batch_size=batch_size)
        for i, out in zip(idx, outs):
            items[i]["assessment"] = out


def make_composite_batched_run_case(rules_client, ft_client, branch_log: dict, test_cases: list,
                                    n_runs: int, batch_size: int = BATCH_SIZE, seed: int = 0):
    from random import Random
    from build_sft_dataset import synth_context
    rng = Random(seed)
    items = []
    for case in test_cases:
        for _ in range(n_runs):
            ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
            items.append({"case": case, "ctx": ctx, "key_windows": key_windows, "branch": _route(ctx)})
    _composite_resolve(items, rules_client, ft_client, batch_size)
    call_idx = [0]

    def run_case(case):
        item = items[call_idx[0]]
        call_idx[0] += 1
        branch_log[case["name"]] = item["branch"]
        return item["assessment"], item["ctx"]

    return run_case


def make_composite_batched_battery_run_case(rules_client, ft_client, branch_log: dict,
                                            battery_cases: list, n_runs: int, batch_size: int = BATCH_SIZE):
    items = []
    for case in battery_cases:
        branch = _route(case["ctx"])
        for _ in range(n_runs):
            items.append({"case": case, "ctx": case["ctx"], "key_windows": case["key_windows"], "branch": branch})
    _composite_resolve(items, rules_client, ft_client, batch_size)
    call_idx = [0]

    def run_case(case):
        item = items[call_idx[0]]
        call_idx[0] += 1
        branch_log[case["name"]] = item["branch"]
        return item["assessment"], item["ctx"]

    return run_case


# --- per-system run_case maker signatures: maker(cases, n_runs, log) -> run_case ---

def build_makers(rules_client, ft_client, v2_client, class_freq):
    return {
        "v2": lambda cases, n_runs, log: make_batched_run_case(v2_client, cases, n_runs, BATCH_SIZE, seed=0),
        "rules_in_prompt": lambda cases, n_runs, log: make_batched_run_case(rules_client, cases, n_runs,
                                                                            BATCH_SIZE, seed=0),
        "v3b-fix": lambda cases, n_runs, log: make_batched_run_case(ft_client, cases, n_runs, BATCH_SIZE, seed=0),
        "composite": lambda cases, n_runs, log: make_composite_batched_run_case(
            rules_client, ft_client, log, cases, n_runs, BATCH_SIZE, seed=0),
        "pipeline_v2": lambda cases, n_runs, log: pipeline_v2.make_pipeline_v2_batched_run_case(
            rules_client, ft_client, class_freq, log, cases, n_runs, BATCH_SIZE, seed=0),
    }


def build_battery_makers(rules_client, ft_client, v2_client, class_freq):
    return {
        "v2": lambda cases, n_runs, log: make_battery_batched_run_case(v2_client, cases, n_runs, BATCH_SIZE),
        "rules_in_prompt": lambda cases, n_runs, log: make_battery_batched_run_case(
            rules_client, cases, n_runs, BATCH_SIZE),
        "v3b-fix": lambda cases, n_runs, log: make_battery_batched_run_case(ft_client, cases, n_runs, BATCH_SIZE),
        "composite": lambda cases, n_runs, log: make_composite_batched_battery_run_case(
            rules_client, ft_client, log, cases, n_runs, BATCH_SIZE),
        "pipeline_v2": lambda cases, n_runs, log: pipeline_v2.make_pipeline_v2_batched_battery_run_case(
            rules_client, ft_client, class_freq, log, cases, n_runs, BATCH_SIZE),
    }


def run_level_accuracy(raw: dict, names: list, n_runs: int) -> list:
    run_accs = []
    for r in range(n_runs):
        hits, scored = 0, 0
        for name in names:
            case = NAME_TO_CASE.get(name)
            a = raw[name][r]
            if is_abstention(a.get("likely_intent", "")):
                continue
            scored += 1
            if case is not None and match_threat(a.get("threat_level", ""), case["expected_threat"]):
                hits += 1
        if scored:
            run_accs.append(hits / scored)
    return run_accs


def escalation_breakdown(raw: dict, names: list, n_runs: int) -> Counter:
    counts = Counter()
    for name in names:
        expected = NAME_TO_CASE[name]["expected_threat"]
        for r in range(n_runs):
            a = raw[name][r]
            if is_abstention(a.get("likely_intent", "")):
                counts["abstained"] += 1
                continue
            pred = normalize_threat(a.get("threat_level", ""))
            if pred == expected:
                counts["correct"] += 1
            elif THREAT_ORDER.index(pred) < THREAT_ORDER.index(expected) if pred in THREAT_ORDER else False:
                counts["under_escalated"] += 1
            elif pred in THREAT_ORDER:
                counts["over_escalated"] += 1
            else:
                counts["unparsed"] += 1
    return counts


def main():
    rules_client = LocalHFClient(BASE_MODEL, adapter_path=None, temperature=0.3, system_prompt=load_rules_txt())
    ft_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V3B_FIX_ADAPTER), temperature=0.3)
    v2_client = LocalHFClient(BASE_MODEL, adapter_path=str(REPO / V2_ADAPTER), temperature=0.3)
    class_freq = pipeline_v2.default_class_freq()
    print(f"pipeline_v2 Layer 3 class_freq (v3b-fix's own training file): {class_freq}")

    makers = build_makers(rules_client, ft_client, v2_client, class_freq)
    battery_makers = build_battery_makers(rules_client, ft_client, v2_client, class_freq)

    stratified_cases = [c for c in TEST_CASES if c["expected_threat"] in ("low", "high", "critical")]
    medium_cases = [c for c in TEST_CASES if c["expected_threat"] == "medium"]
    degradation_battery = build_battery(ORIGINAL_TEST_CASES)
    degradation_cases = [c for axis_cases in degradation_battery.values() for c in axis_cases]
    print(f"clean battery: {len(stratified_cases)} stratified (n_runs={STRATIFIED_N_RUNS}) + "
         f"{len(medium_cases)} medium (n_runs={DEFAULT_N_RUNS}); degradation battery: "
         f"{len(degradation_cases)} cases (n_runs={DEFAULT_N_RUNS})")

    total_units = 5 * (len(stratified_cases) * STRATIFIED_N_RUNS + len(medium_cases) * DEFAULT_N_RUNS
                       + len(degradation_cases) * DEFAULT_N_RUNS)
    reporter = Reporter("eval_pipeline_v2", total_units, rate_hint=0.9)

    out_dir = REPO / "evaluation"
    results = {}
    for label in ("v2", "rules_in_prompt", "v3b-fix", "composite", "pipeline_v2"):
        print(f"\n=== {label}: clean battery stratified (n_runs={STRATIFIED_N_RUNS}) ===")
        log_clean, aux_log_hi = defaultdict(list), defaultdict(list)
        run_case_hi = makers[label](stratified_cases, STRATIFIED_N_RUNS, aux_log_hi)
        res_hi = evaluate_llm(_capturing(run_case_hi, log_clean), stratified_cases, judge_client=None,
                              n_runs=STRATIFIED_N_RUNS, progress_reporter=reporter)

        print(f"=== {label}: clean battery medium (n_runs={DEFAULT_N_RUNS}) ===")
        aux_log_med = defaultdict(list)
        run_case_med = makers[label](medium_cases, DEFAULT_N_RUNS, aux_log_med)
        res_med = evaluate_llm(_capturing(run_case_med, log_clean), medium_cases, judge_client=None,
                               n_runs=DEFAULT_N_RUNS, progress_reporter=reporter)

        print(f"=== {label}: degradation battery (n_runs={DEFAULT_N_RUNS}) ===")
        log_degradation, aux_log_deg = defaultdict(list), defaultdict(list)
        run_case_deg = battery_makers[label](degradation_cases, DEFAULT_N_RUNS, aux_log_deg)
        res_deg = evaluate_llm(_capturing(run_case_deg, log_degradation), degradation_cases, judge_client=None,
                               n_runs=DEFAULT_N_RUNS, progress_reporter=reporter)

        results[label] = {
            "clean_raw": log_clean, "clean_hi_aggregate": res_hi["aggregate"],
            "clean_med_aggregate": res_med["aggregate"], "degradation_raw": log_degradation,
            "degradation_aggregate": res_deg["aggregate"],
            "aux_log_hi": {k: v for k, v in aux_log_hi.items()} if aux_log_hi else None,
            "aux_log_deg": {k: v for k, v in aux_log_deg.items()} if aux_log_deg else None,
        }
        (out_dir / f"eval_pipeline_v2_{label}.json").write_text(json.dumps({
            "clean_hi_aggregate": res_hi["aggregate"], "clean_med_aggregate": res_med["aggregate"],
            "degradation_aggregate": res_deg["aggregate"], "clean_raw": log_clean,
            "degradation_raw": log_degradation,
        }, indent=2))

    reporter.status = "done"
    reporter._write()
    del rules_client, ft_client, v2_client
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    # ================= REPORTING =================
    print("\n\n" + "=" * 100)
    print("STEP 4: clean-battery threat accuracy by stratum, mean +/- 95% CI")
    print("=" * 100)
    print("| system | low | medium | high | critical |")
    print("|---|---|---|---|---|")
    for label in ("v2", "rules_in_prompt", "v3b-fix", "composite", "pipeline_v2"):
        raw = results[label]["clean_raw"]
        cells = []
        for stratum in THREAT_ORDER:
            names = [c["name"] for c in (stratified_cases if stratum != "medium" else medium_cases)
                    if c["expected_threat"] == stratum]
            n_runs = STRATIFIED_N_RUNS if stratum != "medium" else DEFAULT_N_RUNS
            accs = run_level_accuracy(raw, names, n_runs)
            if accs:
                cells.append(f"{np.mean(accs):.1%}+/-{t_ci95(accs):.1%}")
            else:
                cells.append("n/a")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 100)
    print("STEP 4: over- vs under-escalation direction, high+critical, clean battery")
    print("=" * 100)
    print("| system | correct | under_escalated | over_escalated | abstained |")
    print("|---|---|---|---|---|")
    hc_names = [c["name"] for c in stratified_cases if c["expected_threat"] in ("high", "critical")]
    for label in ("v2", "rules_in_prompt", "v3b-fix", "composite", "pipeline_v2"):
        counts = escalation_breakdown(results[label]["clean_raw"], hc_names, STRATIFIED_N_RUNS)
        total = sum(counts.values())
        row = " | ".join(f"{counts.get(k,0)}/{total} ({counts.get(k,0)/total:.1%})"
                         for k in ("correct", "under_escalated", "over_escalated", "abstained"))
        print(f"| {label} | {row} |")

    print("\n" + "=" * 100)
    print("STEP 4: degradation battery -- accuracy_when_answerable / abstention_rate_when_unanswerable / "
         "over_abstention_rate")
    print("=" * 100)
    print("| system | acc_when_answerable | abstention_when_unanswerable | over_abstention_rate |")
    print("|---|---|---|---|")
    for label in ("v2", "rules_in_prompt", "v3b-fix", "composite", "pipeline_v2"):
        agg = results[label]["degradation_aggregate"]
        def fmt(k):
            v = agg.get(k)
            return f"{v:.1%}" if v is not None else "n/a"
        print(f"| {label} | {fmt('accuracy_when_answerable')} | {fmt('abstention_rate_when_unanswerable')} | "
             f"{fmt('over_abstention_rate')} |")

    print("\n" + "=" * 100)
    print("STEP 4: pipeline_v2 layer-firing rates and per-layer accuracy")
    print("=" * 100)
    for battery_name, log_field, cases_list, layer_log in (
        ("clean (stratified)", "aux_log_hi", stratified_cases, results["pipeline_v2"]["aux_log_hi"]),
        ("degradation", "aux_log_deg", degradation_cases, results["pipeline_v2"]["aux_log_deg"]),
    ):
        if not layer_log:
            continue
        layer_counts = Counter()
        layer_hits, layer_total = Counter(), Counter()
        raw = results["pipeline_v2"]["clean_raw"] if "clean" in battery_name else results["pipeline_v2"]["degradation_raw"]
        name_to_case_local = {c["name"]: c for c in cases_list}
        for name, entries in layer_log.items():
            case = name_to_case_local.get(name)
            for r, entry in enumerate(entries):
                layer_counts[entry["layer"]] += 1
                if case is not None and case.get("has_ground_truth", True):
                    a = raw[name][r]
                    if not is_abstention(a.get("likely_intent", "")):
                        layer_total[entry["layer"]] += 1
                        if match_threat(a.get("threat_level", ""), case["expected_threat"]):
                            layer_hits[entry["layer"]] += 1
        total = sum(layer_counts.values())
        print(f"\n-- {battery_name} battery (n={total} case-run units) --")
        print("| layer | share of traffic | threat accuracy within that layer |")
        print("|---|---|---|")
        for layer in ("layer1_deterministic", "layer2_guard", "layer3_llm"):
            n = layer_counts.get(layer, 0)
            acc = (layer_hits[layer] / layer_total[layer]) if layer_total.get(layer) else None
            acc_str = f"{acc:.1%} (n={layer_total[layer]})" if acc is not None else "n/a"
            print(f"| {layer} | {n}/{total} ({n/total:.1%}) | {acc_str} |")

    print(f"\nSaved per-system detail to evaluation/eval_pipeline_v2_<system>.json")


if __name__ == "__main__":
    main()
