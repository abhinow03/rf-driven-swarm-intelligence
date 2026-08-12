"""
Evaluation pipeline for v5-a (and, later, v5-b/c/d) fine-tuned checkpoints.

Metrics, matching this project's established conventions -- checked against
AUDIT.md / src/swarm_intent/llm/evaluate.py / llm_finetuning/eval_real_stgt_output.py
before writing anything new here, per the standing instruction not to invent
metrics ad hoc:

  - per-class threat accuracy (low/medium/high/critical), each with n and a
    Wilson 95% CI. An explicit low-n caveat is printed and recorded whenever
    n<10 -- critical has been n=2 for most of this project's history
    (AUDIT.md: "not statistically meaningful, reported anyway"); never report
    it as a bare percentage without that context.
  - accuracy_when_answerable / abstention_rate_when_unanswerable /
    over_abstention_rate as three SEPARATE fields, never blended into one
    number (src/swarm_intent/llm/evaluate.py's module docstring documents the
    original bug this avoids: an always-abstaining case scored 0.0 accuracy
    AND 1.0 hallucination_rate simultaneously -- double-penalizing the same
    abstention).
  - under-escalation and over-escalation reported SEPARATELY and directionally
    (AUDIT.md sec AC/AD: under-escalation dominates every LLM-in-the-loop
    system measured so far, by a wide margin -- e.g. 15.7% under vs 4.8% over
    on pipeline_v2's real-output eval; collapsing both into one "escalation
    error" number hides which direction actually needs fixing).
  - JSON schema-validity rate: fraction of raw outputs that parse as JSON and
    contain every OUTPUT_SCHEMA key.
  - critical-pair tactical accuracy: same low-n caveat as the per-class table.

Two modes, both zero-GPU while v5-a is still training:

  --mock: fully synthetic, no model/tokenizer/checkpoint touched at all. A
    stub in-process predictor returns synthetic outputs (some correct, some
    deliberately wrong in each direction, one schema-invalid) so every metric
    branch is demonstrated end-to-end.

  --adapter <path> (default checkpoints/v5_sft/, the real training output
    dir) --real: REAL checkpoint resolution (resolve_adapter_path()) + REAL
    4-bit quantized model load (LocalHFClient, same config as training) +
    REAL generation against --phase4-set (default
    evaluation/phase4_eval_set.json). Prints a 3-case sanity check FIRST
    (raw output for 3 hand-picked cases) before running the full batch --
    this is a real gate, not cosmetic: read the printed output before
    trusting anything downstream.

  --adapter <path> (no --real): the zero-GPU dry-run path used while v5-a
    was still training -- REAL checkpoint resolution + REAL meta-device
    model/LoRA construction (load_model_dry_run(), torch.cuda.
    memory_allocated()==0 asserted), generation still stubbed. Kept for
    regression-testing the path-resolution logic without touching the GPU.

--judge: also scores with the advisory-only judge (default_judge_client(),
  src/swarm_intent/llm/client.py -- NvidiaClient/meta-llama-3.1-70b-instruct,
  NOT Nemotron). Adds judge_scores/judge_overall_mean to the output, never
  reads them back into any other field -- see tests/test_eval_sft_v5.py for
  the check that headline metrics are byte-identical with/without --judge.

Usage:
    python llm_finetuning/eval_sft_v5.py --mock
    python llm_finetuning/eval_sft_v5.py --adapter checkpoints/v5_sft/ --judge

Output: logs/eval_v5_results.json -- see OUTPUT_SCHEMA_DOC below for the exact
JSON structure written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.prompts import is_abstention, match_threat  # noqa: E402
from swarm_intent.llm.client import default_judge_client, JUDGE_MODEL  # noqa: E402
from swarm_intent.inference import OUTPUT_SCHEMA  # noqa: E402

DEFAULT_OUTPUT_DIR = "checkpoints/v5_sft/"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

THREAT_ORDER = ("low", "medium", "high", "critical")
LOW_N_THRESHOLD = 10

# The 2 distinct critical RULES pairs (both directions of the same compound
# escalation event) -- see docs/RULES_EXTENSION_PROPOSAL.md for the full
# context on why this tier is so thin.
CRITICAL_PAIRS = {("converging", "encirclement"), ("encirclement", "converging")}

OUTPUT_SCHEMA_DOC = """
logs/eval_v5_results.json structure:
{
  "meta": {"mode": "mock"|"real", "adapter_path": str|null, "n_cases": int,
           "generated_at_utc": str},
  "per_class_threat_accuracy": {
    "<low|medium|high|critical>": {"n": int, "accuracy": float|null,
                                    "ci_lo": float|null, "ci_hi": float|null,
                                    "low_n_caveat": bool}
  },
  "answerability": {
    "accuracy_when_answerable": float|null, "n_answerable": int,
    "abstention_rate_when_unanswerable": float|null, "n_unanswerable": int,
    "over_abstention_rate": float|null
  },
  "escalation": {"n": int, "correct": float, "under_escalated": float,
                 "over_escalated": float, "abstained": float},
  "schema_validity_rate": {"rate": float, "n_valid": int, "n_total": int},
  "critical_pair_tactical_accuracy": {"n": int, "accuracy": float|null,
                                       "low_n_caveat": bool},
  "judge_scores": {"<case_name>": int, ...},  # ADVISORY ONLY, {} if no judge
  "judge_overall_mean": float|null            # ADVISORY ONLY, never read by anything above
}
"""


def wilson_ci95(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    import math
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - margin), min(1.0, center + margin))


def mock_cases() -> list[dict]:
    """Small synthetic battery covering all 4 threat tiers plus one
    unanswerable (multi-hop) case, so every metric branch below is exercised.
    NOT drawn from real STGT output or the real corpus -- purely for pipeline
    validation in mock mode."""
    return [
        {"name": "steady_low", "pair": ("column", "column"), "expected_threat": "low",
         "has_ground_truth": True},
        {"name": "transition_medium", "pair": ("shield", "diamond"), "expected_threat": "medium",
         "has_ground_truth": True},
        {"name": "transition_high", "pair": ("column", "converging"), "expected_threat": "high",
         "has_ground_truth": True},
        {"name": "critical_pair", "pair": ("converging", "encirclement"), "expected_threat": "critical",
         "has_ground_truth": True},
        {"name": "under_escalation_case", "pair": ("dispersed", "converging"), "expected_threat": "high",
         "has_ground_truth": True},
        {"name": "over_escalation_case", "pair": ("column", "dispersed"), "expected_threat": "low",
         "has_ground_truth": True},
        {"name": "over_abstention_case", "pair": ("shield", "column"), "expected_threat": "medium",
         "has_ground_truth": True},
        {"name": "multi_hop_unanswerable", "pair": None, "expected_threat": None,
         "has_ground_truth": False},
    ]


def mock_predict(case: dict) -> str:
    """Returns a raw JSON string (as if decoded from the model), deliberately
    varied so every downstream metric branch is exercised: correct answers,
    one under-escalation, one over-escalation, one over-abstention (abstains
    on an answerable case), one correct abstention (unanswerable case), and
    one schema-invalid / malformed output."""
    base = {
        "situation_summary": "mock", "threat_reasoning": "mock",
        "confidence_in_assessment": "high", "key_indicators": ["a", "b", "c"],
        "follow_up_watch": "mock", "recommended_action": "monitor",
    }
    if case["name"] == "under_escalation_case":
        return json.dumps({**base, "threat_level": "low", "likely_intent": "surveillance"})
    if case["name"] == "over_escalation_case":
        return json.dumps({**base, "threat_level": "high", "likely_intent": "approach"})
    if case["name"] == "over_abstention_case":
        return json.dumps({**base, "threat_level": "unknown", "likely_intent": "unknown"})
    if case["name"] == "multi_hop_unanswerable":
        return json.dumps({**base, "threat_level": "unknown", "likely_intent": "unknown"})
    if case["name"] == "critical_pair":
        # deliberately malformed (missing closing brace) to exercise
        # schema_validity_rate < 100%
        return '{"threat_level": "critical", "likely_intent": "encircle"'
    return json.dumps({**base, "threat_level": case["expected_threat"], "likely_intent": "surveillance"})


def parse_and_validate(raw: str) -> tuple[dict | None, bool]:
    """Returns (parsed_dict_or_None, schema_valid). schema_valid requires both
    valid JSON and every OUTPUT_SCHEMA key present."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(parsed, dict) or not set(OUTPUT_SCHEMA.keys()) <= parsed.keys():
        return parsed if isinstance(parsed, dict) else None, False
    return parsed, True


def evaluate(cases: list[dict], predict_fn, judge_client=None) -> dict:
    """judge_client: optional, ADVISORY ONLY (src/swarm_intent/llm/client.py's
    default_judge_client()). When given, each case's per_case entry gets an
    additional judge_overall_mean field -- every other field/metric below is
    computed identically whether judge_client is None or a real client. This
    is verified, not just asserted: see tests/test_eval_sft_v5.py."""
    raw_by_case = {c["name"]: predict_fn(c) for c in cases}
    parsed_by_case = {}
    valid_by_case = {}
    for name, raw in raw_by_case.items():
        parsed, valid = parse_and_validate(raw)
        parsed_by_case[name] = parsed
        valid_by_case[name] = valid

    judge_scores_by_case = {}
    if judge_client is not None:
        judgeable = [c for c in cases if parsed_by_case[c["name"]] is not None]
        prompts = [f"Rate this tactical assessment's quality from 1-5. "
                  f"Return JSON: {{\"overall_score\": <int>}}.\n\n{json.dumps(parsed_by_case[c['name']])}"
                  for c in judgeable]
        complete_batch = getattr(judge_client, "complete_batch", None)
        try:
            if complete_batch is not None and len(prompts) > 1:
                # Real clients (NvidiaClient/GroqClient) batch via a thread pool --
                # network-bound, GIL-released during I/O -- 1000 sequential judge
                # calls would otherwise dominate total runtime (confirmed: this was
                # the bottleneck on the first real Phase 4 run, generation itself
                # took 29min, judge scoring alone took far longer at one-at-a-time).
                judge_results = complete_batch(prompts)
            else:
                judge_results = [judge_client.complete(p) for p in prompts]
        except Exception:
            judge_results = [{}] * len(prompts)  # advisory only -- a batch-level failure
            # must never affect headline metrics; per-item failures inside complete_batch
            # already degrade to {"error": ...} per-item, not raise, for real clients
        for c, judge_result in zip(judgeable, judge_results):
            if isinstance(judge_result, dict) and "overall_score" in judge_result:
                judge_scores_by_case[c["name"]] = judge_result["overall_score"]

    n_valid = sum(valid_by_case.values())
    schema_validity_rate = {"rate": n_valid / len(cases) if cases else 0.0,
                            "n_valid": n_valid, "n_total": len(cases)}

    gt_cases = [c for c in cases if c["has_ground_truth"]]
    no_gt_cases = [c for c in cases if not c["has_ground_truth"]]

    # --- per-class threat accuracy ---
    per_class = {}
    for level in THREAT_ORDER:
        level_cases = [c for c in gt_cases if c["expected_threat"] == level]
        hits, scored = 0, 0
        for c in level_cases:
            parsed = parsed_by_case[c["name"]]
            if parsed is None:
                continue  # unparseable -- excluded from accuracy, counted in schema_validity_rate only
            intent = parsed.get("likely_intent", "")
            if is_abstention(intent):
                continue  # abstained -- excluded from accuracy denominator, see answerability section
            scored += 1
            if match_threat(parsed.get("threat_level", ""), level):
                hits += 1
        if scored:
            p, lo, hi = wilson_ci95(hits, scored)
            per_class[level] = {"n": scored, "accuracy": p, "ci_lo": lo, "ci_hi": hi,
                                "low_n_caveat": scored < LOW_N_THRESHOLD}
        else:
            per_class[level] = {"n": 0, "accuracy": None, "ci_lo": None, "ci_hi": None,
                                "low_n_caveat": True}

    # --- answerability: accuracy_when_answerable / abstention_rate_when_unanswerable / over_abstention_rate ---
    answerable_hits, answerable_scored, answerable_abstained = 0, 0, 0
    for c in gt_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None:
            continue
        intent = parsed.get("likely_intent", "")
        if is_abstention(intent):
            answerable_abstained += 1
            continue
        answerable_scored += 1
        if match_threat(parsed.get("threat_level", ""), c["expected_threat"]):
            answerable_hits += 1
    accuracy_when_answerable = answerable_hits / answerable_scored if answerable_scored else None
    over_abstention_rate = (answerable_abstained / len(gt_cases)) if gt_cases else None

    unanswerable_abstained, unanswerable_total = 0, 0
    for c in no_gt_cases:
        parsed = parsed_by_case[c["name"]]
        unanswerable_total += 1
        if parsed is not None and is_abstention(parsed.get("likely_intent", "")):
            unanswerable_abstained += 1
    abstention_rate_when_unanswerable = (unanswerable_abstained / unanswerable_total
                                         if unanswerable_total else None)

    answerability = {
        "accuracy_when_answerable": accuracy_when_answerable, "n_answerable": answerable_scored,
        "abstention_rate_when_unanswerable": abstention_rate_when_unanswerable,
        "n_unanswerable": unanswerable_total,
        "over_abstention_rate": over_abstention_rate,
    }

    # --- escalation direction: correct / under / over / abstained, partition of gt_cases ---
    correct = under_esc = over_esc = abstained = 0
    for c in gt_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None or is_abstention(parsed.get("likely_intent", "")):
            abstained += 1
            continue
        pred = parsed.get("threat_level", "").strip().lower()
        expected = c["expected_threat"]
        if pred == expected:
            correct += 1
        elif pred in THREAT_ORDER and THREAT_ORDER.index(pred) < THREAT_ORDER.index(expected):
            under_esc += 1
        elif pred in THREAT_ORDER:
            over_esc += 1
        else:
            abstained += 1  # unparseable/garbage threat_level treated as non-answer, not scored either direction
    n_gt = len(gt_cases)
    escalation = {"n": n_gt,
                 "correct": correct / n_gt if n_gt else 0.0,
                 "under_escalated": under_esc / n_gt if n_gt else 0.0,
                 "over_escalated": over_esc / n_gt if n_gt else 0.0,
                 "abstained": abstained / n_gt if n_gt else 0.0}

    # --- critical-pair tactical accuracy ---
    crit_cases = [c for c in gt_cases if c["pair"] in CRITICAL_PAIRS]
    crit_hits, crit_scored = 0, 0
    for c in crit_cases:
        parsed = parsed_by_case[c["name"]]
        if parsed is None or is_abstention(parsed.get("likely_intent", "")):
            continue
        crit_scored += 1
        if match_threat(parsed.get("threat_level", ""), "critical"):
            crit_hits += 1
    critical_pair_tactical_accuracy = {
        "n": crit_scored, "accuracy": (crit_hits / crit_scored if crit_scored else None),
        "low_n_caveat": crit_scored < LOW_N_THRESHOLD,
    }

    return {
        "per_class_threat_accuracy": per_class,
        "answerability": answerability,
        "escalation": escalation,
        "schema_validity_rate": schema_validity_rate,
        "critical_pair_tactical_accuracy": critical_pair_tactical_accuracy,
        # ADVISORY ONLY -- never read by any of the fields above. judge_scores is {}
        # when judge_client=None; present-but-populated when a real judge is passed.
        # This separation IS the "advisory-only, never load-bearing" contract --
        # see tests/test_eval_sft_v5.py::test_judge_advisory_only for the check.
        # Per-case raw/parsed output -- NOT read by any aggregate metric above,
        # exposed so callers (e.g. score_memorization.py) can do their own
        # per-case analysis without a second predict_fn pass.
        "raw_by_case": raw_by_case,
        "parsed_by_case": parsed_by_case,
        "judge_scores": judge_scores_by_case,
        "judge_overall_mean": (sum(judge_scores_by_case.values()) / len(judge_scores_by_case)
                               if judge_scores_by_case else None),
    }


def print_report(results: dict):
    print("=" * 90)
    print("per-class threat accuracy (Wilson 95% CI)")
    print("=" * 90)
    print("| threat | n | accuracy | 95% CI | caveat |")
    print("|---|---|---|---|---|")
    for level in THREAT_ORDER:
        r = results["per_class_threat_accuracy"][level]
        if r["accuracy"] is None:
            print(f"| {level} | 0 | n/a | n/a | no scored cases |")
        else:
            caveat = f"LOW N (<{LOW_N_THRESHOLD}), not statistically meaningful" if r["low_n_caveat"] else ""
            print(f"| {level} | {r['n']} | {r['accuracy']:.1%} | "
                 f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] | {caveat} |")

    print()
    print("=" * 90)
    print("answerability (three separate columns, never blended)")
    print("=" * 90)
    a = results["answerability"]
    aw = f"{a['accuracy_when_answerable']:.1%}" if a["accuracy_when_answerable"] is not None else "n/a"
    ab = (f"{a['abstention_rate_when_unanswerable']:.1%}"
         if a["abstention_rate_when_unanswerable"] is not None else "n/a")
    oa = f"{a['over_abstention_rate']:.1%}" if a["over_abstention_rate"] is not None else "n/a"
    print(f"accuracy_when_answerable: {aw} (n={a['n_answerable']})")
    print(f"abstention_rate_when_unanswerable: {ab} (n={a['n_unanswerable']})")
    print(f"over_abstention_rate: {oa}")

    print()
    print("=" * 90)
    print("escalation direction (separate, never pooled into one 'escalation error')")
    print("=" * 90)
    e = results["escalation"]
    print(f"n={e['n']}  correct={e['correct']:.1%}  "
         f"UNDER-escalated={e['under_escalated']:.1%}  OVER-escalated={e['over_escalated']:.1%}  "
         f"abstained={e['abstained']:.1%}")

    print()
    s = results["schema_validity_rate"]
    print(f"schema_validity_rate: {s['rate']:.1%} ({s['n_valid']}/{s['n_total']})")

    print()
    c = results["critical_pair_tactical_accuracy"]
    if c["accuracy"] is not None:
        caveat = " (LOW N, not statistically meaningful)" if c["low_n_caveat"] else ""
        print(f"critical_pair_tactical_accuracy: {c['accuracy']:.1%} (n={c['n']}){caveat}")
    else:
        print(f"critical_pair_tactical_accuracy: n/a (n={c['n']})")

    print()
    jm = results.get("judge_overall_mean")
    print(f"judge_overall_mean: {jm:.2f}/5 (n={len(results.get('judge_scores', {}))}) [ADVISORY ONLY]"
         if jm is not None else "judge_overall_mean: n/a (no judge configured) [ADVISORY ONLY]")


def resolve_adapter_path(path: str) -> str:
    """Given either (a) a directory that IS a saved adapter (has its own
    adapter_config.json -- e.g. after trainer.save_model(args.out) at the very
    end of training) or (b) a training output_dir containing checkpoint-<N>
    subdirectories (trl's automatic save_steps checkpointing, prefix
    "checkpoint" -- confirmed via transformers.trainer_utils.PREFIX_CHECKPOINT_DIR),
    returns the path to use for loading: itself in case (a), or the
    HIGHEST-step checkpoint-<N> in case (b) (numeric sort on N, NOT
    lexicographic -- "checkpoint-1500" must sort after "checkpoint-500", which
    a plain string sort would get backwards).

    Raises FileNotFoundError with a specific, actionable message if neither
    shape is found (e.g. training hasn't reached save_steps yet)."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"adapter path does not exist: {path}")

    checkpoints = []
    for name in os.listdir(path):
        m = CHECKPOINT_DIR_RE.match(name)
        candidate = os.path.join(path, name)
        if m and os.path.exists(os.path.join(candidate, "adapter_config.json")):
            checkpoints.append((int(m.group(1)), candidate))
    if not checkpoints:
        raise FileNotFoundError(
            f"{path} has neither its own adapter_config.json nor any checkpoint-<N> "
            f"subdirectory containing one -- has training saved a checkpoint yet? "
            f"(save_steps=500 in train_sft_v5.py; check evaluation/progress/v5a_training.json "
            f"for current step count)")
    checkpoints.sort(key=lambda t: t[0])  # numeric, not lexicographic
    return checkpoints[-1][1]


def load_model_dry_run(base_model: str, adapter_path: str):
    """Real tokenizer + meta-device base model + PEFT wrapping, using the SAME
    LoRA hyperparameters train_sft_v5.py trains with (imported from there, not
    re-typed, so the two can't silently drift out of sync). Asserts zero real
    GPU memory used, matching train_sft_v5.py's own --dry-run convention.

    Does NOT load the adapter's real trained weights -- no real v5-a
    checkpoint exists yet this session. Once one does, the fix is a single
    line: replace `get_peft_model(base, lora)` below with
    `PeftModel.from_pretrained(base, adapter_path)` to load the real deltas.
    Everything else in this function (tokenizer, meta-device base
    construction, the LoRA config itself) is already real and unchanged by
    that swap."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from train_sft_v5 import LORA_R, LORA_ALPHA, LORA_TARGET_MODULES

    pre_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert pre_alloc == 0, f"GPU memory already allocated before load: {pre_alloc} bytes"

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="meta")
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(base, lora)

    post_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    assert post_alloc == 0, f"loading allocated real GPU memory: {post_alloc} bytes"

    print(f"loaded (meta device, torch.cuda.memory_allocated()==0 before and after): "
         f"base={base_model}, resolved_adapter_path={adapter_path}")
    print("NOTE: real trained adapter weights NOT loaded -- no real v5-a checkpoint exists "
         "yet this session. See load_model_dry_run() docstring for the one-line swap needed "
         "once one does.")
    return tok, model


def stub_generate(case: dict) -> str:
    """Stands in for a real model.generate() call -- there is no real
    v5-a checkpoint to generate from yet this session. Reuses mock_predict's
    per-branch logic (same deliberate correct/wrong/schema-invalid coverage)
    so the real-adapter dry-run path exercises identical metric branches to
    --mock, just via the real checkpoint-resolution/loading code above it."""
    return mock_predict(case)


def load_phase4_items(path: str = "evaluation/phase4_eval_set.json") -> list[dict]:
    with open(path) as f:
        payload = json.load(f)
    return payload["items"]


def build_case_from_phase4_item(item: dict) -> dict:
    """Converts a phase4_eval_set.json item into evaluate()'s case shape."""
    return {
        "name": item["name"],
        "pair": tuple(item["pair"]) if item.get("pair") else None,
        "expected_threat": item.get("expected_threat"),
        "has_ground_truth": item["has_ground_truth"],
    }


def build_prompt_from_phase4_item(item: dict) -> str:
    from swarm_intent.inference import build_llm_prompt
    from swarm_intent import pipeline_v2
    return build_llm_prompt(pipeline_v2._preds_from_key_windows(item["key_windows"]), item["ctx"], {})


def load_real_client(base_model: str, adapter_path: str, temperature: float = 0.0):
    """REAL 4-bit quantized model + REAL adapter weights, on the GPU -- same
    LocalHFClient code path every other eval script in this project uses
    (v2_client/ft_client/rules_client in eval_real_stgt_output.py etc.), so
    the load config (NF4 double-quant, bf16 compute) is identical to what
    those comparison systems were loaded with, and to training's own
    quantization config. temperature=0.0 -> greedy (do_sample=False in
    LocalHFClient.generate), matching PREREGISTRATION.md's "greedy for every
    HEADLINE number" requirement."""
    from swarm_intent.llm.client import LocalHFClient
    return LocalHFClient(base_model, adapter_path=adapter_path, temperature=temperature)


def sanity_check_three_cases(client, items: list[dict]) -> None:
    """Hand-picked: one low-threat steady state, one high/critical
    transition, one multi-hop unanswerable case -- printed BEFORE any batch
    run, so a garbage/malformed model can be caught before spending the full
    eval budget on it."""
    low_case = next((it for it in items if it.get("expected_threat") == "low"), items[0])
    high_case = next((it for it in items if it.get("expected_threat") in ("high", "critical")
                      and it["name"] != low_case["name"]), items[1])
    unans_case = next((it for it in items if not it["has_ground_truth"]
                       and it["name"] not in (low_case["name"], high_case["name"])), items[2])

    print("=" * 90)
    print("SANITY CHECK: 3 hand-picked cases, raw output BEFORE any batch run")
    print("=" * 90)
    for label, item in (("low-threat steady state", low_case),
                        ("high/critical transition", high_case),
                        ("multi-hop unanswerable", unans_case)):
        prompt = build_prompt_from_phase4_item(item)
        raw = client.generate(prompt)
        parsed, valid = parse_and_validate(raw)
        print(f"\n--- {label} ({item['name']}, expected_threat="
             f"{item.get('expected_threat')}, has_ground_truth={item['has_ground_truth']}) ---")
        print(f"schema_valid={valid}")
        print(raw[:800])
    print("\n" + "=" * 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", "--dry-run", dest="mock", action="store_true",
                    help="fully synthetic: no checkpoint resolution, no model loading at "
                         "all, stub predictor only.")
    ap.add_argument("--adapter", default=None,
                    help=f"path to a real LoRA adapter OR a training output_dir containing "
                         f"checkpoint-<N> subdirs (default when given with no value: "
                         f"{DEFAULT_OUTPUT_DIR}). Runs the REAL checkpoint-resolution + "
                         f"meta-device model-loading path (zero GPU memory) -- generation "
                         f"itself is still stubbed until a real checkpoint exists.")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge", action="store_true",
                    help="also score with the advisory-only judge (default_judge_client()). "
                         "Requires NVIDIA_API_KEY.")
    ap.add_argument("--real", action="store_true",
                    help="REAL generation on the GPU (LocalHFClient, 4-bit, greedy) against "
                         "--phase4-set. Requires --adapter to resolve to a real checkpoint.")
    ap.add_argument("--phase4-set", default="evaluation/phase4_eval_set.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only evaluate the first N phase4 items (smoke testing)")
    ap.add_argument("--progress-task", default=None,
                    help="if set, writes evaluation/progress/<task>.json (swarm_intent.progress.Reporter)")
    ap.add_argument("--out", default="logs/eval_v5_results.json")
    args = ap.parse_args()

    if not args.mock and args.adapter is None:
        args.adapter = DEFAULT_OUTPUT_DIR

    judge = default_judge_client() if args.judge else None
    if args.judge and judge is None:
        print("--judge requested but NVIDIA_API_KEY is not set -- running WITHOUT a judge; "
             "headline metrics are unaffected either way (advisory-only).")
    elif judge:
        print(f"judge: {JUDGE_MODEL} via NVIDIA NIM (advisory only)")

    if args.mock:
        cases = mock_cases()
        results = evaluate(cases, mock_predict, judge_client=judge)
        adapter_path_reported = None
        mode = "mock"
    elif not args.real:
        cases = mock_cases()
        resolved = resolve_adapter_path(args.adapter)
        tok, model = load_model_dry_run(args.base, resolved)
        results = evaluate(cases, stub_generate, judge_client=judge)
        adapter_path_reported = resolved
        mode = ("real_adapter_dryrun (checkpoint resolution + meta-device load are REAL; "
               "generation is still stubbed -- no real checkpoint weights loaded yet)")
    else:
        resolved = resolve_adapter_path(args.adapter)
        all_items = load_phase4_items(args.phase4_set)
        items = all_items[:args.limit] if args.limit else all_items
        cases = [build_case_from_phase4_item(it) for it in items]
        prompts_by_name = {it["name"]: build_prompt_from_phase4_item(it) for it in items}

        client = load_real_client(args.base, resolved, temperature=0.0)  # greedy
        # Sanity check searches the FULL population (all_items), never the
        # --limit-truncated `items` -- a small --limit could otherwise make the
        # 3 "hand-picked" cases collapse onto the same fallback item, as happened
        # during this file's own development (--limit 3 caught it).
        sanity_check_three_cases(client, all_items)

        prompts_ordered = [prompts_by_name[c["name"]] for c in cases]
        chunk_size = args.batch_size * 8  # report progress every 8 batches
        print(f"=== batched generation, {len(prompts_ordered)} cases, batch_size={args.batch_size}, "
             f"progress every {chunk_size} ===")
        from swarm_intent.progress import Reporter
        reporter = Reporter(args.progress_task or "eval_sft_v5_real", len(prompts_ordered)) \
            if args.progress_task else None
        raw_outputs = []
        for start in range(0, len(prompts_ordered), chunk_size):
            chunk = prompts_ordered[start:start + chunk_size]
            raw_outputs.extend(client.generate_batch(chunk, batch_size=args.batch_size))
            print(f"  {min(start + chunk_size, len(prompts_ordered))}/{len(prompts_ordered)} generated", flush=True)
            if reporter:
                reporter.update(len(chunk))
        if reporter:
            reporter.status = "done"
            reporter._write()
        raw_by_name = dict(zip((c["name"] for c in cases), raw_outputs))

        results = evaluate(cases, lambda c: raw_by_name[c["name"]], judge_client=judge)
        adapter_path_reported = resolved
        mode = "real (greedy, temperature=0.0, LocalHFClient 4-bit)"

    results["meta"] = {"mode": mode, "adapter_path": adapter_path_reported, "n_cases": len(cases),
                       "judge_model": JUDGE_MODEL if judge else None,
                       "generated_at_utc": datetime.now(timezone.utc).isoformat()}

    print_report(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
