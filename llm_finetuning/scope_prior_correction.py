"""
AUDIT.md sec AE step 1 (SAFETY FIX FIRST): the global log-p(c) correction
(sec AD step 4) took rules_in_prompt's high-threat accuracy from 35.7% to
14.3% by letting the log-boost for the rare `critical` class (RULES frequency
4.1%) drag correctly-classified `high` predictions into `critical`. This
script re-measures ALL 55 TEST_CASES with a single deterministic greedy pass
each (same protocol as measure_high_crit_margin_and_prior.py / sec AD, not
n_runs-sampled -- this is inspecting the raw distribution directly), then
applies src/swarm_intent/llm/prior_correction.py's SCOPED correction (only
fires on a medium argmax / low runner-up, restricted to {low, medium} even
then -- see tests/test_prior_correction.py for the guard) instead of the old
global one, and reports low/medium/high/critical accuracy raw vs scoped-
corrected side by side.

Usage (run inside tmux):
    python llm_finetuning/scope_prior_correction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.llm.prior_correction import (  # noqa: E402
    CANDIDATES, THREAT_KEY_MARKER, find_key_marker_position, score_candidates,
    softmax_over_candidates, scoped_correct,
)
from swarm_intent.progress import Reporter  # noqa: E402

from baselines import load_rules_txt  # noqa: E402
from logit_inspection import build_case_prompt  # noqa: E402

# RULES' own canonical class frequency (report_class_balance.py) -- same
# reference distribution sec AD step 4 used for the (unscoped) correction.
RULES_CLASS_FREQ = {"low": 13 / 49, "medium": 22 / 49, "high": 12 / 49, "critical": 2 / 49}

THREAT_ORDER = ("low", "medium", "high", "critical")


def measure_case(model, tok, case, rng, system_prompt, max_new_tokens=512):
    import torch
    prompt = build_case_prompt(case, rng)
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, do_sample=False)
    gen_ids = gen_out[0][prompt_len:]

    pos = find_key_marker_position(tok, gen_ids, THREAT_KEY_MARKER)
    if pos is None:
        return None

    prefix_ids = torch.cat([enc["input_ids"][0], gen_ids[:pos]]).unsqueeze(0)
    greedy_threat = tok.decode(gen_ids[pos:pos + 3], skip_special_tokens=True)

    logprobs = score_candidates(model, tok, prefix_ids)
    raw_p = softmax_over_candidates(logprobs)
    raw_argmax = max(raw_p, key=raw_p.get)
    correction = scoped_correct(raw_p, RULES_CLASS_FREQ)

    return {
        "raw_p": raw_p, "raw_argmax": raw_argmax,
        "corrected_argmax": correction["corrected_argmax"],
        "correction_applied": correction["applied"], "correction_reason": correction["reason"],
        "greedy_completion_starts_with": greedy_threat,
        "expected_threat": case["expected_threat"],
    }


def main():
    import random
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quant,
                                                 device_map="auto", torch_dtype=torch.float16)
    model.eval()

    rules_txt = load_rules_txt()
    reporter = Reporter("scope_prior_correction", len(TEST_CASES), rate_hint=0.3)

    rng = random.Random(0)
    results = {}
    for case in TEST_CASES:
        res = measure_case(model, tok, case, rng, rules_txt)
        reporter.update(1, item=case["name"])
        if res is not None:
            results[case["name"]] = res

    reporter.status = "done"
    reporter._write()

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    out_path = REPO / "evaluation" / "scope_prior_correction.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path} ({len(results)}/{len(TEST_CASES)} cases scored)")

    print("\n=== step 1: raw vs SCOPED-corrected accuracy, by expected threat stratum ===")
    print("| stratum | n | raw acc | scoped-corrected acc | n corrections applied |")
    print("|---|---|---|---|---|")
    name_to_case = {c["name"]: c for c in TEST_CASES}
    for stratum in THREAT_ORDER:
        names = [n for n in results if name_to_case[n]["expected_threat"] == stratum]
        if not names:
            continue
        raw_acc = np.mean([results[n]["raw_argmax"] == stratum for n in names])
        corr_acc = np.mean([results[n]["corrected_argmax"] == stratum for n in names])
        n_applied = sum(results[n]["correction_applied"] for n in names)
        print(f"| {stratum} | {len(names)} | {raw_acc:.1%} | {corr_acc:.1%} | {n_applied} |")

    overall_raw = np.mean([r["raw_argmax"] == r["expected_threat"] for r in results.values()])
    overall_corr = np.mean([r["corrected_argmax"] == r["expected_threat"] for r in results.values()])
    n_applied_total = sum(r["correction_applied"] for r in results.values())
    print(f"| overall | {len(results)} | {overall_raw:.1%} | {overall_corr:.1%} | {n_applied_total} |")

    print("\n=== cases where the scoped correction actually fired ===")
    for n, r in results.items():
        if r["correction_applied"]:
            print(f"  {n:28s} expected={r['expected_threat']:9s} raw={r['raw_argmax']:9s} "
                 f"-> corrected={r['corrected_argmax']:9s} P={ {k: round(v,3) for k,v in r['raw_p'].items()} }")

    print("\n=== sanity: no case with expected_threat in (high, critical) was ever corrected ===")
    bad = [n for n, r in results.items()
          if r["expected_threat"] in ("high", "critical") and r["correction_applied"]]
    print(f"  violations: {len(bad)} ({bad})")
    assert not bad, "scoped correction fired on a high/critical-expected case -- SCOPING BUG"


if __name__ == "__main__":
    main()
