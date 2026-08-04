"""
Steps 1+2 of the "settle the entropy confound, then find the prior's origin"
session (AUDIT.md sec AA).

Sec Z's full-vocabulary entropy comparison between threat_level (4 legal
values: prompts.py THREAT_FAMILIES) and likely_intent (15 legal values:
prompts.py INTENT_FAMILIES) is confounded by the size of each field's legal
candidate set -- full-vocab entropy is naturally lower for a field with a
smaller effective candidate pool, independent of how confident the model
actually is. Normalizing by log2(n_legal_values) -- 2.0 bits for
threat_level, log2(15)=3.907 bits for likely_intent -- flips sec Z's
comparison for v3a: 0.840/2.0=0.420 vs 1.149/3.907=0.294, i.e. threat_level
reads as LESS confident once normalized, not more. This script settles it
properly with two additional measures sec Z didn't have: the margin
P(top)-P(second) of the restricted 4-way threat_level distribution (not
biased by candidate-set size at all, since it's already restricted), and
data for two systems sec Z never ran -- `base` (Qwen2.5-7B-Instruct, no
adapter, no system prompt) and `rules_in_prompt` (same base weights +
RULES.txt as system prompt, baselines.py's make_rules_in_prompt_run_case
protocol) -- loaded ONCE since they share the same underlying weights, only
the message list differs.

This also directly answers step 2's question: if `base` -- which has never
seen a single training row from this pipeline -- ALSO predicts `medium` on
these 15 low-threat cases, the medium prior is pretraining-inherited, not
induced by this project's fine-tuning data/pipeline. If it predicts
something else, fine-tuning induced the skew and the cause is somewhere in
the training pipeline after all.

Reuses logit_inspection.py's CANDIDATES/softmax_over_candidates/
build_case_prompt (same shared-rng protocol, so case draws are positionally
identical to sec Y/CC/Z's data) and measure_threat_intent_entropy.py's
full-vocab entropy technique.

Usage:
    python llm_finetuning/measure_base_rules_prior.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.prompts import TEST_CASES, INTENT_FAMILIES, THREAT_FAMILIES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from logit_inspection import build_case_prompt, CANDIDATES, softmax_over_candidates  # noqa: E402
from baselines import RULES_TXT_PATH  # noqa: E402

THREAT_KEY_MARKER = '"threat_level": "'
INTENT_KEY_MARKER = '"likely_intent": "'

N_THREAT_CLASSES = len(THREAT_FAMILIES)   # 4: low/medium/high/critical
N_INTENT_CLASSES = len(INTENT_FAMILIES)   # 15
LOG2_THREAT = float(np.log2(N_THREAT_CLASSES))
LOG2_INTENT = float(np.log2(N_INTENT_CLASSES))


def _find_marker_positions(tok, gen_ids, markers):
    found = {}
    for n in range(1, len(gen_ids) + 1):
        decoded = tok.decode(gen_ids[:n], skip_special_tokens=True)
        for m in markers:
            if m not in found and m in decoded:
                found[m] = n
        if len(found) == len(markers):
            break
    return found


def entropy_at_prefix(model, prefix_ids):
    import torch
    with torch.no_grad():
        out = model(prefix_ids)
    logits = out.logits[0, -1]
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    ent_nats = -(p * log_p).nan_to_num(0.0).sum().item()
    return ent_nats / np.log(2)


def score_candidates(model, tok, prefix_ids):
    import torch
    logprobs = {}
    for cand in CANDIDATES:
        cand_ids = tok(cand, add_special_tokens=False)["input_ids"]
        full_ids = torch.cat([prefix_ids, torch.tensor([cand_ids], device=prefix_ids.device)], dim=1)
        with torch.no_grad():
            out = model(full_ids)
        logits = out.logits[0]
        start = prefix_ids.shape[1] - 1
        logprob_sum = 0.0
        for i, tid in enumerate(cand_ids):
            logprob_sum += torch.log_softmax(logits[start + i], dim=-1)[tid].item()
        logprobs[cand] = logprob_sum
    return logprobs


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

    positions = _find_marker_positions(tok, gen_ids, (THREAT_KEY_MARKER, INTENT_KEY_MARKER))
    if THREAT_KEY_MARKER not in positions or INTENT_KEY_MARKER not in positions:
        return None

    threat_pos, intent_pos = positions[THREAT_KEY_MARKER], positions[INTENT_KEY_MARKER]
    threat_prefix = torch.cat([enc["input_ids"][0], gen_ids[:threat_pos]]).unsqueeze(0)
    intent_prefix = torch.cat([enc["input_ids"][0], gen_ids[:intent_pos]]).unsqueeze(0)

    greedy_threat = tok.decode(gen_ids[threat_pos:threat_pos + 3], skip_special_tokens=True)
    greedy_intent = tok.decode(gen_ids[intent_pos:intent_pos + 6], skip_special_tokens=True)

    threat_h = entropy_at_prefix(model, threat_prefix)
    intent_h = entropy_at_prefix(model, intent_prefix)
    threat_logprobs = score_candidates(model, tok, threat_prefix)
    raw_p = softmax_over_candidates(threat_logprobs)
    sorted_p = sorted(raw_p.values(), reverse=True)
    margin = sorted_p[0] - sorted_p[1]

    # Canonical label for the free-form greedy decode, by prefix match against
    # the 4 legal candidates -- this is the model's REAL unconstrained output,
    # distinct from raw_p's argmax (restricted to the 4 candidates by
    # construction; they should agree in the overwhelming majority of cases,
    # a disagreement would itself be worth flagging).
    greedy_label = next((c for c in CANDIDATES if greedy_threat.strip().lower().startswith(c)), None)

    return {
        "threat_level": {
            "entropy_bits": threat_h,
            "normalized_entropy": threat_h / LOG2_THREAT,
            "raw_p": raw_p,
            "margin": margin,
            "restricted_argmax": max(raw_p, key=raw_p.get),
            "greedy_completion_starts_with": greedy_threat,
            "greedy_label": greedy_label,
        },
        "likely_intent": {
            "entropy_bits": intent_h,
            "normalized_entropy": intent_h / LOG2_INTENT,
            "greedy_completion_starts_with": greedy_intent,
        },
    }


def main():
    import gc
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    low_cases = [c for c in TEST_CASES if c["expected_threat"] == "low"]
    assert len(low_cases) == 15

    rules_txt = Path(RULES_TXT_PATH).read_text()
    SYSTEM_PROMPTS = {"base": None, "rules_in_prompt": rules_txt}

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quant,
                                                 device_map="auto", torch_dtype=torch.float16)
    model.eval()

    all_results = {}
    reporter = Reporter("base_rules_prior", len(SYSTEM_PROMPTS) * len(low_cases), rate_hint=0.3)

    for label, sys_prompt in SYSTEM_PROMPTS.items():
        print(f"\n=== {label} ===")
        rng = random.Random(0)
        case_results = {}
        for case in TEST_CASES:
            if case["expected_threat"] != "low":
                build_case_prompt(case, rng)  # advance rng, discard (not a low case)
                continue
            res = measure_case(model, tok, case, rng, sys_prompt)
            reporter.update(1, item=f"{label}:{case['name']}")
            if res is not None:
                case_results[case["name"]] = res
        all_results[label] = case_results

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    reporter.status = "done"
    reporter._write()

    out_path = REPO / "evaluation" / "base_rules_prior.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved {out_path}")

    print("\n=== step 1: normalized entropy (bits / log2(n_classes)) ===")
    print("| system | n | norm H(threat_level) | norm H(likely_intent) | delta |")
    print("|---|---|---|---|---|")
    for label, case_results in all_results.items():
        ht = [r["threat_level"]["normalized_entropy"] for r in case_results.values()]
        hi = [r["likely_intent"]["normalized_entropy"] for r in case_results.values()]
        mt, mi = float(np.mean(ht)), float(np.mean(hi))
        print(f"| {label} | {len(case_results)} | {mt:.3f} | {mi:.3f} | {mt - mi:+.3f} |")

    print("\n=== step 1: margin P(top)-P(second) at threat_level, full distribution ===")
    for label, case_results in all_results.items():
        margins = sorted(r["threat_level"]["margin"] for r in case_results.values())
        print(f"{label}: " + ", ".join(f"{m:.3f}" for m in margins))

    print("\n=== step 2: threat_level greedy prediction distribution ===")
    print("| system | " + " | ".join(CANDIDATES) + " | unparsed |")
    print("|---|" + "---|" * (len(CANDIDATES) + 1))
    for label, case_results in all_results.items():
        labels = [r["threat_level"]["greedy_label"] for r in case_results.values()]
        counts = Counter(labels)
        n = len(labels)
        row = " | ".join(f"{counts.get(c, 0)}/{n} ({counts.get(c, 0)/n:.1%})" for c in CANDIDATES)
        print(f"| {label} | {row} | {counts.get(None, 0)} |")


if __name__ == "__main__":
    main()
