"""
Step 1 of the "before committing to v4 coverage-aware pipeline" session
(AUDIT.md sec Y -- see that section for the letter note).

The decisive cheap test: is the low-threat collapse (secs M-P) a decoding/
calibration problem (the model actually assigns "low" real probability mass,
it's just not the argmax under greedy/sampled decoding because "medium" is
trained-in as the more frequent prior) or a genuine knowledge gap (the model
assigns "low" near-zero probability regardless)?

For each case, greedily generates the model's own real completion (deterministic,
so the logit inspection is reproducible), locates the token position where the
`"threat_level": "` key's value begins by incrementally decoding the generated
tokens, then does SEQUENCE SCORING for all four candidate values (low, medium,
high, critical) via teacher-forced forward passes at that exact prefix -- this
handles multi-token candidates correctly (raw next-token comparison would be
wrong if e.g. "critical" tokenizes differently than "low"). Reports P(class) via
softmax over the four candidates' summed log-probs (a distribution over "which
of these four", not the full vocabulary).

Then tests a simple logit-prior correction: corrected_logP(c) = raw_logP(c) -
log(train_class_freq(c)) (using each model's OWN training file's class
frequency, sec N), re-derives the argmax-predicted class under this corrected
rule, and re-scores accuracy on the full 55-case battery (single deterministic
pass per case, not n_runs=5 -- this is inspecting the distribution directly,
not sampling from it).

Usage:
    python llm_finetuning/logit_inspection.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.inference import build_llm_prompt  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402
from swarm_intent.progress import Reporter  # noqa: E402

from build_sft_dataset import synth_context  # noqa: E402

CANDIDATES = ("low", "medium", "high", "critical")
THREAT_KEY_MARKER = '"threat_level": "'

SYSTEMS = {
    "v3a": ("adapters/qwen-swarm-v3a", "data/sft_train_final.jsonl"),
    "v3b": ("adapters/qwen-swarm-v3b", "data/sft_train_final_abstain.jsonl"),
    "v3d": ("adapters/qwen-swarm-v3d", "data/sft_train_v3d.jsonl"),
}


def train_class_freq(path: str) -> dict:
    from collections import Counter
    counts = Counter()
    with open(REPO / path) as f:
        for line in f:
            row = json.loads(line)
            target = json.loads(row["messages"][1]["content"])
            t = target.get("threat_level")
            if t in CANDIDATES:
                counts[t] += 1
    total = sum(counts.values())
    return {c: counts[c] / total for c in CANDIDATES}


def build_case_prompt(case, rng):
    """rng MUST be a single shared Random instance advanced sequentially across
    ALL 55 TEST_CASES in order -- NOT a fresh Random(0) per case. An earlier
    version of this function did exactly that (fresh Random(0) per call), which
    meant every case got the SAME first synth_context() draw (identical
    mean_conf/stability/velocity/spread, differing only in formation_a/
    formation_b) instead of the diverse, case-position-dependent draws every
    other eval script in this project produces via make_rules_in_prompt_run_case's
    shared advancing rng. That bug made this script's numbers not comparable to
    any other eval result on disk -- caught by a step in AUDIT.md sec CC that
    found batched vs unbatched greedy decoding agreed with each other (13.3% vs
    6.67%) but neither matched this script's original 53.3%, which should have
    been impossible if decode mode were the only variable."""
    ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
    preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
             "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
             "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
             "transition_from": kw["from"], "transition_to": kw["to"]} for kw in key_windows]
    return build_llm_prompt(preds, ctx, {})


def logit_inspect_case(model, tok, case, rng, max_new_tokens=512):
    import torch
    prompt = build_case_prompt(case, rng)
    messages = [{"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.eos_token_id, do_sample=False)
    gen_ids = gen_out[0][prompt_len:]

    # Find the minimal prefix of generated tokens whose decoded text contains the
    # threat_level key marker -- the NEXT token is where the value begins.
    marker_pos = None
    for n in range(1, len(gen_ids) + 1):
        decoded = tok.decode(gen_ids[:n], skip_special_tokens=True)
        if THREAT_KEY_MARKER in decoded:
            marker_pos = n
            break
    if marker_pos is None:
        return None, "threat_level key not found in generation"

    prefix_ids = torch.cat([enc["input_ids"][0], gen_ids[:marker_pos]]).unsqueeze(0)
    predicted_value_raw = tok.decode(gen_ids[marker_pos:marker_pos + 3], skip_special_tokens=True)

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
            step_logits = logits[start + i]
            logprob_sum += torch.log_softmax(step_logits, dim=-1)[tid].item()
        logprobs[cand] = logprob_sum

    return {"logprobs": logprobs, "greedy_completion_starts_with": predicted_value_raw}, None


def softmax_over_candidates(logprobs: dict) -> dict:
    vals = np.array([logprobs[c] for c in CANDIDATES])
    vals = vals - vals.max()
    p = np.exp(vals)
    p = p / p.sum()
    return {c: float(p[i]) for i, c in enumerate(CANDIDATES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default=str(REPO / "evaluation" / "logit_inspection.json"))
    args = ap.parse_args()

    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    low_cases = {c["name"] for c in TEST_CASES if c["expected_threat"] == "low"}
    critical_cases = {c["name"] for c in TEST_CASES if c["expected_threat"] == "critical"}
    assert len(low_cases) == 15 and len(critical_cases) == 2

    all_results = {}
    total_units = len(SYSTEMS) * len(TEST_CASES)
    reporter = Reporter("logit_inspection", total_units, rate_hint=0.3)

    for label, (adapter_path, train_file) in SYSTEMS.items():
        print(f"\n=== {label} ===")
        freq = train_class_freq(train_file)
        print(f"  train class freq ({train_file}): {freq}")

        tok = AutoTokenizer.from_pretrained(args.base)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=quant,
                                                     device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, str(REPO / adapter_path))
        model.eval()

        # ONE pass over all 55 cases, ONE shared rng advanced sequentially in
        # TEST_CASES order -- matching make_rules_in_prompt_run_case's protocol
        # exactly, so every case gets its real, case-position-dependent
        # synth_context() draw instead of a fixed first-draw repeated 55 times.
        rng = random.Random(0)
        case_results, battery_results = {}, {}
        for case in TEST_CASES:
            res, err = logit_inspect_case(model, tok, case, rng)
            reporter.update(1, item=f"{label}:{case['name']}")
            if err:
                battery_results[case["name"]] = {"error": err, "expected_threat": case["expected_threat"]}
                continue
            raw_p = softmax_over_candidates(res["logprobs"])
            corrected_logprobs = {c: res["logprobs"][c] - np.log(max(freq[c], 1e-6)) for c in CANDIDATES}
            corrected_p = softmax_over_candidates(corrected_logprobs)
            entry = {
                "expected_threat": case["expected_threat"],
                "raw_logprobs": res["logprobs"],
                "raw_p": raw_p,
                "raw_argmax": max(raw_p, key=raw_p.get),
                "corrected_p": corrected_p,
                "corrected_argmax": max(corrected_p, key=corrected_p.get),
                "greedy_completion_starts_with": res["greedy_completion_starts_with"],
            }
            battery_results[case["name"]] = entry
            if case["name"] in low_cases or case["name"] in critical_cases:
                case_results[case["name"]] = entry

        all_results[label] = {"train_freq": freq, "low_critical_cases": case_results,
                              "battery": battery_results}

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    reporter.status = "done"
    reporter._write()
    Path(args.out).write_text(json.dumps(all_results, indent=2))

    # --- summary ---
    for label, data in all_results.items():
        print(f"\n\n=== {label}: low-threat case logit table ===")
        print("| case | P(low) | P(medium) | P(high) | P(critical) | raw argmax | corrected argmax |")
        print("|---|---|---|---|---|---|---|")
        for name, r in data["low_critical_cases"].items():
            if r.get("expected_threat") != "low":
                continue
            p = r["raw_p"]
            print(f"| {name} | {p['low']:.1%} | {p['medium']:.1%} | {p['high']:.1%} | "
                  f"{p['critical']:.1%} | {r['raw_argmax']} | {r['corrected_argmax']} |")

        print(f"\n=== {label}: critical-threat case logit table ===")
        print("| case | P(low) | P(medium) | P(high) | P(critical) | raw argmax | corrected argmax |")
        print("|---|---|---|---|---|---|---|")
        for name, r in data["low_critical_cases"].items():
            if r.get("expected_threat") != "critical":
                continue
            p = r["raw_p"]
            print(f"| {name} | {p['low']:.1%} | {p['medium']:.1%} | {p['high']:.1%} | "
                  f"{p['critical']:.1%} | {r['raw_argmax']} | {r['corrected_argmax']} |")

        battery = {k: v for k, v in data["battery"].items() if "raw_argmax" in v}
        for cls in ("low", "critical"):
            cases = [r for r in battery.values() if r.get("expected_threat") == cls]
            raw_acc = np.mean([r["raw_argmax"] == cls for r in cases])
            corrected_acc = np.mean([r["corrected_argmax"] == cls for r in cases])
            print(f"\n{label} {cls}-threat accuracy: raw={raw_acc:.1%} -> corrected={corrected_acc:.1%} "
                  f"(n={len(cases)})")
        overall_raw = np.mean([r["raw_argmax"] == r["expected_threat"] for r in battery.values()])
        overall_corrected = np.mean([r["corrected_argmax"] == r["expected_threat"] for r in battery.values()])
        print(f"{label} OVERALL threat accuracy (55-case battery): "
              f"raw={overall_raw:.1%} -> corrected={overall_corrected:.1%}")


if __name__ == "__main__":
    main()
