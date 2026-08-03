"""
Step 3c of the "settle delta_v, recalibrate synth_context, diagnose the prior
skew" session (AUDIT.md sec X).

Parts 3a/3b (see AUDIT.md sec X) established that the 234-row base training
set's threat_level target distribution is NOT meaningfully skewed relative to
RULES (24.8% vs 26.5% low -- within sampling noise from uniform pair
sampling), yet v3a's OBSERVED low-threat accuracy on the 55-case battery is
0% (sec CC). If the training targets aren't skewed, the collapse has to be
either a decoding artefact (already ruled out, sec CC) or something about how
the model's own output distribution behaves at inference time.

This script tests that directly: for each of the 15 low-threat cases, greedily
generate the model's real completion, locate the token position where
`"threat_level": "`'s value begins AND where `"likely_intent": "`'s value
begins (same incremental-decode marker-finding technique as
logit_inspection.py, reusing its `build_case_prompt` -- same shared-rng
protocol, so results are directly comparable to sec CC/Y's numbers), and
compute the FULL-VOCABULARY Shannon entropy (bits) of the next-token
distribution at each position via a teacher-forced forward pass.

If P(medium) is specifically elevated at the threat_level position (a
narrow, field-specific effect), entropy there should be LOW (the model is
confidently wrong, concentrated on "medium") while entropy at likely_intent
(which sec S found does NOT collapse) should be comparable to or higher than
threat_level's. If the whole output distribution is flattened generically
whenever the model is on shaky ground, both positions should show similarly
elevated entropy.

Usage:
    python llm_finetuning/measure_threat_intent_entropy.py
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
from swarm_intent.progress import Reporter  # noqa: E402

from logit_inspection import build_case_prompt, SYSTEMS  # noqa: E402

THREAT_KEY_MARKER = '"threat_level": "'
INTENT_KEY_MARKER = '"likely_intent": "'


def _find_marker_positions(tok, gen_ids, markers, max_scan=None):
    """Incrementally decodes gen_ids and records, for each marker string, the
    token-count prefix length at which the marker first appears in the decoded
    text -- same technique logit_inspection.py uses for THREAT_KEY_MARKER,
    generalised to multiple markers found in one pass."""
    found = {}
    limit = max_scan or len(gen_ids)
    for n in range(1, limit + 1):
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
    logits = out.logits[0, -1]  # next-token distribution right after the prefix
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    # Shannon entropy in bits; mask true-zero-prob entries (log(0)*0 -> nan) out.
    ent_nats = -(p * log_p).nan_to_num(0.0).sum().item()
    return ent_nats / np.log(2)


def measure_case(model, tok, case, rng, max_new_tokens=512):
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

    positions = _find_marker_positions(tok, gen_ids, (THREAT_KEY_MARKER, INTENT_KEY_MARKER))
    if THREAT_KEY_MARKER not in positions or INTENT_KEY_MARKER not in positions:
        return None

    result = {}
    for name, marker in (("threat_level", THREAT_KEY_MARKER), ("likely_intent", INTENT_KEY_MARKER)):
        marker_pos = positions[marker]
        prefix_ids = torch.cat([enc["input_ids"][0], gen_ids[:marker_pos]]).unsqueeze(0)
        value_text = tok.decode(gen_ids[marker_pos:marker_pos + 4], skip_special_tokens=True)
        result[name] = {
            "entropy_bits": entropy_at_prefix(model, prefix_ids),
            "observed_value_starts_with": value_text,
        }
    return result


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import random
    import gc

    low_cases = [c for c in TEST_CASES if c["expected_threat"] == "low"]
    assert len(low_cases) == 15

    all_results = {}
    reporter = Reporter("threat_intent_entropy", len(SYSTEMS) * len(low_cases), rate_hint=0.3)

    for label, (adapter_path, _train_file) in SYSTEMS.items():
        print(f"\n=== {label} ===")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quant,
                                                     device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, str(REPO / adapter_path))
        model.eval()

        # Shared rng advanced across ALL 55 TEST_CASES in order (matching
        # logit_inspection.py's fixed protocol, sec CC), so entropy numbers here
        # are directly comparable to that script's per-case results -- we only
        # keep the 15 low-threat cases, but must walk the full battery in order
        # to reproduce the same context draws for those 15 positions.
        rng = random.Random(0)
        case_results = {}
        for case in TEST_CASES:
            if case["expected_threat"] != "low":
                build_case_prompt(case, rng)  # advance rng, discard (not a low case)
                continue
            res = measure_case(model, tok, case, rng)
            reporter.update(1, item=f"{label}:{case['name']}")
            if res is not None:
                case_results[case["name"]] = res

        all_results[label] = case_results
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    reporter.status = "done"
    reporter._write()

    out_path = REPO / "evaluation" / "threat_intent_entropy.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved {out_path}")

    print("\n=== summary: mean entropy (bits) at threat_level vs likely_intent, low-threat cases ===")
    print("| system | n | mean H(threat_level) | mean H(likely_intent) | delta |")
    print("|---|---|---|---|---|")
    for label, case_results in all_results.items():
        h_threat = [r["threat_level"]["entropy_bits"] for r in case_results.values()]
        h_intent = [r["likely_intent"]["entropy_bits"] for r in case_results.values()]
        mt, mi = float(np.mean(h_threat)), float(np.mean(h_intent))
        print(f"| {label} | {len(case_results)} | {mt:.3f} | {mi:.3f} | {mt - mi:+.3f} |")

    print("\n=== per-case detail ===")
    for label, case_results in all_results.items():
        print(f"\n{label}:")
        for name, r in case_results.items():
            print(f"  {name:28s} H(threat)={r['threat_level']['entropy_bits']:.3f} "
                 f"(observed={r['threat_level']['observed_value_starts_with']!r})  "
                 f"H(intent)={r['likely_intent']['entropy_bits']:.3f} "
                 f"(observed={r['likely_intent']['observed_value_starts_with']!r})")


if __name__ == "__main__":
    main()
