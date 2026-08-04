"""
AUDIT.md sec AD steps 3+4: does the near-tie/prior-correction signature from
the low-threat collapse (secs Y/CC/AA) reappear at the high/critical end,
in a system with NO fine-tuning at all (rules_in_prompt)?

Step 3: same margin-at-threat_level technique as the low-threat margin
histogram (logit_inspection.py's CANDIDATES, sequence-scored 4-candidate
softmax, greedy-generated prefix) -- this session applied to the 14 high +
2 critical cases instead of the 15 low ones.

Step 4: applies the SAME log-p(c) prior-correction subtraction used on
v3a/v3b/v3d (logit_inspection.py: corrected_logP(c) = raw_logP(c) -
log(class_freq(c))) to rules_in_prompt's high/critical logits. The
fine-tuned adapters' corrections used each adapter's OWN training file's
class frequency; rules_in_prompt has no training file, so the most
principled analog is RULES' own canonical class frequency (llm_finetuning/
report_class_balance.py: low 26.5%, medium 44.9%, high 24.5%, critical
4.1%) -- the actual target distribution rules_in_prompt is given (verbatim,
in RULES.txt) and expected to reflect, not a distribution it was trained
toward (it wasn't trained at all).

Usage (run inside tmux):
    python llm_finetuning/measure_high_crit_margin_and_prior.py
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

from baselines import load_rules_txt  # noqa: E402
from logit_inspection import CANDIDATES, softmax_over_candidates  # noqa: E402
from measure_base_rules_prior import (_find_marker_positions, entropy_at_prefix,  # noqa: E402
                                      score_candidates, THREAT_KEY_MARKER, INTENT_KEY_MARKER)

# RULES' own canonical class frequency (report_class_balance.py) -- the
# target distribution rules_in_prompt is handed verbatim, not a training-file
# distribution (it has none).
RULES_CLASS_FREQ = {"low": 13 / 49, "medium": 22 / 49, "high": 12 / 49, "critical": 2 / 49}

HIGH_CASES = [c for c in TEST_CASES if c["expected_threat"] == "high"]
CRIT_CASES = [c for c in TEST_CASES if c["expected_threat"] == "critical"]
assert len(HIGH_CASES) == 14 and len(CRIT_CASES) == 2


def measure_case(model, tok, case, system_prompt, max_new_tokens=512):
    import torch
    from logit_inspection import build_case_prompt
    import random
    # Deterministic per-case prompt: same ctx a fresh Random(0) would draw at
    # this case's position if walking TEST_CASES in order -- replicate that
    # by advancing a fresh rng identically each call (cheap, no state to share
    # across calls since this script only visits each case once).
    rng = random.Random(0)
    for c in TEST_CASES:
        if c["name"] == case["name"]:
            prompt = build_case_prompt(c, rng)
            break
        build_case_prompt(c, rng)

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
    if THREAT_KEY_MARKER not in positions:
        return None

    threat_pos = positions[THREAT_KEY_MARKER]
    threat_prefix = torch.cat([enc["input_ids"][0], gen_ids[:threat_pos]]).unsqueeze(0)
    greedy_threat = tok.decode(gen_ids[threat_pos:threat_pos + 3], skip_special_tokens=True)

    threat_logprobs = score_candidates(model, tok, threat_prefix)
    raw_p = softmax_over_candidates(threat_logprobs)
    sorted_p = sorted(raw_p.values(), reverse=True)
    margin = sorted_p[0] - sorted_p[1]

    corrected_logprobs = {c: threat_logprobs[c] - np.log(max(RULES_CLASS_FREQ[c], 1e-6))
                          for c in CANDIDATES}
    corrected_p = softmax_over_candidates(corrected_logprobs)

    return {
        "raw_p": raw_p,
        "margin": margin,
        "raw_argmax": max(raw_p, key=raw_p.get),
        "corrected_p": corrected_p,
        "corrected_argmax": max(corrected_p, key=corrected_p.get),
        "greedy_completion_starts_with": greedy_threat,
    }


def main():
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
    cases = HIGH_CASES + CRIT_CASES
    reporter = Reporter("high_crit_margin_prior", len(cases), rate_hint=0.3)

    results = {}
    for case in cases:
        res = measure_case(model, tok, case, rules_txt)
        reporter.update(1, item=case["name"])
        if res is not None:
            res["expected_threat"] = case["expected_threat"]
            results[case["name"]] = res

    reporter.status = "done"
    reporter._write()

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    out_path = REPO / "evaluation" / "high_crit_margin_prior.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")

    print("\n=== step 3: margin P(top)-P(second) at threat_level, high + critical ===")
    for stratum, names in (("high", [c["name"] for c in HIGH_CASES]),
                          ("critical", [c["name"] for c in CRIT_CASES])):
        margins = sorted(results[n]["margin"] for n in names if n in results)
        print(f"{stratum} (n={len(margins)}): " + ", ".join(f"{m:.3f}" for m in margins))
        hist, edges = np.histogram(np.array(margins), bins=np.arange(0, 1.1, 0.1))
        print(f"  histogram (0.0-1.0, width 0.1): " + " ".join(str(h) for h in hist))

    print("\n=== step 4: prior correction (RULES' own class freq) ===")
    for stratum, names, expected in (("high", [c["name"] for c in HIGH_CASES], "high"),
                                     ("critical", [c["name"] for c in CRIT_CASES], "critical")):
        cases_here = [results[n] for n in names if n in results]
        raw_acc = np.mean([r["raw_argmax"] == expected for r in cases_here])
        corrected_acc = np.mean([r["corrected_argmax"] == expected for r in cases_here])
        print(f"{stratum} accuracy: raw={raw_acc:.1%} -> corrected={corrected_acc:.1%} (n={len(cases_here)})")
        for n in names:
            if n in results:
                r = results[n]
                print(f"  {n:28s} raw={r['raw_argmax']:9s} corrected={r['corrected_argmax']:9s} "
                     f"P={ {k: round(v,3) for k,v in r['raw_p'].items()} }")


if __name__ == "__main__":
    main()
