"""
Evaluate the fine-tuned LLM against the prompt-engineered baseline, using an
INDEPENDENT judge (this is the fix for the old self-judging eval).

The system under test = your local fine-tuned model (LocalHFClient).
The judge = a DIFFERENT hosted model (GroqClient, llama-3.3-70b) so no model
grades its own output.

Reports objective intent/threat accuracy as the HEADLINE metric.

Usage:
    export GROQ_API_KEY=...
    python llm_finetuning/evaluate_finetuned.py \
        --base Qwen/Qwen2.5-7B-Instruct --adapter adapters/qwen-swarm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swarm_intent.llm.client import GroqClient, LocalHFClient
from swarm_intent.llm.evaluate import evaluate_llm
from swarm_intent.llm.prompts import TEST_CASES
from swarm_intent.inference import build_llm_prompt

# Reuse the same scenario synthesiser used for training data.
from build_sft_dataset import synth_context  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit to test the base model")
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="evaluation/finetuned_eval.json")
    args = ap.parse_args()

    system = LocalHFClient(args.base, adapter_path=args.adapter, temperature=0.3)
    judge = None if args.no_judge else GroqClient(model="llama-3.3-70b-versatile")
    rng = random.Random(0)

    def run_case(case):
        ctx, key_windows = synth_context(case["formation_a"], case["formation_b"], rng)
        preds = [{**kw, "time_start_s": 0, "time_end_s": 0, "formation_type": kw["formation"],
                  "centroid_velocity": kw["velocity"], "approach_rate": kw["approach"],
                  "formation_stability": kw["stability"], "formation_confidence": kw["confidence"],
                  "role_differentiation": False, "transition_from": kw["from"],
                  "transition_to": kw["to"]} for kw in key_windows]
        prompt = build_llm_prompt(preds, ctx, {})
        assessment = system.complete(prompt)
        return assessment, ctx

    results = evaluate_llm(run_case, TEST_CASES, judge_client=judge, n_runs=args.n_runs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    agg = results["aggregate"]
    print("\n=== HEADLINE (objective) ===")
    print(f"Intent accuracy:  {agg['mean_intent_accuracy']:.2%}")
    print(f"Threat accuracy:  {agg['mean_threat_accuracy']:.2%}")
    print(f"Hallucination:    {agg['mean_hallucination_rate']:.2%}")
    print(f"Consistency:      {agg['mean_consistency']:.2f}")
    print(f"(judge scores are advisory only; saved to {args.out})")


if __name__ == "__main__":
    main()
