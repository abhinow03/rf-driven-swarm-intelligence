# LLM Fine-Tuning — Swarm Tactical Reasoning

This is your part of the project: turning the prompt-engineered Groq baseline
into a **domain-adapted, fine-tuned model** that reads the ML pipeline's
structured output and produces consistent tactical assessments.

## Recommended model

**Qwen2.5-7B-Instruct** (Apache-2.0). It has the best JSON-adherence and
reasoning among 7B models, QLoRA-fits a free Colab T4, and has a permissive
license for a capstone you may publish. Alternatives:

| Model | When to pick it |
|---|---|
| **Qwen2.5-7B-Instruct** | Default. Best quality/JSON on a T4. |
| Mistral-7B-Instruct-v0.3 | If you want to match the original README plan. |
| Qwen2.5-3B / Llama-3.2-3B | If you hit OOM or need lower edge latency. |

## Pipeline

```
build_sft_dataset.py   →  data/sft_train.jsonl (+ _val)
        │  teacher-distilled prose + rule-clean decision labels
        ▼
train_qlora.py         →  adapters/qwen-swarm/  (LoRA adapter, ~hundreds of MB)
        │  4-bit NF4 + LoRA r=16, fits T4
        ▼
evaluate_finetuned.py  →  evaluation/finetuned_eval.json
           independent judge (Groq) + objective intent/threat accuracy
```

## Quickstart (Colab, free T4)

```bash
pip install -U transformers peft bitsandbytes datasets accelerate trl
export GROQ_API_KEY=...                      # for the teacher + judge

# 1. Build data (start ~600 examples; scale up later)
python llm_finetuning/build_sft_dataset.py --n 600 --out data/sft_train.jsonl

# 2. Fine-tune
python llm_finetuning/train_qlora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train data/sft_train.jsonl --val data/sft_train_val.jsonl \
    --out adapters/qwen-swarm

# 3. Evaluate (independent judge — NOT the model judging itself)
python llm_finetuning/evaluate_finetuned.py \
    --base Qwen/Qwen2.5-7B-Instruct --adapter adapters/qwen-swarm
```

## What makes a good fine-tune here (read before you train)

1. **Data quality > data quantity.** 500–1000 *diverse, correctly-labelled*
   examples beat 5k noisy ones. The decision fields (threat/intent/action) are
   rule-cleaned in `build_sft_dataset.py` precisely so the model learns a
   *consistent* policy, not the teacher's noise. Curate `RULES` with your team.

2. **Cover the hard cases.** Add ambiguous and adversarial scenarios
   (slow encirclement, feints, sensor dropout, conflicting cues) — that's where
   a fine-tune earns its keep over prompting.

3. **Evaluate honestly.** The headline number is objective intent/threat
   accuracy against your domain rules, measured over ≥5 runs per case, with an
   *independent* judge. Never report the model judging itself.

4. **Compare against the baseline.** Run the same `evaluate_finetuned.py` flow
   on the Groq baseline (or the un-adapted base model with `--adapter` omitted)
   so you can show the fine-tune actually improved something.

5. **Latency / deployment.** Once happy: `merge_and_unload()` the adapter, then
   export to GGUF (llama.cpp) or ONNX for the edge target in the roadmap.
