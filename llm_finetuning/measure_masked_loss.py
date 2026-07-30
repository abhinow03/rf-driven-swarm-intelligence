"""
Measure TRUE assistant-only loss for the trained adapters — forward passes only,
no training, no gradient updates.

Why this exists: AUDIT.md §B confirmed `assistant_only_loss=False` on both real
QLoRA runs (adapters/qwen-swarm, adapters/qwen-swarm-v2) — train_qlora.py computed
loss over the full templated prompt + JSON answer, not the answer alone. The
reported 0.053 could be "the model reasons well" or "the model reproduces a
repetitive prompt template it has seen hundreds of times" and the trainer's own
logs cannot tell those apart. This script re-tokenizes each val row exactly as
train_qlora.py does, manually locates the assistant span, and reports BOTH the
correctly-masked assistant-only loss and the original unmasked full-sequence loss
for the same rows — so the gap between them is directly visible.

It also reports the base model (no adapter) on each adapter's own matching val set,
so "how much did the fine-tune improve the tokens that matter" is a direct
subtraction, without spending any GPU time on a retrain.

Design: one base model load in 4-bit (matching train_qlora.py's BitsAndBytesConfig
exactly), then PEFT's disable_adapter()/set_adapter() to switch between "no
adapter", "qwen-swarm", and "qwen-swarm-v2" on the SAME loaded base weights —
cheaper than three separate loads and guarantees byte-identical base weights
across all three measurements.

Usage:
    python llm_finetuning/measure_masked_loss.py
    python llm_finetuning/measure_masked_loss.py --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_val_rows(path: Path, limit: int | None = None):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def find_assistant_boundary(tok, messages, full_text: str):
    """Returns (token_boundary_index, ok). token_boundary_index is the first
    token of full_text's tokenization that belongs to the assistant turn.
    ok=False means the prefix-alignment assumption failed for this row (caller
    should skip it and count it, not silently mismeasure it)."""
    prompt_text = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        return None, False
    split_char = len(prompt_text)

    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    boundary = len(offsets)  # default: nothing is assistant (shouldn't happen)
    for i, (start, _end) in enumerate(offsets):
        if start >= split_char:
            boundary = i
            break
    return boundary, True


def measure_system(model, tok, rows, device, label: str, val_file: str):
    import torch
    import torch.nn.functional as F

    total_masked_loss_sum = 0.0
    total_masked_tokens = 0
    total_unmasked_loss_sum = 0.0
    total_unmasked_tokens = 0
    assistant_pct_sum = 0.0
    n_scored = 0
    n_skipped_boundary = 0
    n_skipped_empty = 0

    model.eval()
    with torch.no_grad():
        for row in rows:
            messages = row["messages"]
            full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            boundary, ok = find_assistant_boundary(tok, messages, full_text)
            if not ok:
                n_skipped_boundary += 1
                continue

            enc = tok(full_text, add_special_tokens=False)
            input_ids = enc["input_ids"]
            if boundary >= len(input_ids):
                n_skipped_empty += 1
                continue

            input_ids_t = torch.tensor([input_ids], device=device)
            attn_t = torch.ones_like(input_ids_t)
            logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits[0]  # (seq_len, vocab)

            shift_logits = logits[:-1, :]
            shift_input_ids = input_ids_t[0, 1:]

            labels_masked = list(input_ids)
            for i in range(boundary):
                labels_masked[i] = -100
            shift_labels_masked = torch.tensor(labels_masked[1:], device=device)

            n_masked_tokens = int((shift_labels_masked != -100).sum().item())
            if n_masked_tokens == 0:
                n_skipped_empty += 1
                continue

            masked_sum = F.cross_entropy(shift_logits.float(), shift_labels_masked,
                                          ignore_index=-100, reduction="sum")
            unmasked_sum = F.cross_entropy(shift_logits.float(), shift_input_ids, reduction="sum")

            total_masked_loss_sum += masked_sum.item()
            total_masked_tokens += n_masked_tokens
            total_unmasked_loss_sum += unmasked_sum.item()
            total_unmasked_tokens += shift_input_ids.numel()
            assistant_pct_sum += 100.0 * n_masked_tokens / len(input_ids)
            n_scored += 1

    result = {
        "system": label,
        "val_file": val_file,
        "n_rows_total": len(rows),
        "n_rows_scored": n_scored,
        "n_rows_skipped_boundary_mismatch": n_skipped_boundary,
        "n_rows_skipped_empty_assistant_span": n_skipped_empty,
        "masked_loss_assistant_only": (total_masked_loss_sum / total_masked_tokens
                                        if total_masked_tokens else None),
        "unmasked_loss_full_sequence": (total_unmasked_loss_sum / total_unmasked_tokens
                                         if total_unmasked_tokens else None),
        "assistant_token_pct_mean": (assistant_pct_sum / n_scored if n_scored else None),
        "total_assistant_tokens_scored": total_masked_tokens,
        "total_sequence_tokens_scored": total_unmasked_tokens,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per val file (smoke test)")
    ap.add_argument("--out", default=str(REPO / "evaluation/masked_loss.json"))
    ap.add_argument("--common-heldout", default=None,
                     help="path to a single val file (relative to repo root) that NONE of the "
                          "three systems trained on. When set, scores base/qwen-swarm/"
                          "qwen-swarm-v2 all on this one file (removing the different-val-set "
                          "confound from the default per-adapter-own-val-set plan) and writes "
                          "the result under a 'common_heldout' key, merged into --out rather "
                          "than overwriting its existing 'rows' key.")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"loading tokenizer + base model in 4-bit: {args.base}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Mirrors train_qlora.py's BitsAndBytesConfig exactly.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    device = base_model.device

    print("wrapping with PEFT (multi-adapter, switchable) ...", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, str(REPO / "adapters/qwen-swarm"),
                                            adapter_name="qwen-swarm")
    peft_model.load_adapter(str(REPO / "adapters/qwen-swarm-v2"), adapter_name="qwen-swarm-v2")

    if args.common_heldout:
        heldout_path = REPO / args.common_heldout
        plan = [
            ("base (no adapter)", None, heldout_path),
            ("qwen-swarm", "qwen-swarm", heldout_path),
            ("qwen-swarm-v2", "qwen-swarm-v2", heldout_path),
        ]
    else:
        plan = [
            ("base (no adapter)", None, REPO / "data/sft_train_val.jsonl"),
            ("qwen-swarm", "qwen-swarm", REPO / "data/sft_train_val.jsonl"),
            ("base (no adapter)", None, REPO / "data/sft_train_v2_val.jsonl"),
            ("qwen-swarm-v2", "qwen-swarm-v2", REPO / "data/sft_train_v2_val.jsonl"),
        ]

    results = []
    for label, adapter_name, val_path in plan:
        rows = load_val_rows(val_path, args.limit)
        print(f"measuring: {label} on {val_path.name} ({len(rows)} rows)", flush=True)
        if adapter_name is None:
            with peft_model.disable_adapter():
                res = measure_system(peft_model, tok, rows, device, label, str(val_path.relative_to(REPO)))
        else:
            peft_model.set_adapter(adapter_name)
            res = measure_system(peft_model, tok, rows, device, label, str(val_path.relative_to(REPO)))
        results.append(res)
        print(f"  masked={res['masked_loss_assistant_only']:.4f}  "
              f"unmasked={res['unmasked_loss_full_sequence']:.4f}  "
              f"assistant%={res['assistant_token_pct_mean']:.1f}  "
              f"(scored {res['n_rows_scored']}/{res['n_rows_total']})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing["base_model"] = args.base
    existing["note"] = ("masked_loss_assistant_only = mean cross-entropy over assistant-turn tokens "
                         "only (labels=-100 before the assistant span). unmasked_loss_full_sequence = "
                         "mean cross-entropy over the entire prompt+answer sequence, i.e. what "
                         "train_qlora.py actually optimized (assistant_only_loss=False, per AUDIT.md "
                         "sec B). Both are token-count-weighted means across all scored rows, not a "
                         "mean of per-row means.")
    if args.common_heldout:
        existing["common_heldout"] = {
            "val_file": args.common_heldout,
            "note": ("All three systems scored on the SAME file, which none of them trained on "
                     "(sft_train_final.jsonl/_val is 100% teacher prose, built after and separately "
                     "from qwen-swarm/qwen-swarm-v2's training data) — removes the different-val-set "
                     "confound present in the per-adapter-own-val-set 'rows' above."),
            "rows": results,
        }
    else:
        existing["rows"] = results
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {out_path}")

    print("\n| system | val_file | n_scored | masked (assistant-only) | unmasked (full-seq) | assistant token % |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['system']} | {Path(r['val_file']).name} | {r['n_rows_scored']} | "
              f"{r['masked_loss_assistant_only']:.4f} | {r['unmasked_loss_full_sequence']:.4f} | "
              f"{r['assistant_token_pct_mean']:.1f}% |")


if __name__ == "__main__":
    main()
