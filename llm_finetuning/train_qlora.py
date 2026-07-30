"""
QLoRA supervised fine-tuning of a 7B instruct model for swarm tactical reasoning.

Tuned to fit a FREE Colab T4 (16 GB):
  - 4-bit NF4 quantisation (bitsandbytes)
  - LoRA adapters (r=16, alpha=32) on attention + MLP projections
  - batch_size=1 + grad accumulation, gradient checkpointing

Recommended base model: Qwen2.5-7B-Instruct (Apache-2.0, strong JSON adherence).
OOM fallback: --model Qwen/Qwen2.5-3B-Instruct

Usage (Colab):
    pip install -U transformers peft bitsandbytes datasets accelerate trl
    python llm_finetuning/train_qlora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train data/sft_train.jsonl --val data/sft_train_val.jsonl \
        --out adapters/qwen-swarm
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--val",   default="data/sft_train_val.jsonl")
    ap.add_argument("--out",   default="adapters/qwen-swarm")
    ap.add_argument("--epochs",      type=float, default=3.0)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lora-r",      type=int,   default=16)
    ap.add_argument("--lora-alpha",  type=int,   default=32)
    ap.add_argument("--grad-accum",  type=int,   default=8)
    ap.add_argument("--resume", default=None, help="path to checkpoint dir to resume from")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # Apply LoRA manually — more robust than passing peft_config to SFTTrainer
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds_train = load_dataset("json", data_files=args.train, split="train")
    ds_val   = load_dataset("json", data_files=args.val,   split="train")

    def to_text(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds_train = ds_train.map(to_text)
    ds_val   = ds_val.map(to_text)

    sft_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
