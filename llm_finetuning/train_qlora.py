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
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


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
    ap.add_argument("--load-best-model", action="store_true",
                     help="keep the checkpoint with the lowest eval_loss across all epochs "
                          "(trainer.save_model() then saves THAT checkpoint's weights, not "
                          "necessarily the last epoch's) -- for high-epoch runs where "
                          "overfitting past the optimum is a real risk. Off by default so "
                          "v1/v2/v3a/v3b stay exactly reproducible if ever re-run.")
    ap.add_argument("--progress-task", default=None,
                     help="if set, writes evaluation/progress/<task>.json once per optimizer "
                          "step via swarm_intent.progress.Reporter, total=expected step count "
                          "computed from --epochs and the training file's row count.")
    ap.add_argument("--assistant-only-loss", action="store_true",
                     help="mask the loss to assistant turns only (trl SFTConfig.assistant_only_loss). "
                          "Default off to keep v1/v2 adapters exactly reproducible. Qwen2.5-7B-Instruct's "
                          "OWN chat template has no {%% generation %%} markers -- apply_chat_template("
                          "return_assistant_tokens_mask=True) on it silently returns an all-zero mask, "
                          "no error, just a warning (verified before adding this flag). trl>=1.8 detects "
                          "this and auto-swaps in its bundled qwen2_5_training.jinja (which does have the "
                          "markers) whenever the dataset is conversational -- see chat_template_utils."
                          "get_training_chat_template. That auto-swap requires the dataset to still expose "
                          "the raw 'messages' field, so when this flag is set we do NOT pre-flatten to a "
                          "'text' column below.")
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

    if not args.assistant_only_loss:
        # Legacy path (v1/v2): flatten to a plain "text" column up front, exactly
        # as before -- untouched so those adapters stay reproducible.
        def to_text(ex):
            return {"text": tok.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False)}
        ds_train = ds_train.map(to_text)
        ds_val   = ds_val.map(to_text)
    # else: leave ds_train/ds_val as raw {"messages": [...]} rows. SFTTrainer's
    # own tokenization path needs the conversational "messages" field present to
    # detect assistant_only_loss is usable and auto-swap in the training chat
    # template with {% generation %} markers (see --assistant-only-loss help).

    sft_cfg = SFTConfig(
        assistant_only_loss=args.assistant_only_loss,
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
        load_best_model_at_end=args.load_best_model,
        metric_for_best_model="eval_loss" if args.load_best_model else None,
        greater_is_better=False if args.load_best_model else None,
    )

    callbacks = []
    if args.progress_task:
        from swarm_intent.progress import Reporter
        from transformers import TrainerCallback

        steps_per_epoch = -(-len(ds_train) // (sft_cfg.per_device_train_batch_size * args.grad_accum))  # ceil
        total_steps = round(steps_per_epoch * args.epochs)
        reporter = Reporter(args.progress_task, total_steps)

        class ReporterCallback(TrainerCallback):
            def on_step_end(self, targs, state, control, **kwargs):
                reporter.update(1, item=f"step {state.global_step}/{total_steps} "
                                        f"epoch {state.epoch:.2f}")

            def on_train_end(self, targs, state, control, **kwargs):
                reporter.status = "done"
                reporter._write()

        callbacks.append(ReporterCallback())

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tok,
        callbacks=callbacks or None,
    )
    if args.assistant_only_loss:
        from trl.chat_template_utils import has_generation_markers
        template_has_markers = has_generation_markers(tok.chat_template)
        effective_template = trainer.chat_template or tok.chat_template
        print(f"assistant_only_loss requested. Base tokenizer chat_template has "
              f"{{% generation %}} markers: {template_has_markers}. "
              f"trl auto-swapped a training template: {trainer.chat_template is not None}.")
        sample = tok.apply_chat_template(
            ds_train[0]["messages"], tokenize=True, return_assistant_tokens_mask=True,
            return_dict=True, chat_template=effective_template)
        n_masked = sum(sample["assistant_masks"])
        assert n_masked > 0, "assistant_masks are all zero -- the chat template swap did not take effect"
        print(f"Verified on a real training row: {n_masked}/{len(sample['assistant_masks'])} "
              f"tokens masked as assistant-only.")

    trainer.train(resume_from_checkpoint=args.resume or None)

    eval_curve = [(e["epoch"], e["eval_loss"]) for e in trainer.state.log_history if "eval_loss" in e]
    print("eval_loss by epoch:", ", ".join(f"{ep:.1f}->{loss:.4f}" for ep, loss in eval_curve))
    if args.load_best_model:
        print(f"best checkpoint: {trainer.state.best_model_checkpoint} "
              f"(eval_loss={trainer.state.best_metric:.4f}) -- this is what gets saved below, "
              f"not necessarily the final epoch's weights.")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    import torch as _torch
    ta = _torch.load(f"{args.out}/training_args.bin", weights_only=False)
    print(f"training_args.bin assistant_only_loss = {ta.assistant_only_loss}")
    assert ta.assistant_only_loss == args.assistant_only_loss, \
        "training_args.bin does not reflect the requested --assistant-only-loss setting"

    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
