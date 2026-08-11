"""
QLoRA SFT training for v5-a: the plain-SFT baseline arm of the planned Phase 2
4-way ablation (v5-a plain SFT / v5-b + context distillation / v5-c +
logit-adjusted loss / v5-d unbalanced control -- only v5-a is built this
session; the other three are follow-on work, see V5_LOG.md).

QLoRA, Qwen2.5-7B-Instruct, 4-bit NF4 double-quant. LoRA r=32/alpha=64 on all
seven projections (q/k/v/o/gate/up/down_proj) -- deliberately larger than
train_qlora.py's r=16/alpha=32, given the ~10x larger Phase 1 corpus.

assistant_only_loss=True is REQUIRED here (not an opt-in flag like
train_qlora.py's --assistant-only-loss) -- this project's own history (v3a vs
v3a-nomask) measured +13.5pt in-distribution / +21.8pt mean-under-perturbation
accuracy from this one setting. train_qlora.py already established that
Qwen2.5-7B-Instruct's OWN chat template has no {% generation %} markers
(apply_chat_template silently returns an all-zero assistant mask, no error)
and that trl>=1.8 auto-swaps in its bundled qwen2_5_training.jinja whenever the
dataset still exposes the raw "messages" field. verify_assistant_mask() below
re-checks that finding explicitly (tokenizer-only, no model weights) rather
than trusting it blindly, since a trl/transformers version bump could change
it silently.

--dry-run is the ONLY mode this script has been exercised in so far (AUDIT.md
V5 Phase 1 step 3 parallel-prep session: Phase 1 corpus regeneration was using
the network/CPU -- not GPU -- in a separate tmux session at the time this was
written; this script independently must not touch the GPU regardless, since
there is no trained v5-a checkpoint yet). Dry-run builds the model on
device_map="meta" (zero real weights materialized) and asserts
torch.cuda.memory_allocated() == 0 both before and after.

Usage:
    python llm_finetuning/train_sft_v5.py --dry-run
    # real run (NOT exercised this session):
    python llm_finetuning/train_sft_v5.py --train data/sft_train_v5_phase1.jsonl \
        --val data/sft_train_v5_phase1_val.jsonl --out checkpoints/v5_sft/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

LORA_R = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SIZE = 32  # agreed v5 plan -- 8*4, NOT 4*4
assert PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS == EFFECTIVE_BATCH_SIZE, \
    f"effective batch size drifted: {PER_DEVICE_TRAIN_BATCH_SIZE}*{GRADIENT_ACCUMULATION_STEPS} != {EFFECTIVE_BATCH_SIZE}"

LEARNING_RATE = 1e-4  # not 2e-4 -- lower given the larger corpus + longer run
OUTPUT_SCHEMA_KEYS = {"situation_summary", "threat_level", "threat_reasoning", "likely_intent",
                      "recommended_action", "confidence_in_assessment", "key_indicators",
                      "follow_up_watch"}

# Conservative floors, not the exact Phase 1 target (12,000 total, val_frac=0.1) --
# dedup/regeneration can shift exact counts slightly. These exist only to catch
# GROSSLY wrong inputs (an empty file, a stale ~20-row smoke-test file mistaken
# for the real corpus, a schema that doesn't match OUTPUT_SCHEMA at all) -- not
# to second-guess Phase 1's own quality gates (AUDIT.md V5_LOG.md), which is a
# separate, already-tracked concern.
MIN_TRAIN_ROWS = 9000
MIN_VAL_ROWS = 400


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_corpus(path: str, min_rows: int, label: str) -> list[dict]:
    """Fails loudly (not silently) if the file is missing, too small, or has
    rows that don't match the schema every downstream script in this project
    assumes -- training on partial/malformed data without noticing is exactly
    the failure mode this guards against."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} corpus not found: {path}. Phase 1 corpus generation/regeneration "
            f"must complete before real training can start -- see docs/V5_STATE.json "
            f"phase1_step2_corpus_generation / phase1_step3_step1_teacher_prose_gap.")

    rows = load_jsonl(path)
    if len(rows) < min_rows:
        raise ValueError(
            f"{label} corpus at {path} has only {len(rows)} rows, expected at least "
            f"{min_rows}. This looks like a stale/partial/smoke-test file, not the full "
            f"Phase 1 corpus -- refusing to train silently on it. Check docs/V5_STATE.json "
            f"for the corpus's actual current row count before proceeding.")

    schema_errors = []
    for i, r in enumerate(rows):
        msgs = r.get("messages")
        if not (isinstance(msgs, list) and len(msgs) == 2
                and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant"):
            schema_errors.append((i, "messages shape"))
            continue
        try:
            assistant = json.loads(msgs[1]["content"])
        except json.JSONDecodeError:
            schema_errors.append((i, "assistant content not valid JSON"))
            continue
        missing = OUTPUT_SCHEMA_KEYS - assistant.keys()
        if missing:
            schema_errors.append((i, f"missing keys {sorted(missing)}"))

    if schema_errors:
        sample = "; ".join(f"row {i}: {reason}" for i, reason in schema_errors[:5])
        raise ValueError(
            f"{label} corpus at {path}: {len(schema_errors)}/{len(rows)} rows failed schema "
            f"validation (expected {{messages: [user, assistant]}}, assistant content JSON "
            f"with all of {sorted(OUTPUT_SCHEMA_KEYS)}). First failures: {sample}")

    print(f"{label} corpus OK: {path} ({len(rows)} rows, schema-valid, >= {min_rows} floor)")
    return rows


def verify_assistant_mask(tokenizer, sample_messages: list[dict]) -> bool:
    """Tokenizer-only check (no model weights) that the assistant-token mask
    trl's SFTConfig(assistant_only_loss=True) depends on actually produces a
    non-trivial mask on THIS tokenizer/trl/transformers combination, rather
    than trusting train_qlora.py's prior finding to still hold. Returns
    whether the mask is real (any assistant tokens marked)."""
    from trl.chat_template_utils import has_generation_markers

    base_has_markers = has_generation_markers(tokenizer.chat_template)
    print(f"Base {MODEL_NAME} chat_template has {{% generation %}} markers: {base_has_markers}")

    training_template = None
    try:
        from trl.chat_template_utils import get_training_chat_template
        training_template = get_training_chat_template(tokenizer)
    except Exception as exc:  # pragma: no cover -- diagnostic only
        print(f"get_training_chat_template raised {exc!r}; falling back to base chat_template")

    effective_template = training_template or tokenizer.chat_template
    sample = tokenizer.apply_chat_template(
        sample_messages, tokenize=True, return_assistant_tokens_mask=True,
        return_dict=True, chat_template=effective_template)
    masks = sample.get("assistant_masks", [])
    n_masked = sum(masks)
    print(f"trl auto-swapped a training template: {training_template is not None and training_template != tokenizer.chat_template}")
    print(f"Assistant-mask check on a real row: {n_masked}/{len(masks)} tokens masked.")
    return n_masked > 0


def print_assistant_mask_example(tokenizer, sample_messages: list[dict]) -> None:
    """Prints the assistant portion the mask actually covers, token-aligned,
    so a human can visually confirm the mask lands on the assistant turn and
    not (e.g.) the system/user turns or the chat-template scaffolding."""
    from trl.chat_template_utils import get_training_chat_template

    try:
        training_template = get_training_chat_template(tokenizer)
    except Exception:
        training_template = None
    effective_template = training_template or tokenizer.chat_template
    sample = tokenizer.apply_chat_template(
        sample_messages, tokenize=True, return_assistant_tokens_mask=True,
        return_dict=True, chat_template=effective_template)
    ids, masks = sample["input_ids"], sample["assistant_masks"]
    masked_ids = [tid for tid, m in zip(ids, masks) if m]
    decoded = tokenizer.decode(masked_ids) if masked_ids else "<EMPTY -- mask is all zero>"
    print("Assistant-masked span decodes to:")
    print(decoded[:400] + ("..." if len(decoded) > 400 else ""))


def synthetic_stub_rows(n: int = 4) -> list[dict]:
    """Used only when no real val file is available/permitted for this run
    (e.g. this session, which must not read data/sft_train_v5_phase1_val.jsonl
    while Phase 1 regeneration is actively rewriting it in a separate tmux
    session -- see the --dry-run default --val path)."""
    rows = []
    for i in range(n):
        user = (f"Given this UAV swarm tactical context, write the JSON assessment.\n\n"
                f"Dominant formation: column\nFormation history: column\n"
                f"No formation transitions detected. [stub row {i}]")
        assistant = {
            "situation_summary": "Stub summary for dry-run tokenization only.",
            "threat_level": "low", "threat_reasoning": "stub",
            "likely_intent": "surveillance", "recommended_action": "monitor",
            "confidence_in_assessment": "high",
            "key_indicators": ["stub1", "stub2", "stub3"],
            "follow_up_watch": "stub",
        }
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": json.dumps(assistant)}]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--train", default="data/sft_train_v5_phase1.jsonl")
    ap.add_argument("--val", default="data/sft_train_v5_phase1_val.jsonl")
    ap.add_argument("--out", default="checkpoints/v5_sft/")
    ap.add_argument("--dry-run", action="store_true",
                     help="device_map='meta', zero GPU memory allocated, tokenizer/dataset/"
                          "collator/mask logic only. The only mode this script may be run in "
                          "while Phase 1 corpus work is still in flight (see module docstring).")
    ap.add_argument("--progress-task", default=None,
                     help="if set, writes evaluation/progress/<task>.json once per optimizer "
                          "step via swarm_intent.progress.Reporter, same convention as "
                          "train_qlora.py's --progress-task.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    if args.dry_run:
        pre_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        assert pre_alloc == 0, f"GPU memory already allocated before dry-run started: {pre_alloc} bytes"

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # If --val wasn't overridden and points at the live Phase 1 regeneration
        # target, use synthetic stubs instead of reading a file that's being
        # actively rewritten by a separate tmux job this session must not touch.
        if args.val == "data/sft_train_v5_phase1_val.jsonl":
            print("--dry-run with the default --val path: Phase 1 corpus regeneration owns "
                 "that file in a separate tmux session right now. Using synthetic stub rows "
                 "instead of reading it, per this session's hard no-touch constraint.")
            stub_rows = synthetic_stub_rows()
        elif os.path.exists(args.val):
            stub_rows = load_jsonl(args.val)[:4]
        else:
            print(f"--val path {args.val} does not exist yet; using synthetic stub rows.")
            stub_rows = synthetic_stub_rows()

        mask_ok = verify_assistant_mask(tok, stub_rows[0]["messages"])
        assert mask_ok, "assistant-token mask is all-zero -- assistant_only_loss=True would silently train on the full sequence"
        print_assistant_mask_example(tok, stub_rows[0]["messages"])

        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="meta")
        lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=LORA_TARGET_MODULES)
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        post_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        assert post_alloc == 0, (
            f"dry-run allocated {post_alloc} bytes of real GPU memory -- device_map='meta' "
            f"should have kept every parameter meta/uninitialized")
        print(f"DRY RUN OK: torch.cuda.memory_allocated() == 0 before and after "
             f"({'no CUDA device visible' if not torch.cuda.is_available() else 'CUDA present, unused'}).")
        print(f"LoRA config: r={LORA_R} alpha={LORA_ALPHA} targets={LORA_TARGET_MODULES}")
        print(f"Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}*{GRADIENT_ACCUMULATION_STEPS} "
             f"= {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        return

    # --------------------- real training path (not run this session) ---------------------
    train_rows = validate_corpus(args.train, MIN_TRAIN_ROWS, "train")
    val_rows = validate_corpus(args.val, MIN_VAL_ROWS, "val")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mask_ok = verify_assistant_mask(tok, val_rows[0]["messages"])
    assert mask_ok, "assistant-token mask is all-zero -- refusing to train with assistant_only_loss=True unmasked"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds_train = load_dataset("json", data_files=args.train, split="train")
    ds_val = load_dataset("json", data_files=args.val, split="train")

    sft_cfg = SFTConfig(
        assistant_only_loss=True,
        output_dir=args.out,
        num_train_epochs=3,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        # group_by_length=True was specified in the agreed v5 plan but is NOT a valid
        # SFTConfig/TrainingArguments field in the installed transformers==5.13.1 /
        # trl==1.8.0 (removed upstream -- confirmed via inspect.signature(), not a typo).
        # No equivalent replacement kwarg exists in this version. Deliberately omitted
        # rather than worked around; flagged plainly in V5_LOG.md, not silently dropped.
        bf16=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    assert sft_cfg.assistant_only_loss is True, "SFTConfig.assistant_only_loss is not True"
    assert sft_cfg.per_device_train_batch_size * sft_cfg.gradient_accumulation_steps == EFFECTIVE_BATCH_SIZE, \
        "effective batch size drifted from the agreed v5 plan"
    print(f"CONFIRMED before training starts: assistant_only_loss={sft_cfg.assistant_only_loss}, "
         f"effective_batch_size={sft_cfg.per_device_train_batch_size}*"
         f"{sft_cfg.gradient_accumulation_steps}={sft_cfg.per_device_train_batch_size * sft_cfg.gradient_accumulation_steps}")

    callbacks = []
    if args.progress_task:
        from swarm_intent.progress import Reporter
        from transformers import TrainerCallback

        steps_per_epoch = -(-len(ds_train) // (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
        total_steps = round(steps_per_epoch * sft_cfg.num_train_epochs)
        reporter = Reporter(args.progress_task, total_steps)

        class ReporterCallback(TrainerCallback):
            def on_step_end(self, targs, state, control, **kwargs):
                reporter.update(1, item=f"step {state.global_step}/{total_steps} epoch {state.epoch:.2f}")

            def on_train_end(self, targs, state, control, **kwargs):
                reporter.status = "done"
                reporter._write()

        callbacks.append(ReporterCallback())
        print(f"progress reporter on: evaluation/progress/{args.progress_task}.json, "
             f"{total_steps} total steps estimated ({steps_per_epoch}/epoch x {sft_cfg.num_train_epochs} epochs)")

    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds_train,
                        eval_dataset=ds_val, processing_class=tok, callbacks=callbacks or None)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    ta = torch.load(f"{args.out}/training_args.bin", weights_only=False)
    assert ta.assistant_only_loss is True, "training_args.bin does not show assistant_only_loss=True"
    print(f"training_args.bin assistant_only_loss = {ta.assistant_only_loss}")
    print(f"Saved v5-a LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
