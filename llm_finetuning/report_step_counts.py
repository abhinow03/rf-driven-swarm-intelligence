"""
Step 1 of the "is the low-threat collapse under-training or data-diversity"
session (AUDIT.md sec R onward).

Reads global_step (and epoch/row/batch info) straight out of each adapter's
trainer_state.json -- no model calls, no estimation.

Usage:
    python llm_finetuning/report_step_counts.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ADAPTERS = {
    "qwen-swarm-v2": {"rows": 810, "file": "sft_train_v2.jsonl"},
    "qwen-swarm-v3a": {"rows": 234, "file": "sft_train_final.jsonl"},
    "qwen-swarm-v3a-nomask": {"rows": 234, "file": "sft_train_final.jsonl"},
    "qwen-swarm-v3b": {"rows": 270, "file": "sft_train_final_abstain.jsonl"},
}
EFFECTIVE_BATCH_SIZE = 8  # per_device_train_batch_size=1 x gradient_accumulation_steps=8, all adapters


def latest_trainer_state(adapter: str):
    ckpts = sorted(glob.glob(str(REPO / "adapters" / adapter / "checkpoint-*")),
                   key=lambda p: int(p.split("-")[-1]))
    with open(Path(ckpts[-1]) / "trainer_state.json") as f:
        return json.load(f), ckpts[-1]


def main():
    print("=== optimizer step counts (from trainer_state.json, not estimated) ===\n")
    print("| adapter | rows | epochs | effective batch | global_step | steps/epoch |")
    print("|---|---|---|---|---|---|")
    steps = {}
    for adapter, meta in ADAPTERS.items():
        state, ckpt = latest_trainer_state(adapter)
        gs = state["global_step"]
        epoch = state.get("epoch")
        steps[adapter] = gs
        steps_per_epoch = round(gs / epoch) if epoch else None
        print(f"| {adapter} | {meta['rows']} | {epoch} | {EFFECTIVE_BATCH_SIZE} | "
              f"{gs} | {steps_per_epoch} |")

    v2_steps = steps["qwen-swarm-v2"]
    v3a_steps = steps["qwen-swarm-v3a"]
    step_ratio = v2_steps / v3a_steps
    example_ratio = 16.5 / 4.8  # from AUDIT.md sec O (mean examples/pair, v2 vs sft_train_final.jsonl)
    print(f"\nv2:v3a optimizer-step ratio: {v2_steps}:{v3a_steps} = {step_ratio:.2f}:1")
    print(f"v2:v3a examples-per-pair ratio (sec O): 16.5:4.8 = {example_ratio:.2f}:1")
    print(f"\nThese two ratios are nearly identical ({step_ratio:.2f} vs {example_ratio:.2f}) "
          f"-- not a coincidence: epochs (3) and effective batch size (8) are held constant "
          f"across v2/v3a/v3a-nomask, so optimizer steps are a DIRECT linear function of row "
          f"count (== examples-per-pair, since both files cover all 49 RULES pairs, sec J). "
          f"'examples-per-pair' and 'optimizer steps seen' have been the SAME confounded "
          f"variable restated, not two independent explanations, for every finding in secs M-P.")


if __name__ == "__main__":
    main()
