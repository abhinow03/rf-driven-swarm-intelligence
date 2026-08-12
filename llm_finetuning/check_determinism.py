"""
Determinism re-run: confirms greedy decoding (temperature=0.0, do_sample=False)
on v5-a's real adapter actually produces identical output across repeated
calls on the SAME prompts. This check did not exist before -- AUDIT.md
sec AI's follow-up audit found it had never been done and asked for it
explicitly, rather than letting it be silently assumed.

Usage:
    python llm_finetuning/check_determinism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "llm_finetuning"))

from eval_sft_v5 import (  # noqa: E402
    resolve_adapter_path, load_real_client, build_prompt_from_phase4_item, load_phase4_items,
)

N_SAMPLE = 10


def main():
    items = load_phase4_items()[:N_SAMPLE]
    prompts = [build_prompt_from_phase4_item(it) for it in items]

    resolved = resolve_adapter_path(str(REPO / "checkpoints" / "v5_sft"))
    client = load_real_client("Qwen/Qwen2.5-7B-Instruct", resolved, temperature=0.0)

    print(f"=== run 1 ({N_SAMPLE} prompts, greedy) ===")
    run1 = client.generate_batch(prompts, batch_size=N_SAMPLE)
    print(f"=== run 2 (same {N_SAMPLE} prompts, same client, greedy) ===")
    run2 = client.generate_batch(prompts, batch_size=N_SAMPLE)

    diffs = [i for i in range(N_SAMPLE) if run1[i] != run2[i]]
    print(f"\ndiff count: {len(diffs)}/{N_SAMPLE}")
    for i in diffs:
        print(f"\n--- case {i} ({items[i]['name']}) DIFFERS ---")
        print("run1:", run1[i][:300])
        print("run2:", run2[i][:300])

    Path(REPO / "evaluation" / "determinism_check.json").write_text(json.dumps({
        "n_sample": N_SAMPLE, "diff_count": len(diffs), "diff_indices": diffs,
        "run1": run1, "run2": run2,
    }, indent=2))
    print(f"\nsaved evaluation/determinism_check.json")


if __name__ == "__main__":
    main()
