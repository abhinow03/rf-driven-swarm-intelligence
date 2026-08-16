"""Defense demo entrypoint: Act 1 -> pause -> Act 2 -> pause -> Act 3.

Loads the 4-bit Qwen2.5-7B-Instruct base ONCE (see scripts/demo_common.HotSwapClient)
and hot-swaps PEFT adapters (base / v2 / config.demo_config.ACTIVE_ADAPTER) across
all three acts, instead of reloading per-act -- same lesson as the phase4-baseline
OOM incident this project already hit once with sequential full reloads.

Before touching anything: re-verifies the protected v5-a checkpoint safety copy
(config/demo_config.PROTECTED_CHECKPOINT_DIR) is still byte-identical/read-only
(scripts/phase3a_verify_safety_copy.py, reused unmodified) and checks free GPU
memory, halting rather than attempting and crashing mid-demo.

Usage:
    python run_demo.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

import demo_act1_contrast as act1  # noqa: E402
import demo_act2_pipeline as act2  # noqa: E402
import demo_act3_live as act3  # noqa: E402
import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402


def pause(msg: str) -> None:
    input(f"\n--- {msg} [press Enter to continue] ---\n")


def verify_protected_checkpoint() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "phase3a_verify_safety_copy.py")],
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(
            "HALTING: the protected v5-a checkpoint safety copy failed its byte-identical/"
            "read-only check -- refusing to run the demo against a possibly-drifted checkpoint.")


def main() -> None:
    print(f"ACTIVE_ADAPTER = {demo_config.ACTIVE_ADAPTER}")
    verify_protected_checkpoint()

    free_gib = dc.gpu_free_gib(required_gib=15.0)
    print(f"free GPU memory: {free_gib:.1f} GiB -- proceeding")

    client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
    client.add_adapter("v2", demo_config.V2_ADAPTER)
    client.add_adapter("active", demo_config.ACTIVE_ADAPTER)
    print("base model loaded once; v2 and active adapters registered for hot-swap")

    act1.run(client=client)
    pause("End of Act 1 (three-system contrast)")

    act2.run(client=client)
    pause("End of Act 2 (pipeline architecture)")

    act3.run(client=client)

    print("\n=== demo complete ===")


if __name__ == "__main__":
    main()
