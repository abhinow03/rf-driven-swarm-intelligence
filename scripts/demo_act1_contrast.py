"""Defense demo, Act 1: three-system contrast on one real scenario.

Loads the 4-bit Qwen2.5-7B-Instruct base ONCE, then hot-swaps between no
adapter (base) / qwen-swarm-v2 / ACTIVE_ADAPTER (config/demo_config.py --
currently v5-a) via demo_common.HotSwapClient, instead of three separate
7B loads (the phase4-baseline OOM incident this project already hit once).

The scenario is pulled programmatically from the locked seed=999 eval set
(config/demo_config.SEED999_EVAL_SET): the first has_ground_truth=True case
where v5-a's ALREADY-MEASURED real output (evaluation/v5a_seed999_results.json)
is correct -- not hand-picked. Reuses build_llm_prompt/OUTPUT_SCHEMA and the
same match_threat/is_abstention scoring convention every eval script in this
project uses -- no new scoring logic here.

Usage:
    python scripts/demo_act1_contrast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402
from swarm_intent.llm.client import _parse_llm_response  # noqa: E402


def run(client: "dc.HotSwapClient" = None):
    case = dc.pick_act1_case()
    prompt = dc.case_prompt(case)

    dc.banner(f"ACT 1: three-system contrast -- scenario {case['name']!r} "
             f"({case['pair'][0]} -> {case['pair'][1]}, expected_threat={case['expected_threat']})")
    print(case["ctx"])
    print()

    owns_client = client is None
    if owns_client:
        dc.gpu_free_gib(required_gib=15.0)
        client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
        client.add_adapter("v2", demo_config.V2_ADAPTER)
        client.add_adapter("active", demo_config.ACTIVE_ADAPTER)

    systems = [
        ("base (no adapter)", None, dc.FACT_BASE_IS_WEAKEST),
        ("v2 (prior fine-tuned baseline)", "v2", dc.FACT_V2_ZERO_ABSTENTION),
        ("v5-a (current production, config/demo_config.ACTIVE_ADAPTER)", "active", dc.FACT_V5A_HEADLINE),
    ]

    for label, adapter_name, fact in systems:
        dc.banner(label, char="-")
        with client.use_adapter(adapter_name):
            try:
                raw = client.generate(prompt)
            except Exception as e:
                print(f"GENERATION FAILED: {type(e).__name__}: {e}")
                continue
        parsed = _parse_llm_response(raw)
        correct = dc.is_case_correct(parsed, case)
        verdict = "correct" if correct else ("incorrect" if correct is False else "abstained/unparseable")
        print(raw.strip()[:1000])
        print(f"\n>>> {verdict} -- {fact}")
        print()

    if owns_client:
        del client
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
