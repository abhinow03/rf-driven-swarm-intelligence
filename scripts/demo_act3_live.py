"""Defense demo, Act 3: live scenario, full pipeline_v2, presenter-driven input.

Disclosure-first (LOCKED CONFIG step a): prints v5-a's real, already-measured
0.0% correct-abstention rate on structurally unanswerable cases before
accepting any input -- pulled from AUDIT.md sec AI's Phase 4 finding (502
genuinely unanswerable multi-hop cases), not rounded favorably.

Live input is two formation names (e.g. "column encirclement"); the demo
builds a scenario the SAME way this project's own corpus generation does --
llm_finetuning/build_sft_dataset.synth_context(form_a, form_b, rng), reused
unmodified, not reimplemented -- then runs it through pipeline_v2.assess_ctx()
exactly as Act 2 does, same hot-swap client, same layer-attribution printing.

Usage:
    python scripts/demo_act3_live.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "config"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent import pipeline_v2  # noqa: E402
from swarm_intent.config import BASE_FORMATIONS  # noqa: E402
from build_sft_dataset import synth_context  # noqa: E402

import demo_common as dc  # noqa: E402
import demo_config  # noqa: E402

DISCLOSURE = f"""
{'=' * 90}
DISCLOSURE (read before any live input is accepted)
{'=' * 90}
config/demo_config.ACTIVE_ADAPTER ({demo_config.ACTIVE_ADAPTER.split('/')[-1] or demo_config.ACTIVE_ADAPTER}) has
NOT been trained on abstention. On the 502 genuinely unanswerable (structurally
multi-hop formation-chain) cases in this project's locked Phase 4 evaluation,
it shows 0.0% correct-abstention -- it will confidently narrate an answer even
when the correct response is "I don't know." (AUDIT.md sec AI). This is not
unique to this checkpoint: the prior fine-tuned baseline (v2) shows the same
0.0%.

A successor model trained specifically on abstention examples (v5a2,
docs/PREREGISTRATION_V5A2.md) is in progress and has not yet been scored
against its preregistered bars as of this demo.

Any live scenario below that resembles a multi-hop or oscillation pattern
should be expected to get a confident, potentially wrong answer -- not a
hedge. That failure mode is being shown deliberately, not hidden.
{'=' * 90}
"""


def prompt_formations() -> tuple[str, str] | None:
    print(f"Valid formations: {', '.join(BASE_FORMATIONS)}")
    raw = input("Enter two formations, space-separated (e.g. 'column encirclement'), "
               "or blank to skip Act 3: ").strip()
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2:
        print(f"expected exactly 2 formation names, got {len(parts)} -- try again")
        return prompt_formations()
    form_a, form_b = parts
    if form_a not in BASE_FORMATIONS or form_b not in BASE_FORMATIONS:
        bad = [f for f in (form_a, form_b) if f not in BASE_FORMATIONS]
        print(f"not a valid formation: {bad} -- try again")
        return prompt_formations()
    return form_a, form_b


def run_scenario(client: "dc.HotSwapClient", form_a: str, form_b: str, seed: int = None) -> dict:
    """Pure data, no printing -- reused by print_scenario() below and
    scripts/demo_web.py's live /api/act3 endpoint, so the terminal and web
    presentations run the exact same scenario-build + pipeline path."""
    if form_a not in BASE_FORMATIONS or form_b not in BASE_FORMATIONS:
        bad = [f for f in (form_a, form_b) if f not in BASE_FORMATIONS]
        return {"form_a": form_a, "form_b": form_b, "error": f"not a valid formation: {bad}"}

    class_freq = pipeline_v2._train_class_freq(demo_config.ACTIVE_ADAPTER_TRAIN_FILE)
    rng = random.Random(seed if seed is not None else random.SystemRandom().randint(0, 2**31))

    try:
        ctx, key_windows = synth_context(form_a, form_b, rng)
    except Exception as e:
        return {"form_a": form_a, "form_b": form_b,
                "error": f"SCENARIO BUILD FAILED: {type(e).__name__}: {e}"}

    result = dc.run_pipeline_case(client, class_freq, ctx, key_windows)
    if result["error"]:
        return {"form_a": form_a, "form_b": form_b, "ctx": ctx,
                "error": f"GENERATION FAILED: {result['error']}"}

    return {"form_a": form_a, "form_b": form_b, "ctx": ctx, "error": None,
            "bucket": result["bucket"], "layer": result["layer"],
            "detail": result["detail"], "assessment": result["assessment"]}


def print_scenario(data: dict) -> None:
    dc.banner(f"ACT 3 LIVE: {data['form_a']} -> {data['form_b']}")
    if data["error"] and "ctx" not in data:
        print(data["error"])
        print()
        return
    print(data["ctx"])
    print()
    if data["error"]:
        print(data["error"])
        print("(no assessment produced -- this is reported as a failure, not silently "
             "relabeled as a successful abstention or answer)")
        return

    layer, detail, assessment = data["layer"], data["detail"], data["assessment"]
    dc.layer_banner(layer)
    if layer == pipeline_v2.LAYER_1_DETERMINISTIC:
        print(f"rule-table key: {detail['rules_key']}")
    elif layer == pipeline_v2.LAYER_2_GUARD:
        print("guard reason(s): " + "; ".join(pipeline_v2.GUARD_REASON_TEXT.get(r, r)
                                              for r in detail["guard_reasons"]))
    else:
        print(f"subtype: {detail['subtype']} -- no RULES entry exists for this pair, "
             f"ACTIVE_ADAPTER's judgment is load-bearing")

    print()
    print("final assessment:")
    for k in ("threat_level", "likely_intent", "recommended_action", "confidence_in_assessment"):
        print(f"  {k}: {assessment.get(k)}")
    print()


def run(client: "dc.HotSwapClient" = None, seed: int = None):
    print(DISCLOSURE)

    owns_client = client is None
    if owns_client:
        dc.gpu_free_gib(required_gib=15.0)
        client = dc.HotSwapClient(demo_config.BASE_MODEL, temperature=0.0)
        client.add_adapter("active", demo_config.ACTIVE_ADAPTER)

    formations = prompt_formations()
    if formations is None:
        print("no input given -- Act 3 skipped")
        return

    form_a, form_b = formations
    data = run_scenario(client, form_a, form_b, seed=seed)
    print_scenario(data)

    if owns_client:
        del client
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
