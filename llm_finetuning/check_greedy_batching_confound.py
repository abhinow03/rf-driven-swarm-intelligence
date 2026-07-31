"""
Ad-hoc check: does batched (left-padded) greedy decoding differ from unbatched
greedy decoding? sec U's equivalence check only validated batching under
SAMPLING (temperature=0.3, averaged over n_runs=5) -- greedy is deterministic
and single-shot, so it could be far more sensitive to left-padding numerical
artifacts (attention-mask/position-id edge effects) that sampling's averaging
washes out. Runs qwen-swarm-v3b on the 15 low-threat cases, greedy, batched vs
unbatched, to check before trusting either number in the step-1 report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "llm_finetuning"))

from swarm_intent.llm.client import LocalHFClient  # noqa: E402
from swarm_intent.llm.evaluate import evaluate_llm  # noqa: E402
from swarm_intent.llm.prompts import TEST_CASES  # noqa: E402

from baselines import make_rules_in_prompt_run_case, make_batched_run_case  # noqa: E402

low_cases = [c for c in TEST_CASES if c["expected_threat"] == "low"]
assert len(low_cases) == 15

client = LocalHFClient("Qwen/Qwen2.5-7B-Instruct", adapter_path=str(REPO / "adapters/qwen-swarm-v3b"),
                       temperature=0.0)

print("=== unbatched greedy ===")
run_case_unbatched = make_rules_in_prompt_run_case(client, seed=0)
res_unbatched = evaluate_llm(run_case_unbatched, low_cases, judge_client=None, n_runs=1)
print(f"accuracy_when_answerable={res_unbatched['aggregate']['accuracy_when_answerable']:.2%} "
     f"threat_acc={res_unbatched['aggregate']['mean_threat_accuracy']:.2%}")
for c in res_unbatched["per_case"]:
    print(f"  {c['name']}: majority_threat={c['majority_threat']} threat_accuracy={c['threat_accuracy']}")

print("\n=== batched (batch_size=8) greedy ===")
run_case_batched = make_batched_run_case(client, low_cases, 1, 8, seed=0)
res_batched = evaluate_llm(run_case_batched, low_cases, judge_client=None, n_runs=1)
print(f"accuracy_when_answerable={res_batched['aggregate']['accuracy_when_answerable']:.2%} "
     f"threat_acc={res_batched['aggregate']['mean_threat_accuracy']:.2%}")
for c in res_batched["per_case"]:
    print(f"  {c['name']}: majority_threat={c['majority_threat']} threat_accuracy={c['threat_accuracy']}")

Path(REPO / "evaluation" / "greedy_batching_confound_check.json").write_text(
    json.dumps({"unbatched": res_unbatched, "batched": res_batched}, indent=2))
