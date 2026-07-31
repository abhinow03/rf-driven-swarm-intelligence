"""
Evaluation for both the ML classifier and the LLM interpretation layer.

KEY FIX over the original eval (see CODE_REVIEW.md):
  * The LLM-as-judge is now a SEPARATE, independent client. Letting a model
    grade its own output (the original used the same call_llm) is invalid and
    produced 5/5 self-scores while objective accuracy was ~0. Pass a different
    provider/model as ``judge_client`` (e.g. judge with Groq llama-70b while
    evaluating your fine-tuned Qwen), or None to skip judging entirely.
  * Objective intent/threat accuracy is the HEADLINE metric and is always
    reported. Judge scores are secondary/advisory only.
  * recommended_action accuracy is now scored too (via match_action/ACTION_FAMILIES
    in prompts.py) — previously computed nowhere. abstention_rate ("unknown"
    responses) is reported as its own metric, separate from accuracy and
    hallucination_rate, since a correct abstention is neither a hit nor a hallucination.
  * ABSTENTION-AWARE SCORING: an abstaining response (likely_intent in
    ABSTENTION_TOKENS) is excluded from intent/threat/action accuracy AND
    hallucination_rate entirely, not counted as a miss or a hallucination in either.
    Before this fix, an always-abstaining run_case scored intent/threat/action
    accuracy 0.0 AND hallucination_rate 1.0 simultaneously — i.e. abstention was
    being penalized twice, as both "always wrong" and "always hallucinating", while
    abstention_rate=1.0 sat right next to those numbers uncommented. Per-case
    accuracy/hallucination fields are None (not 0.0) when every run for that case
    abstained — None means "not applicable", 0.0 would silently claim "confidently
    wrong every time", a different and false statement.
  * has_ground_truth=False cases (see llm_finetuning/degradation.py): for inputs
    that have no defensible expected answer, correct behaviour IS abstention.
    These cases skip intent/threat/action accuracy (there's nothing to compare
    against) and instead report correct_abstention_rate.
  * DECOMPOSED METRICS (not a single blended number): a severity group can mix
    has_ground_truth=True and False cases (e.g. degradation.py's dropped_lines
    axis, where whether the transition line survives is per-case), so picking
    ONE of mean_intent_accuracy / mean_correct_abstention_rate per cell silently
    drops whichever case type is a minority in that cell. The aggregate reports
    three separate, always-computed fields instead:
      - accuracy_when_answerable: intent accuracy, scored ONLY over
        has_ground_truth=True cases (alias of mean_intent_accuracy).
      - abstention_rate_when_unanswerable: how often the system abstained on
        has_ground_truth=False cases -- abstaining there IS correct, so this
        doubles as "correct abstention rate" (alias of mean_correct_abstention_rate).
      - over_abstention_rate (NEW): how often the system abstained on
        has_ground_truth=True cases, i.e. cases it should have been able to
        answer. This was previously invisible -- mean_abstention_rate blended
        it with abstention_rate_when_unanswerable into one number that couldn't
        distinguish "correctly declined an unanswerable case" from "wrongly
        declined an answerable one."
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

from .client import LLMClient
from .prompts import (TEST_CASES, JUDGE_PROMPT, match_intent, match_threat,
                      match_action, is_hallucination, is_abstention)

def judge(judge_client: LLMClient, tactical_context: str, assessment: dict) -> dict:
    import json
    return judge_client.complete(
        JUDGE_PROMPT.format(tactical_context=tactical_context,
                            llm_assessment=json.dumps(assessment, indent=2))
    )


def evaluate_llm(run_case: Callable[[dict], tuple],
                 test_cases: list = TEST_CASES,
                 judge_client: Optional[LLMClient] = None,
                 n_runs: int = 1) -> dict:
    """Evaluate the LLM layer objectively.

    Parameters
    ----------
    run_case : callable(test_case) -> (assessment_dict, tactical_context_str)
        You provide this — it runs your pipeline for one scenario. Keeping it a
        callback decouples evaluation from how scenarios are generated.
    judge_client : independent LLM client (MUST differ from the system under
        test) or None.
    n_runs : repetitions per case (>=5 recommended for meaningful consistency).
    test_cases : each case needs expected_intent/expected_threat (+ optionally
        expected_action) UNLESS it sets has_ground_truth=False, in which case no
        expected_* fields are read at all — the only thing scored is whether the
        system correctly abstained (see correct_abstention_rate).
    """
    results = []
    for case in test_cases:
        has_gt = case.get("has_ground_truth", True)
        runs = []
        for _ in range(n_runs):
            assessment, ctx = run_case(case)
            runs.append((assessment, ctx))

        intents = [a.get("likely_intent", "") for a, _ in runs]
        threats = [a.get("threat_level", "") for a, _ in runs]
        actions = [a.get("recommended_action", "") for a, _ in runs]
        abstentions = [is_abstention(i) for i in intents]
        abstention_rate = float(np.mean(abstentions))
        # Everything below is scored on the NON-abstained subset only. An abstaining
        # response isn't claiming a specific intent/threat/action, so it can't be a
        # hit, a miss, or a hallucination on those axes -- it's a third outcome,
        # tracked only via abstention_rate. See module docstring.
        scored_idx = [k for k, ab in enumerate(abstentions) if not ab]

        judge_scores = []
        if judge_client is not None:
            for assessment, ctx in runs:
                js = judge(judge_client, ctx, assessment)
                if "overall_score" in js:
                    judge_scores.append(js["overall_score"])

        result = {
            "name": case["name"],
            "has_ground_truth": has_gt,
            "n_runs": len(runs),
            "n_abstained": len(runs) - len(scored_idx),
            "abstention_rate": abstention_rate,
            "judge_overall_mean": float(np.mean(judge_scores)) if judge_scores else None,
            "majority_intent": Counter(intents).most_common(1)[0][0] if intents else None,
            "majority_threat": Counter(threats).most_common(1)[0][0] if threats else None,
            "majority_action": Counter(actions).most_common(1)[0][0] if actions else None,
        }

        if has_gt:
            intent_hits = [match_intent(intents[k], case["expected_intent"]) for k in scored_idx]
            threat_hits = [match_threat(threats[k], case["expected_threat"]) for k in scored_idx]
            # expected_action is required per TEST_CASES; if a caller passes custom
            # test cases without one, action accuracy is left unreported (None)
            # rather than silently scored against a made-up expectation. Same None
            # result either way if every run for this case abstained.
            action_hits = ([match_action(actions[k], case["expected_action"]) for k in scored_idx]
                           if "expected_action" in case else None)
            halluc = [is_hallucination(intents[k], threats[k]) for k in scored_idx]
            result.update({
                "correct_abstention_rate": None,  # n/a — this case HAS an expected answer
                "intent_accuracy": float(np.mean(intent_hits)) if intent_hits else None,
                "threat_accuracy": float(np.mean(threat_hits)) if threat_hits else None,
                "action_accuracy": float(np.mean(action_hits)) if action_hits else None,
                "hallucination_rate": float(np.mean(halluc)) if halluc else None,
            })
        else:
            # No defensible expected answer for this case -- correct behaviour IS
            # abstention. Accuracy fields don't apply (nothing to compare against).
            result.update({
                "correct_abstention_rate": abstention_rate,
                "intent_accuracy": None, "threat_accuracy": None, "action_accuracy": None,
                # hallucination_rate still computed over non-abstained runs: a
                # vocabulary-valid guess on an unanswerable input isn't what
                # is_hallucination() checks for, but garbage vocabulary still is.
                "hallucination_rate": (float(np.mean([is_hallucination(intents[k], threats[k])
                                                       for k in scored_idx])) if scored_idx else None),
            })
        results.append(result)

    def mean_or_none(key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    gt_results = [r for r in results if r["has_ground_truth"]]
    no_gt_results = [r for r in results if not r["has_ground_truth"]]
    accuracy_when_answerable = mean_or_none("intent_accuracy")
    abstention_rate_when_unanswerable = (float(np.mean([r["correct_abstention_rate"] for r in no_gt_results]))
                                         if no_gt_results else None)
    over_abstention_rate = (float(np.mean([r["abstention_rate"] for r in gt_results]))
                            if gt_results else None)
    agg = {
        # Decomposed metrics -- see module docstring. Always compute all three;
        # never pick one per cell based on which case type dominates.
        "accuracy_when_answerable": accuracy_when_answerable,
        "abstention_rate_when_unanswerable": abstention_rate_when_unanswerable,
        "over_abstention_rate": over_abstention_rate,
        "mean_threat_accuracy": mean_or_none("threat_accuracy"),
        "mean_action_accuracy": mean_or_none("action_accuracy"),
        "mean_hallucination_rate": mean_or_none("hallucination_rate"),
        "mean_abstention_rate": mean_or_none("abstention_rate"),
        # Kept as aliases of the two fields above for backward compatibility with
        # existing readers (run_degradation_eval.py, plot_degradation.py, tests).
        "mean_intent_accuracy": accuracy_when_answerable,
        "mean_correct_abstention_rate": abstention_rate_when_unanswerable,
        "n_cases": len(results), "n_cases_with_ground_truth": len(gt_results),
        "n_cases_without_ground_truth": len(no_gt_results), "n_runs": n_runs,
    }
    return {"per_case": results, "aggregate": agg}


def evaluate_ml_model(model, X_test, y_test, formation_names, train_mean,
                      train_std, reg_mean, reg_std, cfg, device=None,
                      save_path: Optional[str] = "ml_confusion_matrix.png"):
    """Classification report + confusion matrix on the test set."""
    import torch
    from sklearn.metrics import classification_report, confusion_matrix

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from ..graph import sequence_to_graphs

    X_norm = (X_test - train_mean) / train_std
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_norm)):
            graphs = sequence_to_graphs(X_norm[i], cfg.edge_threshold)
            logits, _ = model([graphs])
            preds.append(int(logits.argmax(1).item()))
    preds = np.array(preds)

    report = classification_report(y_test, preds, target_names=formation_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    if save_path:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=formation_names, yticklabels=formation_names)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                    xticklabels=formation_names, yticklabels=formation_names)
        axes[0].set_title("Confusion Matrix (counts)")
        axes[1].set_title("Confusion Matrix (normalised)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # Report macro-F1 prominently (robust to the transitioning-class imbalance).
    return {"classification_report": report, "macro_f1": report["macro avg"]["f1-score"],
            "confusion_matrix": cm.tolist()}
