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
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np

from .client import LLMClient
from .prompts import (TEST_CASES, JUDGE_PROMPT, match_intent, match_threat,
                      is_hallucination)

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
    """
    results = []
    for case in test_cases:
        runs = []
        for _ in range(n_runs):
            assessment, ctx = run_case(case)
            runs.append((assessment, ctx))

        intents = [a.get("likely_intent", "") for a, _ in runs]
        threats = [a.get("threat_level", "") for a, _ in runs]

        intent_hits = [match_intent(i, case["expected_intent"]) for i in intents]
        threat_hits = [match_threat(t, case["expected_threat"]) for t in threats]
        halluc = [is_hallucination(i, t) for i, t in zip(intents, threats)]

        judge_scores = []
        if judge_client is not None:
            for assessment, ctx in runs:
                js = judge(judge_client, ctx, assessment)
                if "overall_score" in js:
                    judge_scores.append(js["overall_score"])

        results.append({
            "name": case["name"],
            "intent_accuracy": float(np.mean(intent_hits)),
            "threat_accuracy": float(np.mean(threat_hits)),
            "hallucination_rate": float(np.mean(halluc)),
            "judge_overall_mean": float(np.mean(judge_scores)) if judge_scores else None,
            "majority_intent": Counter(intents).most_common(1)[0][0] if intents else None,
            "majority_threat": Counter(threats).most_common(1)[0][0] if threats else None,
        })

    agg = {
        "mean_intent_accuracy": float(np.mean([r["intent_accuracy"] for r in results])),
        "mean_threat_accuracy": float(np.mean([r["threat_accuracy"] for r in results])),
        "mean_hallucination_rate": float(np.mean([r["hallucination_rate"] for r in results])),
        "n_cases": len(results), "n_runs": n_runs,
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
