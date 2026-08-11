"""
Tests for llm_finetuning/eval_sft_v5.py's real (non-mock) plumbing:
checkpoint-path resolution against a fake directory tree (no real v5-a
checkpoint exists yet), the zero-GPU meta-device load path, and the
judge-is-advisory-only contract (headline metrics identical with/without a
judge). Built during the "post-training prep" session (V5_STATE.json) while
v5-a training was running in a separate tmux session -- no GPU touched here.

Usage:
    python -m unittest tests.test_eval_sft_v5 -v
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llm_finetuning"))

from eval_sft_v5 import (  # noqa: E402
    resolve_adapter_path, load_model_dry_run, evaluate, mock_cases, mock_predict,
)


class FakeJudgeClient:
    """Deterministic stand-in for a real NvidiaClient judge -- no network
    call, no NVIDIA_API_KEY needed. Always returns a fixed score, so tests
    can assert judge_scores/judge_overall_mean are populated without touching
    the network, while every OTHER field must stay identical to the
    judge_client=None run."""
    def complete(self, prompt: str) -> dict:
        return {"overall_score": 4}


class TestResolveAdapterPath(unittest.TestCase):
    def test_direct_adapter_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "adapter_config.json"), "w") as f:
                f.write("{}")
            self.assertEqual(resolve_adapter_path(d), d)

    def test_picks_highest_step_numerically_not_lexicographically(self):
        """checkpoint-1500 must be picked over checkpoint-500 -- a plain
        string sort would get this backwards ('1' < '5')."""
        with tempfile.TemporaryDirectory() as d:
            for step in (500, 1000, 1500):
                sub = os.path.join(d, f"checkpoint-{step}")
                os.makedirs(sub)
                with open(os.path.join(sub, "adapter_config.json"), "w") as f:
                    f.write("{}")
            resolved = resolve_adapter_path(d)
            self.assertEqual(resolved, os.path.join(d, "checkpoint-1500"))

    def test_ignores_checkpoint_dir_missing_adapter_config(self):
        """A checkpoint-<N> dir that doesn't actually have adapter_config.json
        yet (e.g. save still in progress) must not be picked."""
        with tempfile.TemporaryDirectory() as d:
            complete = os.path.join(d, "checkpoint-500")
            os.makedirs(complete)
            with open(os.path.join(complete, "adapter_config.json"), "w") as f:
                f.write("{}")
            incomplete = os.path.join(d, "checkpoint-1000")
            os.makedirs(incomplete)  # no adapter_config.json inside
            resolved = resolve_adapter_path(d)
            self.assertEqual(resolved, complete)

    def test_no_checkpoint_yet_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_adapter_path(d)
            self.assertIn("checkpoint", str(ctx.exception))

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_adapter_path("/tmp/definitely_does_not_exist_v5_test_xyz")


class TestLoadModelDryRun(unittest.TestCase):
    def test_zero_gpu_memory_and_returns_usable_objects(self):
        import torch
        pre = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        tok, model = load_model_dry_run("Qwen/Qwen2.5-7B-Instruct", "fake/path")
        post = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.assertEqual(pre, 0)
        self.assertEqual(post, 0)
        self.assertIsNotNone(tok)
        self.assertIsNotNone(model)
        # confirm the LoRA config actually attached (trainable params > 0),
        # matching train_sft_v5.py's own dry-run assertion
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(trainable, 0)


class TestJudgeAdvisoryOnly(unittest.TestCase):
    def test_headline_metrics_identical_with_and_without_judge(self):
        cases = mock_cases()
        res_no_judge = evaluate(cases, mock_predict, judge_client=None)
        res_with_judge = evaluate(cases, mock_predict, judge_client=FakeJudgeClient())

        no_judge_stripped = copy.deepcopy(res_no_judge)
        with_judge_stripped = copy.deepcopy(res_with_judge)
        for r in (no_judge_stripped, with_judge_stripped):
            r.pop("judge_scores", None)
            r.pop("judge_overall_mean", None)

        self.assertEqual(no_judge_stripped, with_judge_stripped,
                         "headline metrics differ depending on judge presence -- "
                         "the advisory-only contract is broken")

    def test_judge_fields_actually_differ_when_judge_present(self):
        """The flip side of the test above -- if judge_scores/judge_overall_mean
        were ALSO always empty, the test above would trivially pass without
        proving the judge was ever really exercised."""
        cases = mock_cases()
        res_no_judge = evaluate(cases, mock_predict, judge_client=None)
        res_with_judge = evaluate(cases, mock_predict, judge_client=FakeJudgeClient())

        self.assertEqual(res_no_judge["judge_scores"], {})
        self.assertIsNone(res_no_judge["judge_overall_mean"])
        self.assertGreater(len(res_with_judge["judge_scores"]), 0)
        self.assertIsNotNone(res_with_judge["judge_overall_mean"])

    def test_judge_exception_does_not_affect_headline_metrics(self):
        class RaisingJudgeClient:
            def complete(self, prompt):
                raise RuntimeError("simulated judge failure")

        cases = mock_cases()
        res_no_judge = evaluate(cases, mock_predict, judge_client=None)
        res_raising_judge = evaluate(cases, mock_predict, judge_client=RaisingJudgeClient())

        no_judge_stripped = copy.deepcopy(res_no_judge)
        raising_stripped = copy.deepcopy(res_raising_judge)
        for r in (no_judge_stripped, raising_stripped):
            r.pop("judge_scores", None)
            r.pop("judge_overall_mean", None)
        self.assertEqual(no_judge_stripped, raising_stripped)
        self.assertEqual(res_raising_judge["judge_scores"], {})


if __name__ == "__main__":
    unittest.main()
