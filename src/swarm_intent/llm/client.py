"""
Provider-agnostic LLM client.

Replaces the copy-pasted ``call_llm`` in every notebook with a single class that
can talk to:
  - Groq (hosted, for the prompt-engineered baseline)
  - a local HuggingFace model (your QLoRA-fine-tuned Qwen/Mistral)

This lets the rest of the codebase (pipeline, eval) stay identical whether you
are using the API baseline or your fine-tuned model — just swap the client.

Robustness fixes over the original: HTTP error checking, retry with backoff,
and tolerant JSON extraction.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, fences or not."""
    # Reasoning models (qwen3, deepseek-r1) prepend a <think>...</think> block;
    # drop it first so a stray brace inside the thinking prose can't poison the
    # greedy {.*} fallback search below.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class LLMClient:
    """Base interface. Implement ``generate(prompt) -> str``."""

    def complete(self, prompt: str) -> dict:
        for attempt in range(3):
            try:
                return _extract_json(self.generate(prompt))
            except json.JSONDecodeError:
                if attempt == 2:
                    return {"error": "JSON parse failed"}
            except Exception as e:  # network etc.
                if attempt == 2:
                    return {"error": str(e)}
                time.sleep(2 ** attempt)
        return {"error": "unreachable"}

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile",
                 api_key: Optional[str] = None, temperature: float = 0.3,
                 max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Set GROQ_API_KEY (env var) or pass api_key=...")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model,
                  "messages": [{"role": "user",
                                "content": f"Return ONLY valid JSON.\n\n{prompt}"}],
                  "temperature": self.temperature, "max_tokens": self.max_tokens},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class LocalHFClient(LLMClient):
    """Wraps a local (optionally QLoRA-fine-tuned) HuggingFace causal LM.

    Lazy-imports transformers so the rest of the package works without it.
    """

    def __init__(self, model_path: str, adapter_path: Optional[str] = None,
                 temperature: float = 0.3, max_new_tokens: int = 512, load_4bit: bool = True,
                 system_prompt: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.tok = AutoTokenizer.from_pretrained(model_path)
        quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)
                 if load_4bit else None)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant, device_map="auto",
            torch_dtype=torch.float16,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch
        messages = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else [])
        messages.append({"role": "user", "content": f"Return ONLY valid JSON.\n\n{prompt}"})
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(self.model.device)
        sample_kwargs = ({"do_sample": True, "temperature": self.temperature}
                         if self.temperature > 0 else {"do_sample": False})
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **sample_kwargs)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
