from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LocalTransformersLLM:
    tokenizer: Any
    model: Any
    temperature: float = 0.6
    top_p: float = 0.9

    def generate(self, prompt: str, max_new_tokens: int = 360) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        ).strip()


@dataclass
class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat client for aihubmix or similar gateways."""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2

    def generate(self, prompt: str, max_new_tokens: int = 360) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You answer strictly from retrieved evidence."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": max_new_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def build_api_llm_from_env() -> OpenAICompatibleLLM:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIHUBMIX_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AIHUBMIX_BASE_URL")
    model = os.getenv("OPENAI_MODEL") or os.getenv("AIHUBMIX_MODEL") or "gpt-4o-mini"
    if not api_key or not base_url:
        raise RuntimeError("API LLM backend requires OPENAI_API_KEY/AIHUBMIX_API_KEY and OPENAI_BASE_URL/AIHUBMIX_BASE_URL.")
    return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)

