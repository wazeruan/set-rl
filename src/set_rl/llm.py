"""Model loading and batched sampling with mlx-lm."""

from __future__ import annotations

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def load_model(model_id: str = DEFAULT_MODEL, adapter_path: str | None = None):
    return load(model_id, adapter_path=adapter_path)


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def sample(model, tokenizer, prompt_ids: list[list[int]], max_tokens: int, temp: float):
    """Returns (texts, completion token ids). A trailing EOS is appended to ids of
    completions that stopped on their own, so training also learns to stop."""
    res = batch_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temp),
        return_token_ids=True,
    )
    eos = tokenizer.eos_token_id
    ids = [t + [eos] if len(t) < max_tokens else t for t in res.token_ids]
    return res.texts, ids
