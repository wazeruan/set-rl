"""GRPO with LoRA on MLX for SET.

Each step deals `--batch` tables, samples `--group` completions per table, scores them
with the exact checker, and normalizes rewards within each group into advantages.
One gradient step per batch (on-policy), so the PPO ratio is 1 and no clipping or
KL term is needed. Card count grows as a curriculum once accuracy is high enough.

uv run python -m set_rl.train --steps 200 --out runs/first
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import linear_to_lora_layers

from set_rl.env import build_prompt, deal, parse_answer, reward
from set_rl.evaluate import evaluate
from set_rl.llm import DEFAULT_MODEL, encode_prompt, load_model, sample


def completion_logprob_loss(model, tokens, prompt_lens, total_lens, adv, norm):
    """-sum(adv * log p(completion tokens)) / norm over a padded micro-batch."""
    logits = model(tokens[:, :-1]).astype(mx.float32)
    targets = tokens[:, 1:]
    logp = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1) - mx.logsumexp(logits, axis=-1)
    pos = mx.arange(targets.shape[1])[None, :] + 1  # position of each target token
    mask = (pos >= prompt_lens[:, None]) & (pos < total_lens[:, None])
    return -(adv[:, None] * logp * mask).sum() / norm


def save_adapter(model, out: Path, num_layers: int, lora: dict):
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="runs/grpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=8, help="tables per step")
    p.add_argument("--group", type=int, default=8, help="completions per table")
    p.add_argument("--micro", type=int, default=4, help="sequences per backward pass")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--lora-layers", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--min-cards", type=int, default=4)
    p.add_argument("--max-cards", type=int, default=12)
    p.add_argument("--promote-at", type=float, default=0.7, help="accuracy that adds a card")
    p.add_argument("--attrs", type=int, default=4)
    p.add_argument("--format-bonus", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    log = open(out / "log.jsonl", "a")

    mx.random.seed(a.seed)
    rng = random.Random(a.seed)
    model, tok = load_model(a.model)
    model.freeze()
    lora = {"rank": a.rank, "scale": 20.0, "dropout": 0.0}
    linear_to_lora_layers(model, a.lora_layers, lora)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.2f}M")

    opt = optim.Adam(learning_rate=a.lr)
    loss_and_grad = nn.value_and_grad(model, completion_logprob_loss)
    n_cards, recent = a.min_cards, deque(maxlen=10)

    for step in range(1, a.steps + 1):
        t0 = time.time()
        tables = [deal(rng, n_cards=n_cards, n_attrs=a.attrs) for _ in range(a.batch)]
        prompts = [encode_prompt(tok, build_prompt(c)) for c in tables]
        texts, comp = sample(model, tok, [pr for pr in prompts for _ in range(a.group)], a.max_tokens, a.temp)

        rewards = mx.array([reward(t, tables[i // a.group], a.format_bonus) for i, t in enumerate(texts)])
        r = rewards.reshape(a.batch, a.group)
        adv = ((r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, keepdims=True) + 1e-4)).reshape(-1)

        # Only sequences with nonzero advantage carry gradient.
        seqs = [(prompts[i // a.group] + comp[i], len(prompts[i // a.group]), adv[i].item())
                for i in range(len(texts)) if abs(adv[i].item()) > 1e-6]
        loss_val = 0.0
        if seqs:
            norm = sum(len(s) - pl for s, pl, _ in seqs)
            grads = None
            for j in range(0, len(seqs), a.micro):
                mb = seqs[j : j + a.micro]
                width = max(len(s) for s, _, _ in mb)
                tokens = mx.array([s + [tok.eos_token_id] * (width - len(s)) for s, _, _ in mb])
                loss, g = loss_and_grad(model, tokens, mx.array([pl for _, pl, _ in mb]),
                                        mx.array([len(s) for s, _, _ in mb]),
                                        mx.array([ad for _, _, ad in mb]), norm)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                mx.eval(grads)
                loss_val += loss.item()
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)

        acc = (rewards == 1.0).mean().item()
        recent.append(acc)
        rec = {"step": step, "n_cards": n_cards, "reward": rewards.mean().item(), "accuracy": acc,
               "well_formed": sum(parse_answer(t) is not None for t in texts) / len(texts),
               "mean_tokens": sum(map(len, comp)) / len(comp), "trained_seqs": len(seqs),
               "loss": loss_val, "seconds": round(time.time() - t0, 1)}
        if len(recent) == recent.maxlen and sum(recent) / len(recent) >= a.promote_at and n_cards < a.max_cards:
            n_cards += 1
            recent.clear()
            rec["promoted_to"] = n_cards
        if step % a.eval_every == 0 or step == a.steps:
            rec["eval"] = evaluate(model, tok, a.eval_n, [5, 8, 12], a.attrs, a.max_tokens)
            save_adapter(model, out / "adapter", a.lora_layers, lora)
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    save_adapter(model, out / "adapter", a.lora_layers, lora)
    log.close()


if __name__ == "__main__":
    main()
