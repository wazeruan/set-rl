"""Evaluate a model (optionally with a LoRA adapter) on fixed SET tables.

uv run python -m set_rl.evaluate --n 50 --cards 5 8 12 [--adapter runs/x/adapter]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

from set_rl.env import build_prompt, deal, find_sets, parse_answer, reward
from set_rl.llm import DEFAULT_MODEL, encode_prompt, load_model, sample

EVAL_SEED = 12345  # training uses other seeds, so eval tables stay held out


def eval_tables(n: int, n_cards: int, n_attrs: int = 4) -> list[list[tuple[int, ...]]]:
    rng = random.Random(EVAL_SEED + 1000 * n_cards + n_attrs)
    return [deal(rng, n_cards=n_cards, n_attrs=n_attrs) for _ in range(n)]


def evaluate(model, tokenizer, n: int, card_counts: list[int], n_attrs: int = 4,
             max_tokens: int = 512, temp: float = 0.0, batch: int = 16, show: int = 0) -> dict:
    results = {}
    for n_cards in card_counts:
        tables = eval_tables(n, n_cards, n_attrs)
        texts: list[str] = []
        for i in range(0, n, batch):
            chunk = tables[i : i + batch]
            t, _ = sample(model, tokenizer, [encode_prompt(tokenizer, build_prompt(c)) for c in chunk],
                          max_tokens, temp)
            texts += t
        rewards = [reward(t, c) for t, c in zip(texts, tables)]
        chance = sum(len(find_sets(c)) / math.comb(n_cards, 3) for c in tables) / n
        results[n_cards] = {
            "accuracy": sum(r == 1.0 for r in rewards) / n,
            "well_formed": sum(parse_answer(t) is not None for t in texts) / n,
            "random_guess": chance,
            "mean_len_chars": sum(map(len, texts)) / n,
        }
        for t in texts[:show]:
            print(f"--- sample ({n_cards} cards) ---\n{t}\n")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--cards", type=int, nargs="+", default=[5, 8, 12])
    p.add_argument("--attrs", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = load_model(a.model, a.adapter)
    t0 = time.time()
    res = evaluate(model, tok, a.n, a.cards, a.attrs, a.max_tokens, show=a.show)
    print(json.dumps({"model": a.model, "adapter": a.adapter, "seconds": round(time.time() - t0),
                      "results": res}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
