"""Five held-out generalization probes. Training only ever sees: find-a-set, 4 attributes,
the default vocabulary, <= 12 cards. Each probe changes exactly one of those.

  more_cards     15 cards on the table (table size beyond the training range)
  five_attrs     5 attributes, adding `size` (a dimension never trained on)
  renamed        same rule, entirely new attribute names and values (surface change)
  complete_pair  given two cards, write the third card (new task: needs the rule itself)
  math_vectors   vectors in Z_3^4, find three summing to 0 mod 3 (no cards, no SET wording)

uv run python -m set_rl.probes --n 30 [--adapter runs/x/adapter]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Callable

from set_rl.env import ATTRIBUTES, build_prompt, deal, find_sets, format_card, reward, third_card

PROBE_SEED = 777_000  # disjoint from EVAL_SEED and training seeds

RENAMED = [
    ("pitch", ("low", "middle", "high")),
    ("tempo", ("slow", "moderate", "fast")),
    ("instrument", ("flute", "violin", "drum")),
    ("volume", ("quiet", "normal", "loud")),
]


@dataclass
class Item:
    prompt: str
    score: Callable[[str], float]  # 1.0 correct, 0.0 otherwise
    solution: str  # one correct response, for testing the scorer


def _correct(text: str, cards) -> float:
    return 1.0 if reward(text, cards, format_bonus=0.0) == 1.0 else 0.0


def _find_set_item(cards, attributes=ATTRIBUTES) -> Item:
    i, j, k = find_sets(cards)[0]
    return Item(build_prompt(cards, attributes), lambda t: _correct(t, cards),
                f"<answer>{i + 1}, {j + 1}, {k + 1}</answer>")


def more_cards(rng: random.Random) -> Item:
    return _find_set_item(deal(rng, n_cards=15))


def five_attrs(rng: random.Random) -> Item:
    return _find_set_item(deal(rng, n_cards=9, n_attrs=5))


def renamed(rng: random.Random) -> Item:
    return _find_set_item(deal(rng, n_cards=8), RENAMED)


_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_KV_RE = re.compile(r"([a-z]+)\s*=\s*([a-z]+)")


def complete_pair(rng: random.Random) -> Item:
    a, b = deal(rng, n_cards=2, min_sets=0)
    target = format_card(third_card(a, b))
    want = dict(_KV_RE.findall(target))
    prompt = (
        "Each card has 4 attributes (count, color, fill, shape), each with 3 possible values:\n"
        + "\n".join(f"- {name}: {', '.join(vals)}" for name, vals in ATTRIBUTES[:4])
        + "\nThree cards form a SET if, for every attribute, the three values are either all the same "
        "or all different.\n\n"
        f"Card A: {format_card(a)}\nCard B: {format_card(b)}\n\n"
        "Which card C makes A, B, C a SET? Think briefly, then answer in the form "
        "<answer>count=..., color=..., fill=..., shape=...</answer>."
    )

    def score(text: str) -> float:
        tags = _TAG_RE.findall(text)
        if not tags:
            return 0.0
        pairs = _KV_RE.findall(tags[-1].lower())
        # Repeated keys (e.g. count=one, count=two) are contradictory, so they score 0.
        return 1.0 if len(pairs) == len(want) and dict(pairs) == want else 0.0

    return Item(prompt, score, f"<answer>{target}</answer>")


def math_vectors(rng: random.Random) -> Item:
    vecs = deal(rng, n_cards=8)
    lines = "\n".join(f"v{i + 1} = ({', '.join(map(str, v))})" for i, v in enumerate(vecs))
    prompt = (
        "Work in arithmetic modulo 3. Here are 8 distinct vectors with 4 coordinates each:\n"
        f"{lines}\n\n"
        "Find three different vectors whose sum is (0, 0, 0, 0) modulo 3. Think briefly, then give "
        "their indices in the form <answer>i, j, k</answer>."
    )
    i, j, k = find_sets(vecs)[0]
    return Item(prompt, lambda t: _correct(t, vecs), f"<answer>{i + 1}, {j + 1}, {k + 1}</answer>")


PROBES: dict[str, Callable[[random.Random], Item]] = {
    "more_cards": more_cards,
    "five_attrs": five_attrs,
    "renamed": renamed,
    "complete_pair": complete_pair,
    "math_vectors": math_vectors,
}


def build_probes(n: int) -> dict[str, list[Item]]:
    out = {}
    for p_idx, (name, make) in enumerate(PROBES.items()):
        rng = random.Random(PROBE_SEED + p_idx)
        out[name] = [make(rng) for _ in range(n)]
    return out


def run_probes(model, tokenizer, n: int = 30, max_tokens: int = 512, batch: int = 16) -> dict[str, float]:
    from set_rl.llm import encode_prompt, sample

    results = {}
    for name, items in build_probes(n).items():
        texts: list[str] = []
        for i in range(0, n, batch):
            chunk = items[i : i + batch]
            t, _ = sample(model, tokenizer, [encode_prompt(tokenizer, it.prompt) for it in chunk],
                          max_tokens, temp=0.0)
            texts += t
        results[name] = sum(it.score(t) for it, t in zip(items, texts)) / n
    return results


def main():
    from set_rl.llm import DEFAULT_MODEL, load_model

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    model, tok = load_model(a.model, a.adapter)
    res = run_probes(model, tok, a.n, a.max_tokens)
    print(json.dumps({"model": a.model, "adapter": a.adapter, "probes": res}, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
