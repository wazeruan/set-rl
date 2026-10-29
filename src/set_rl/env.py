"""SET environment.

A card is a vector in Z_3^n (default n = 4). Three cards form a set iff, for every
attribute, the values are all equal or all different, which is equivalent to
a + b + c == 0 (mod 3) componentwise.
"""

from __future__ import annotations

import itertools
import math
import random
import re

Card = tuple[int, ...]

# Generic attribute names; the env itself only uses the integer vectors.
ATTRIBUTES: list[tuple[str, tuple[str, str, str]]] = [
    ("count", ("one", "two", "three")),
    ("color", ("red", "green", "blue")),
    ("fill", ("solid", "striped", "empty")),
    ("shape", ("circle", "square", "triangle")),
    ("size", ("small", "medium", "large")),
]


def is_set(a: Card, b: Card, c: Card) -> bool:
    return all((x + y + z) % 3 == 0 for x, y, z in zip(a, b, c, strict=True))


def third_card(a: Card, b: Card) -> Card:
    """The unique card that completes a and b into a set."""
    return tuple((-(x + y)) % 3 for x, y in zip(a, b, strict=True))


def find_sets(cards: list[Card]) -> list[tuple[int, int, int]]:
    """All index triples i < j < k (0-based) that form a set."""
    return [t for t in itertools.combinations(range(len(cards)), 3) if is_set(*(cards[i] for i in t))]


def deal(rng: random.Random, n_cards: int = 12, n_attrs: int = 4, min_sets: int = 1,
         max_tries: int = 10_000) -> list[Card]:
    """Deal distinct cards, resampling until the table holds at least min_sets sets."""
    if not 1 <= n_attrs <= len(ATTRIBUTES):
        raise ValueError(f"n_attrs must be in 1..{len(ATTRIBUTES)}")
    if not 0 <= n_cards <= 3**n_attrs:
        raise ValueError(f"n_cards must be in 0..{3**n_attrs}")
    if min_sets > math.comb(n_cards, 3):
        raise ValueError("min_sets exceeds the number of triples on the table")
    deck = list(itertools.product(range(3), repeat=n_attrs))
    for _ in range(max_tries):
        cards = rng.sample(deck, n_cards)
        if len(find_sets(cards)) >= min_sets:
            return cards
    raise RuntimeError(f"no table with >= {min_sets} sets after {max_tries} tries")


def format_card(card: Card) -> str:
    return ", ".join(f"{name}={values[v]}" for (name, values), v in zip(ATTRIBUTES, card))


def build_prompt(cards: list[Card]) -> str:
    n_attrs = len(cards[0])
    if n_attrs > len(ATTRIBUTES) or any(len(c) != n_attrs or not all(v in (0, 1, 2) for v in c) for c in cards):
        raise ValueError("cards must share one length <= len(ATTRIBUTES) with values in {0, 1, 2}")
    names = ", ".join(name for name, _ in ATTRIBUTES[:n_attrs])
    lines = "\n".join(f"{i + 1}. {format_card(c)}" for i, c in enumerate(cards))
    return (
        f"Each card has {n_attrs} attributes ({names}), each with 3 possible values.\n"
        "Three cards form a SET if, for every attribute, the three values are either all the same "
        "or all different.\n\n"
        f"Cards:\n{lines}\n\n"
        "Find one SET among these cards. Think briefly, then give the three card numbers "
        "in the form <answer>i, j, k</answer>."
    )


_ANSWER_RE = re.compile(r"<answer>\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*</answer>")


def parse_answer(text: str) -> tuple[int, int, int] | None:
    """Last well-formed <answer>i, j, k</answer>, as 1-based card numbers."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return tuple(int(x) for x in matches[-1])  # type: ignore[return-value]


def reward(text: str, cards: list[Card], format_bonus: float = 0.1) -> float:
    """1.0 for a valid set; format_bonus for a well-formed but wrong answer; else 0."""
    ans = parse_answer(text)
    if ans is None:
        return 0.0
    idx = [a - 1 for a in ans]
    if len(set(idx)) != 3 or not all(0 <= i < len(cards) for i in idx):
        return format_bonus
    return 1.0 if is_set(*(cards[i] for i in idx)) else format_bonus
