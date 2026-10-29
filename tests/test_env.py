import itertools
import random

import pytest

from set_rl.env import build_prompt, deal, format_card, find_sets, is_set, parse_answer, reward, third_card


def test_full_deck_has_1080_sets():
    deck = list(itertools.product(range(3), repeat=4))
    assert len(find_sets(deck)) == 1080


def test_third_card_completes_any_pair():
    deck = list(itertools.product(range(3), repeat=4))
    for a, b in itertools.combinations(deck, 2):
        c = third_card(a, b)
        assert c not in (a, b) and is_set(a, b, c)
        others = [x for x in deck if x not in (a, b, c)]
        assert not any(is_set(a, b, x) for x in others)


def test_deal_is_distinct_and_has_a_set():
    rng = random.Random(0)
    for n_cards in (5, 9, 12):
        cards = deal(rng, n_cards=n_cards)
        assert len(set(cards)) == n_cards and find_sets(cards)


def test_parse_answer_takes_last_tag():
    assert parse_answer("<answer>1, 2, 3</answer> then <answer>4,5,6</answer>") == (4, 5, 6)
    assert parse_answer("cards 1 2 3") is None


def test_reward():
    cards = [(0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (0, 0, 0, 1)]
    assert reward("<answer>1, 2, 3</answer>", cards) == 1.0
    assert reward("<answer>3, 1, 2</answer>", cards) == 1.0
    assert reward("<answer>1, 2, 4</answer>", cards) == 0.1
    assert reward("<answer>1, 1, 2</answer>", cards) == 0.1
    assert reward("<answer>1, 2, 9</answer>", cards) == 0.1
    assert reward("no answer", cards) == 0.0


def test_prompt_lists_every_card():
    cards = deal(random.Random(1), n_cards=6)
    p = build_prompt(cards)
    assert all(f"{i}. {format_card(c)}" in p for i, c in enumerate(cards, 1)) and "<answer>" in p
    assert "count=one, color=red, fill=solid, shape=circle" == format_card((0, 0, 0, 0))


def test_impossible_deals_raise():
    rng = random.Random(0)
    with pytest.raises(ValueError):
        deal(rng, n_cards=2)
    with pytest.raises(ValueError):
        deal(rng, n_cards=82)
    with pytest.raises(RuntimeError):
        deal(rng, n_cards=12, min_sets=200, max_tries=50)


def test_parse_answer_ascii_digits_only():
    arabic_one = "\u0661"
    assert parse_answer(f"<answer>{arabic_one}, 2, 3</answer>") is None
