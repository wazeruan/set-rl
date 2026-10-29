from set_rl.env import ATTRIBUTES
from set_rl.probes import PROBES, build_probes


def test_five_probes_with_working_scorers():
    probes = build_probes(10)
    assert list(probes) == list(PROBES) and len(probes) == 5
    for name, items in probes.items():
        for it in items:
            assert it.score(it.solution) == 1.0, name
            assert it.score("no answer") == 0.0, name


def test_probes_are_deterministic():
    a, b = build_probes(3), build_probes(3)
    assert all(x.prompt == y.prompt for k in a for x, y in zip(a[k], b[k]))


def test_probes_differ_from_training_distribution():
    probes = build_probes(5)
    assert all(it.prompt.count("\n15. ") == 1 for it in probes["more_cards"])
    assert all("size=" in it.prompt for it in probes["five_attrs"])
    default_values = {v for _, vals in ATTRIBUTES for v in vals}
    for it in probes["renamed"]:
        assert "pitch=" in it.prompt and not any(f"={v}" in it.prompt for v in default_values)
    assert all("SET" not in it.prompt and "card" not in it.prompt.lower() for it in probes["math_vectors"])


def test_complete_pair_rejects_wrong_card():
    for it in build_probes(10)["complete_pair"]:
        count = next(v for v in ("one", "two", "three") if f"count={v}" in it.solution)
        other = "two" if count == "one" else "one"
        assert it.score(it.solution.replace(f"count={count}", f"count={other}")) == 0.0
        loud = "<answer>" + it.solution[8:-9].upper().replace(", ", " ,  ") + "</answer>"
        assert it.score(loud) == 1.0  # case and spacing are tolerated
        hedged = it.solution.replace(f"count={count}", f"count={other}, count={count}")
        assert it.score(hedged) == 0.0  # contradictory duplicate keys
