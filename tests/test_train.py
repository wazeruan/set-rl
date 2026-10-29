import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from set_rl.train import completion_logprob_loss


class CausalToyLM(nn.Module):
    """Logits at position t depend on all tokens <= t (cumulative sum of embeddings),
    so right-padding leaking into the loss would change the result."""

    def __init__(self, vocab=7):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.out = nn.Linear(8, vocab)

    def __call__(self, x):
        return self.out(mx.cumsum(self.emb(x), axis=1))


SEQS = [([1, 2, 3, 4, 5], 2), ([1, 6, 2], 2), ([3, 3, 1, 2], 1), ([4, 5, 6, 1, 2, 3], 4)]
ADV = [1.5, -0.5, 0.7, -1.2]
NORM = sum(len(s) - pl for s, pl in SEQS)


def batch(seqs, adv, pad):
    width = max(len(s) for s, _ in seqs)
    return (mx.array([s + [pad] * (width - len(s)) for s, _ in seqs]), mx.array([pl for _, pl in seqs]),
            mx.array([len(s) for s, _ in seqs]), mx.array(adv))


def manual_loss(model, seqs, adv, norm):
    total = 0.0
    for (s, pl), a in zip(seqs, adv):
        logits = model(mx.array([s[:-1]]))[0]
        for t in range(pl, len(s)):  # completion tokens s[pl:], predicted from position t-1
            row = logits[t - 1]
            total += a * (row[s[t]].item() - math.log(mx.exp(row).sum().item()))
    return -total / norm


def test_loss_counts_only_completion_tokens_and_ignores_padding():
    mx.random.seed(0)
    model = CausalToyLM()
    want = manual_loss(model, SEQS, ADV, NORM)
    for pad in (0, 3, 6):
        got = completion_logprob_loss(model, *batch(SEQS, ADV, pad), NORM).item()
        assert abs(got - want) < 1e-4


def test_microbatch_accumulation_equals_full_batch():
    mx.random.seed(1)
    model = CausalToyLM()
    grad_fn = nn.value_and_grad(model, completion_logprob_loss)
    _, full = grad_fn(model, *batch(SEQS, ADV, 0), NORM)
    acc = None
    for j in range(0, len(SEQS), 3):  # ragged micro-batches: 3 + 1
        _, g = grad_fn(model, *batch(SEQS[j : j + 3], ADV[j : j + 3], 0), NORM)
        acc = g if acc is None else tree_map(mx.add, acc, g)
    diffs = [mx.abs(a - b).max().item() for (_, a), (_, b) in zip(tree_flatten(full), tree_flatten(acc))]
    assert max(diffs) < 1e-5
