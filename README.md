# set-rl

Train a small LLM with RL (GRPO + LoRA, on Apple Silicon via MLX) to find SETs.

Cards are vectors in Z_3^4; three cards form a set iff they sum to 0 mod 3
(every attribute all-same or all-different). Rewards are checked exactly.

```bash
uv sync
uv run pytest -q
```
