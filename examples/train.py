"""
Raven model training sanity check.

Quick training loop on random data to verify forward/backward passes work.

Usage:
    python examples/train.py --config configs/raven_340M_1.json --steps 10
"""

import argparse
import json
import time

import torch
import torch.nn as nn

import raven  # noqa: registers models with transformers
from raven.models.raven import RavenConfig, RavenForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--steps', type=int, default=10, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config) as f:
        config_dict = json.load(f)
    config = RavenConfig(**config_dict)

    # Build model
    print(f"Building Raven model: {config.num_hidden_layers}L, {config.hidden_size}D, {config.num_heads}H")
    model = RavenForCausalLM(config).to(device).to(torch.bfloat16)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"\nRunning {args.steps} training steps (batch_size={args.batch_size}, seq_len={args.seq_len})...\n")

    for step in range(1, args.steps + 1):
        # Random data
        input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        labels = input_ids.clone()

        t0 = time.time()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        dt = time.time() - t0

        tokens_per_sec = args.batch_size * args.seq_len / dt
        print(f"Step {step:3d} | Loss: {loss.item():.4f} | Grad norm: {grad_norm:.4f} | "
              f"{tokens_per_sec:.0f} tok/s | {dt:.2f}s")

    print("\nTraining sanity check passed!")


if __name__ == '__main__':
    main()
