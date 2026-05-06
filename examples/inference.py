"""
Raven model inference example.

Usage:
    python examples/inference.py --config configs/raven_340M_1.json
"""

import argparse
import json

import torch

import raven  # noqa: registers models with transformers
from raven.models.raven import RavenConfig, RavenForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to model config JSON')
    parser.add_argument('--prompt', type=str, default='The meaning of life is')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config) as f:
        config_dict = json.load(f)
    config = RavenConfig(**config_dict)

    # Build model (random weights for demo)
    print(f"Building Raven model: {config.num_hidden_layers}L, {config.hidden_size}D, {config.num_heads}H")
    model = RavenForCausalLM(config).to(device).eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params / 1e6:.1f}M")

    # Tokenize (use a simple tokenizer for demo)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors='pt').to(device)
    print(f"\nPrompt: {args.prompt}")

    # Forward pass sanity check
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        print(f"Loss: {outputs.loss.item():.4f}")
        print(f"Logits shape: {outputs.logits.shape}")

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {text}")


if __name__ == '__main__':
    main()
