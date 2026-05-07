<div align="center">

<img src="assets/img/raven_walnut.png" width="300" alt="Raven mascot"/>

# Raven

**Raven** is a linear-time sequence model built on top of [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention).
It introduces a **routing memory mechanism** that selectively writes to a fixed set of persistent memory slots using a learned sparse router — achieving sub-quadratic complexity while maintaining strong associative recall.

</div>

---

<div align="center">

<a href="https://arshiaafzal.github.io/SSM_Story/assets/html/raven_recurrent_matrix_update.html">
<img src="https://arshiaafzal.github.io/SSM_Story/assets/video/best.gif" width="720" alt="Sparse Memory Routing in Raven"/>
</a>

<p><strong>Sparse Memory Routing in Raven.</strong> Unlike SSMs that update the entire state densely, or SWA that enforces strict FIFO overwriting, Raven uses an input-dependent router. At each step, only a specific subset of memory slots (highlighted) is selected to undergo decay and receive new information. Unselected slots remain completely untouched, preventing interference and preserving long-range recall.</p>

</div>

---

## Architecture

<div align="center">
  <img src="assets/img/arch.png" width="560" alt="Raven vs SSM architecture"/>
</div>

Raven replaces the SSM block with an **RSM (Routing State Model)** layer. Unlike GLA/Mamba2 which write to all memory slots uniformly, Raven learns a per-token sparse router `R` that selects which slots to update.

---

## How Raven Works

<div align="center">
  <img src="assets/img/mem.png" width="680" alt="Routing memory mechanism"/>
</div>

Each Raven layer maintains a matrix memory state `H ∈ R^(slots × d_v)`. At each timestep the router selects the top-k most relevant slots and performs a gated update:

```
route_scores = TopK( sigmoid(r_proj(x)) )
decay         = exp( route_scores * f )     # sparse forgetting gate
H             = H * decay + (1 - decay) * k ⊗ v
o             = q · H                       # read
```

The table below places Raven in the broader landscape of linear models:

<div align="center">
  <img src="assets/img/tab.png" width="680" alt="Unified view of linear models"/>
</div>

Key design choices:
- **Sparse top-k routing** — each token writes to a small subset of memory slots
- **Gumbel noise** during training for exploration (optional)
- **Mamba2 or GLA decay** for the forgetting gate
- **RoPE** on Q/K (optional)
- **Chunked Triton kernels** for training, fused recurrent kernels for generation

---

## Results

### In-Context Recall Benchmarks

<div align="center">
  <img src="assets/img/recall.png" width="700" alt="Recall and benchmark results"/>
</div>

**Table 2: In-context recall benchmarks and NIAH accuracy vs. context length.** Accuracy (%) on SWDE/FDA/SQuAD and single NIAH-1/2/3 across context lengths. **Bold** = best, <u>underline</u> = second best.

<details>
<summary>~400M parameter models</summary>



| Model | Params | Mem (M) | SWDE | FDA | SQuAD | N1-1K | N1-2K | N1-4K | N1-8K | N1-16K | N1-32K | N2-1K | N2-2K | N2-4K | N2-8K | N2-16K | N2-32K | N3-1K | N3-2K | N3-4K | N3-8K | N3-16K | N3-32K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ***Transformer*** | | | | | | | | | | | | | | | | | | | | | | | |
| *w. RoPE* | 340 | ∞ / 0 | <u>42.3</u> | <u>34.5</u> | <u>22.1</u> | **100** | **100** | 0 | 0 | 0 | 0 | **100** | **100** | 0 | 0 | 0 | 0 | <u>71.6</u> | <u>47.6</u> | 0 | 0 | 0 | 0 |
| *w. Gate (FoX)* | 376 | ∞ / 0 | **52.5** | **64.3** | **30.1** | **100** | **100** | **32.2** | **8.0** | **4.2** | 0 | **100** | **100** | **100** | **24.0** | **11.6** | **3.2** | **95.4** | **85.6** | **64.2** | **11.6** | **7.2** | 0 |
| ***SSM*** | | | | | | | | | | | | | | | | | | | | | | | |
| GLA | 475 | 12.5 / 0.4 | 29.0 | 11.4 | 30.3 | 74.6 | 25.1 | 8.2 | 2.2 | 0 | 0 | 91.2 | 37.2 | 21.4 | 3.6 | 0 | 0 | <u>84.2</u> | <u>57.1</u> | <u>20.8</u> | **10.2** | <u>2.3</u> | 0 |
| GSA | 399 | 12.5 / **0** | 23.8 | 14.5 | 24.9 | <u>99.2</u> | <u>97.1</u> | <u>90.0</u> | 67.4 | 29.6 | 11.0 | 96.6 | **98.8** | 28.0 | 5.1 | 1.0 | 0 | 60.0 | 30.1 | 13.5 | 1.0 | 0 | 0 |
| GDN | 475 | 12.5 / 0.4 | <u>29.5</u> | 8.3 | 31.3 | <u>99.2</u> | **100** | **99.8** | <u>92.0</u> | <u>41.8</u> | <u>22.1</u> | <u>99.2</u> | 92.0 | 43.6 | <u>17.8</u> | <u>6.2</u> | <u>4.0</u> | **92.6** | **80.6** | **37.8** | <u>5.2</u> | **6.8** | <u>2.5</u> |
| Mamba-2 | 382 | 12.5 / 0.4 | 25.7 | <u>14.9</u> | <u>31.9</u> | <u>99.2</u> | 95.6 | 52.2 | 12.8 | 5.4 | 2.8 | **99.8** | <u>98.0</u> | <u>68.2</u> | 15.4 | 4.4 | 3.8 | 53.4 | 53.6 | 17.4 | 1.8 | 2.2 | **3.2** |
| SWA | 374 | 12.5 / **0** | 10.0 | 14.4 | 29.7 | 29.8 | 11.0 | 6.2 | 3.4 | 1.2 | 0 | 36.2 | 14.4 | 10.2 | 3.8 | 3.2 | 0 | 26.2 | 9.2 | 7.4 | 1.4 | 1.8 | 0 |
| **Raven** | **424** | **12.5 / 0** | **34.1** | **22.7** | **35.4** | **99.8** | **100** | **99.8** | **99.8** | **99.4** | **91.4** | 98.8 | <u>98.0</u> | **98.8** | **81.6** | **23.0** | **8.8** | 76.8 | 43.6 | 13.4 | 1.0 | 0 | 0 |

</details>

### Language Modeling & Zero-Shot Evaluation

**Table 3: Language modeling and zero-shot evaluation results.** Perplexity on Lambada (LMB.) and zero-shot accuracy across standard benchmarks. **Bold** = best, <u>underline</u> = second best.

<details>
<summary>~400M parameter models</summary>

| Model | Params | LMB. ppl↓ | LMB. acc↑ | PIQA↑ | Hella.↑ | Wino.↑ | ARC-e↑ | ARC-c↑ | Avg.↑ |
|---|---|---|---|---|---|---|---|---|---|
| ***Transformer*** | | | | | | | | | |
| *w. RoPE* | 340 | 42.0 | 31.0 | 64.4 | 30.2 | 51.0 | 44.3 | 18.7 | 39.9 |
| *w. Gate (FoX)* | 376 | 48.1 | 30.6 | 64.9 | 30.7 | 51.1 | 44.7 | 18.9 | 40.1 |
| ***SSM*** | | | | | | | | | |
| GLA | 400 | 42.1 | 30.7 | 64.4 | 30.1 | 52.7 | 43.8 | 19.6 | 40.2 |
| GSA | 399 | 44.1 | 30.3 | 64.9 | 30.7 | 51.5 | 45.6 | 20.5 | <u>40.5</u> |
| GDN | 475 | 40.1 | 31.6 | 65.6 | 31.4 | 50.2 | 45.7 | 19.3 | **40.6** |
| Mamba-2 | 382 | 43.0 | 29.9 | 65.0 | 31.5 | 51.2 | 47.5 | 20.5 | 40.1 |
| SWA | 374 | 40.7 | 30.5 | 64.5 | 30.4 | 51.6 | 44.9 | 18.6 | 40.0 |
| **Raven** | **424** | 41.0 | **32.7** | 64.1 | 30.3 | 51.7 | 43.9 | 18.4 | 40.2 |

</details>

### Hybrid Models Retrieval Ability

**Table 4: Hybrid-Raven vs. other hybrid architectures on retrieval tasks.** ✓ = no convolutional memory needed.

<details>
<summary>~400M parameter models</summary>

| Model | No Conv. | SWDE | FDA | SQuAD | N1-1K | N1-2K | N1-4K | N1-8K | N1-16K | N1-32K | N2-1K | N2-2K | N2-4K | N2-8K | N2-16K | N2-32K | N3-1K | N3-2K | N3-4K | N3-8K | N3-16K | N3-32K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GDN | ✗ | <u>54.6</u> | 67.2 | <u>34.5</u> | **100** | **100** | **100** | **100** | 93.2 | 70.5 | **100** | **100** | **100** | 8.0 | 0 | 0 | 93.2 | 70.2 | 50.0 | 0 | 0 | 0 |
| Mamba-2 | ✗ | **56.3** | **68.8** | **36.0** | **100** | **100** | 16.4 | 0 | 0 | 0 | **100** | **100** | 85.8 | 0 | 0 | 0 | 76.9 | **80.6** | 60.8 | 0 | 0 | 0 |
| SWA-RoPE | ✓ | 51.0 | <u>68.1</u> | 34.1 | **100** | **100** | **100** | **100** | 98.2 | 60.4 | **100** | **100** | **100** | 98.2 | 3.1 | 0 | **93.4** | 78.2 | 12.8 | **60.0** | 4.4 | 0 |
| **Raven** | **✓** | 51.4 | 64.2 | 31.4 | **100** | **100** | **100** | **100** | 98.4 | **78.6** | **100** | **100** | **100** | **100** | **95.4** | **65.4** | 90.0 | 67.0 | **73.8** | **60.0** | **10.2** | **14.4** |

</details>

---

## Model Structure

| Component | Details |
|-----------|---------|
| Layer type | `RavenAttention` (replaces standard attention) |
| Memory | Fixed-size slot matrix per head (`num_slots` slots) |
| Router | Linear or MLP projection → top-k sigmoid/softmax |
| Decay | Mamba2 (`A_log` + `dt_bias`) or GLA (`logsigmoid`) |
| Feature map | Swish, ReLU, or T2R |
| Computation | Chunked (training) / Fused recurrent (inference) |
| Hybrid layers | Optional standard attention layers at specified indices |

---

## Installation

Install the FLA dependency first, following the [official FLA guide](https://github.com/fla-org/flash-linear-attention):

```sh
pip install flash-linear-attention
```

Then clone this repo:

```sh
git clone https://github.com/AvivBick/RoutingMemory
cd RoutingMemory
pip install -e .
```

Requirements: PyTorch ≥ 2.5, Triton ≥ 3.0, einops, transformers ≥ 4.45.0

---

## Usage

### As a layer

```python
from fla.layers import RavenAttention

attn = RavenAttention(
    hidden_size=1024,
    num_heads=4,
    num_slots=256,
    topk=32,
    decay_type='Mamba2',    # or 'GLA'
    feature_map='swish',
    router_type='lin',      # or 'mlp'
    router_score='sigmoid', # or 'softmax'
).cuda()

x = torch.randn(1, 2048, 1024).cuda()
y, _, _ = attn(x)  # (batch, seq_len, hidden_size)
```

### As a full causal LM

```python
from fla.models import RavenConfig, RavenForCausalLM
from transformers import AutoModelForCausalLM

config = RavenConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_heads=4,
    num_slots=256,
    topk=32,
    decay_type='Mamba2',
    feature_map='swish',
    vocab_size=32000,
)
model = AutoModelForCausalLM.from_config(config).cuda()
```

---

## Training

Raven uses the [flame](https://github.com/fla-org/flame) training framework. Add a config from `configs/raven_340M_*.json` to `flame/configs/`, then:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 bash train.sh \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/raven-340M \
    --model.config configs/raven_340M_1.json \
    --optimizer.name AdamW \
    --optimizer.lr 3e-4 \
    --lr_scheduler.warmup_steps 1024 \
    --lr_scheduler.decay_type cosine \
    --training.batch_size 16 \
    --training.seq_len 2048 \
    --training.gradient_accumulation_steps 4 \
    --training.steps 30720 \
    --training.dataset /path/to/SlimPajama-627B \
    --training.streaming \
    --training.compile \
    --checkpoint.interval 3072
```

The `configs/` directory contains 12 ablation configurations varying the router design (linear vs. MLP, sigmoid vs. softmax, with/without Gumbel noise and bias).

---

## Repository Structure

```
fla/
├── layers/
│   └── raven.py              # RavenAttention layer
├── models/
│   └── raven/
│       ├── configuration_raven.py
│       ├── modeling_raven.py
│       └── lg.py               # perplexity evaluation
└── ops/
    └── raven/
        ├── chunk.py            # chunked Triton kernel (training)
        ├── fused_recurrent.py  # fused recurrent kernel (inference)
        └── naive.py            # reference implementation
configs/
└── raven_340M_*.json         # 12 ablation configs (340M scale)
assets/img/                     # figures used in this README
```

---

## Upstream: Flash Linear Attention

This repo is a fork of [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), which provides hardware-efficient Triton-based implementations of many linear attention models. All upstream models (GLA, GSA, DeltaNet, Mamba2, RWKV7, etc.) remain available.

[![hf_model](https://img.shields.io/badge/-Models-gray.svg?logo=huggingface&style=flat-square)](https://huggingface.co/fla-hub) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/vDaJTmKNcS)

-----

## Citation


```

@article{afzal2025raven,
  title={Raven: High-Recall Sequence Modeling with Sparse Memory Routing},
  author={Arshia Afzal, Aviv Bick, Eric P. Xing, Volkan Cevher, Albert Gu},
  year={2025},
  publisher={MDPI}
}

```





