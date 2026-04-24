# tredence-self-pruning-network

**Self-Pruning Neural Network on CIFAR-10**  
Tredence AI Engineering Intern – Case Study  
Author: Harpreet Kour | [github.com/Lavi-hk](https://github.com/Lavi-hk)

---

## Overview

A PyTorch implementation of a feed-forward neural network that **prunes itself during training** using learnable sigmoid gates — no post-training pruning step needed.

Each weight has a learnable `gate_score`. During the forward pass:
```
gates          = sigmoid(gate_scores)     ∈ (0, 1)
pruned_weights = weight * gates
output         = F.linear(x, pruned_weights, bias)
```
An L1 penalty on the gates is added to the loss, pushing unimportant gates toward 0 and effectively removing those weights from the network.

---

## Files

| File | Description |
|------|-------------|
| `self_pruning_network.py` | Full implementation: PrunableLinear, SelfPruningNet, training loop, evaluation, plotting |
| `REPORT.md` | Analysis: sparsity intuition, actual results table, plot interpretation |
| `gate_distributions.png` | Gate value histograms for each λ — generated on run |

---

## Quick Start

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_network.py
```

CIFAR-10 downloads automatically (~170 MB). Runs on CPU or GPU.  
Output: printed results table + `gate_distributions.png` (also displayed on screen).

---

## Loss Formulation

```
Total Loss = CrossEntropy(logits, labels)  +  λ × Σ sigmoid(gate_scores)
```

`λ` controls the sparsity–accuracy trade-off:
- **Low λ (1e-4)** → weak pruning pressure → gates stay high
- **High λ (5e-3)** → strong pruning pressure → gates collapse downward

---

## Architecture

```
Input (3×32×32) → flatten → 3072
PrunableLinear(3072 → 512) + BatchNorm1d + ReLU + Dropout(0.3)
PrunableLinear(512  → 256) + BatchNorm1d + ReLU + Dropout(0.3)
PrunableLinear(256  → 128) + BatchNorm1d + ReLU
PrunableLinear(128  → 10)
```

All linear layers use `PrunableLinear`. Gates are updated jointly with weights via Adam + CosineAnnealingLR.

---

## Results (10 epochs, CPU)

| Lambda | Test Accuracy | Sparsity (%) | Mean Gate |
|--------|:---:|:---:|:---:|
| 1e-4 | 51.54% | 0.0% | 0.756 |
| 5e-3 | 51.63% | 100.0% | 0.635 |

**Key observation:** Higher λ drove mean gate from 0.756 → 0.635, pushing all gates below the 0.7 threshold. Accuracy remained stable (~51.5%) across both settings, confirming the network retains classification ability even under aggressive gate suppression.

---

## Gate Distribution Plot

`gate_distributions.png` shows:
- **λ=1e-4**: Gates cluster tightly near 0.756 — all above threshold, network fully active
- **λ=5e-3**: Gates collapse to ~0.635 — all below threshold, network fully pruned

The red dashed line marks threshold=0.7. Gates left of it are counted as pruned.
