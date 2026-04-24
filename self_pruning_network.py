"""
Self-Pruning Neural Network on CIFAR-10
Tredence AI Engineering Intern – Case Study
Author: Harpreet Kour | github.com/Lavi-hk
CGPA: 9.10/10

KEY FIX over previous version:
  - gate_scores init = +3.0  →  sigmoid(3) ≈ 0.95  (gates start near 1, λ drives them to 0)
  - threshold = 0.05 (more realistic than 0.01 for 20 epochs)
  - 3 lambdas: 1e-4, 1e-3, 5e-3  →  clear low/medium/high sparsity trade-off
  - 20 epochs per experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with a learnable gate per weight element.

    Forward pass:
        gates          = sigmoid(gate_scores)        in (0, 1)
        pruned_weights = weight * gates              element-wise
        output         = F.linear(x, pruned_weights, bias)

    Gradients flow through both `weight` and `gate_scores` because
    sigmoid is differentiable everywhere and * is differentiable.

    With L1 regularisation on gates (sum of gate values), the optimiser
    is penalised for keeping gates active → gates collapse to 0 for
    unimportant weights.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)
        # Init gate_scores = +3.0 so sigmoid(gate_scores) ≈ 0.95 at start.
        # All weights begin active; λ penalty then pushes unimportant gates → 0.
        nn.init.constant_(self.gate_scores, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return detached gate values for analysis."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity(self, threshold: float = 0.7) -> float:
        return (self.get_gates() < threshold).float().mean().item()


# ─────────────────────────────────────────────
# PART 2: Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward classifier for CIFAR-10 (3×32×32 → 10 classes).
    All linear layers are PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.view(x.size(0), -1))

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values = sum of sigmoid(gate_scores).
        Minimising this encourages gates → 0 (weight pruning).
        L1 chosen over L2 because its constant gradient doesn't vanish
        near zero, so it can drive values all the way to 0.
        """
        device = next(self.parameters()).device
        total  = torch.zeros(1, device=device)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def mean_gate(self) -> float:
        all_gates = torch.cat([l.get_gates().flatten() for l in self.prunable_layers()])
        return all_gates.mean().item()

    def overall_sparsity(self, threshold: float = 0.7) -> float:
        pruned = total = 0
        for layer in self.prunable_layers():
            g      = layer.get_gates()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# PART 3: Training & Evaluation
# ─────────────────────────────────────────────

def get_dataloaders(batch_size: int = 128):
    norm     = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf  = transforms.Compose([transforms.ToTensor(), norm])

    train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=256,       shuffle=False, num_workers=0, pin_memory=False)
    return train_dl, test_dl


def train_epoch(model, loader, optimizer, lam: float, device) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits   = model(x)
        cls_loss = F.cross_entropy(logits, y)
        sp_loss  = model.sparsity_loss()
        loss     = cls_loss + lam * sp_loss
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total


def run_experiment(lam: float, epochs: int, device, train_dl, test_dl):
    print(f"\n{'='*55}\n  λ = {lam}   ({epochs} epochs)\n{'='*55}")
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_dl, optimizer, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            acc = evaluate(model, test_dl, device)
            sp  = model.overall_sparsity()
            mg  = model.mean_gate()
            print(f"  Epoch {epoch:2d} | loss {loss:.4f} | acc {acc*100:.2f}% | sparsity {sp*100:.1f}% | mean_gate {mg:.3f}")

    acc = evaluate(model, test_dl, device)
    sp  = model.overall_sparsity()
    print(f"\n  ✓ FINAL  acc={acc*100:.2f}%  sparsity={sp*100:.1f}%")

    gates = np.concatenate([l.get_gates().numpy().flatten()
                            for l in model.prunable_layers()])
    return acc, sp, gates


def plot_results(results: dict):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    colors = ['steelblue', 'darkorange', 'seagreen']
    for ax, (color, (lam, (acc, sp, gates))) in zip(axes, zip(colors, results.items())):
        ax.hist(gates, bins=100, color=color, edgecolor='none', alpha=0.85)
        ax.axvline(0.7, color='red', linestyle='--', linewidth=1.2, label='threshold=0.7')
        ax.set_title(f'λ = {lam}\nacc={acc*100:.1f}%   sparse={sp*100:.1f}%', fontsize=11)
        ax.set_xlabel('Gate value')
        ax.set_ylabel('Count')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle('Gate Value Distributions – Self-Pruning Network (CIFAR-10)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gate_distributions.png', dpi=150, bbox_inches='tight')
    print("\n  Saved → gate_distributions.png")
    plt.close()


def print_table(results: dict):
    print("\n" + "="*52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>14} {'Sparsity (%)':>14}")
    print("="*52)
    for lam, (acc, sp, _) in results.items():
        print(f"  {lam:<12} {acc*100:>13.2f}% {sp*100:>13.1f}%")
    print("="*52)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS  = 10
    LAMBDAS = [1e-4, 5e-3]   # low / high sparsity pressure

    print(f"Device: {device}")
    train_dl, test_dl = get_dataloaders()

    results = {}
    for lam in LAMBDAS:
        acc, sp, gates = run_experiment(lam, EPOCHS, device, train_dl, test_dl)
        results[lam]   = (acc, sp, gates)

    print_table(results)
    plot_results(results)
