"""E19: ResNet — residual connections fix deep-network gradients.

Two experiments on MNIST:
  1. at initialization, the gradient reaching the first layer of a
     12-block plain conv net is orders of magnitude smaller than in the
     identical residual net (the vanishing-gradient measurement),
  2. after identical training, the residual net is at least as accurate.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsl.common import DATA, ExperimentResult, run_experiment


class Block(nn.Module):
    def __init__(self, ch: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = self.c2(h)
        return F.relu(x + h) if self.residual else F.relu(h)


class Net(nn.Module):
    def __init__(self, blocks: int = 4, ch: int = 32, residual: bool = False):
        super().__init__()
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(ch, residual) for _ in range(blocks)])
        self.head = nn.Linear(ch, 10)

    def forward(self, x):
        h = F.relu(self.stem(x))
        h = self.blocks(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(h)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA / "mnist", train=True, download=True, transform=tf)
    te = datasets.MNIST(DATA / "mnist", train=False, download=True, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(20000)])
    ytr = torch.tensor([tr[i][1] for i in range(20000)])
    Xte = torch.stack([te[i][0] for i in range(5000)])
    yte = torch.tensor([te[i][1] for i in range(5000)])
    return Xtr, ytr, Xte, yte


def grad_norm(residual: bool, depth: int = 12, seed: int = 0) -> float:
    torch.manual_seed(seed)
    net = Net(blocks=depth, residual=residual)
    x = torch.randn(16, 1, 28, 28)
    net(x).sum().backward()
    return float(net.stem.weight.grad.norm())


def train_eval(residual: bool, Xtr, ytr, Xte, yte, seed: int = 0) -> float:
    torch.manual_seed(seed)
    net = Net(blocks=4, residual=residual)
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-3)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(1200):
        idx = rng.integers(0, len(Xtr), size=128)
        loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return acc


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        g_plain = grad_norm(False)
        g_res = grad_norm(True)
        ratio = g_plain / (g_res + 1e-30)
        res.check("residual net preserves first-layer gradients at depth 12",
                  g_plain < g_res / 10,
                  f"grad_plain={g_plain:.2e}, grad_res={g_res:.2e}, ratio={ratio:.1f}x")

        Xtr, ytr, Xte, yte = load_mnist()
        acc_plain = train_eval(False, Xtr, ytr, Xte, yte)
        acc_res = train_eval(True, Xtr, ytr, Xte, yte)
        res.check("both networks learn MNIST", acc_plain > 0.90 and acc_res > 0.90,
                  f"plain={acc_plain:.3f}, residual={acc_res:.3f}")
        res.check("residual net is more accurate at equal budget", acc_res > acc_plain + 0.01,
                  f"residual={acc_res:.3f} vs plain={acc_plain:.3f}")

        res.metrics.update({"grad_plain": g_plain, "grad_res": g_res,
                            "grad_ratio": ratio, "acc_plain": acc_plain,
                            "acc_residual": acc_res})
    return run_experiment("e19_resnet", body)
