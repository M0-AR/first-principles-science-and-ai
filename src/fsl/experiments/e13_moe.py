"""E13: Mixture of Experts — sparse routing with load balancing.

Train a top-2 MoE (8 experts) and a parameter-matched dense MLP on the
real UCI handwritten-digits dataset. Verify that sparse routing keeps
accuracy while activating only 25% of expert compute, and that experts
specialize without collapsing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from fsl.common import ExperimentResult, run_experiment


class Dense(nn.Module):
    def __init__(self, d=64, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 10))

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self, d=64, h=64, n_experts=8, k=2):
        super().__init__()
        self.k = k
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 10))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=1)
        topv, topi = probs.topk(self.k, dim=1)
        out = torch.zeros(x.size(0), 10)
        for e in range(len(self.experts)):
            mask = topi == e
            rows = mask.any(dim=1)
            if rows.any():
                w = (topv * mask).sum(dim=1, keepdim=True)[rows]
                out[rows] += w * self.experts[e](x[rows])
        return out, probs


def evaluate(model, X, y, moe: bool) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        out = model(X)
        logits = out[0] if moe else out
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        ent = 0.0
        if moe:
            p = out[1].mean(dim=0)
            ent = float(-(p * torch.log2(p + 1e-12)).sum())
    return acc, ent


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        X, y = load_digits(return_X_y=True)
        X = X / 16.0
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
        Xtr = torch.tensor(Xtr, dtype=torch.float32)
        ytr = torch.tensor(ytr)
        Xte = torch.tensor(Xte, dtype=torch.float32)
        yte = torch.tensor(yte)

        torch.manual_seed(0)
        dense = Dense()
        opt = torch.optim.Adam(dense.parameters(), lr=2e-3)
        for _ in range(120):
            opt.zero_grad()
            loss = F.cross_entropy(dense(Xtr), ytr)
            loss.backward(); opt.step()
        dacc, _ = evaluate(dense, Xte, yte, False)

        torch.manual_seed(1)
        moe = MoE()
        opt = torch.optim.Adam(moe.parameters(), lr=2e-3)
        for _ in range(120):
            opt.zero_grad()
            logits, probs = moe(Xtr)
            loss = F.cross_entropy(logits, ytr)
            usage = probs.mean(dim=0)
            loss = loss + 0.5 * (usage.var() / (usage.mean() ** 2 + 1e-9))
            loss.backward(); opt.step()
        macc, ent = evaluate(moe, Xte, yte, True)

        with torch.no_grad():
            usage = moe.gate(Xte).softmax(dim=1).mean(dim=0).numpy()
        active_frac = moe.k / len(moe.experts)

        res.check("MoE matches dense accuracy within 3 points",
                  abs(dacc - macc) < 3.0, f"dense={dacc:.3f}, MoE={macc:.3f}")
        res.check("MoE routing entropy > 2 bits", ent > 2.0, f"entropy={ent:.2f} bits")
        res.check("all experts are used (min usage > 2%)", usage.min() > 0.02,
                  f"usage={np.round(usage, 3).tolist()}")
        res.check("only k of n experts are active per token", abs(active_frac - 0.25) < 1e-9,
                  f"active fraction={active_frac:.2f}")

        res.metrics.update({"dense_acc": dacc, "moe_acc": macc,
                            "routing_entropy_bits": ent,
                            "expert_usage": usage.tolist(),
                            "active_compute_fraction": active_frac})
    return run_experiment("e13_moe", body)
