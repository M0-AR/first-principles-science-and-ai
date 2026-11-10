"""E24: JEPA-style self-supervised learning.

Learn representations of digits without labels using SimCLR-style
contrastive learning (the pragmatic cousin of JEPA: predict the
representation of one view from another, in embedding space). Then probe
the frozen features with a linear classifier and verify the
representation did not collapse.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fsl.common import ExperimentResult, run_experiment


class Encoder(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 256), nn.LayerNorm(256), nn.GELU(),
                                 nn.Linear(256, d))

    def forward(self, x):
        return self.net(x)


def augment(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=rng) * 0.15
    mask = (torch.rand(x.shape, generator=rng) > 0.2).float()
    return (x + noise) * mask


def nt_xent(z: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temp
    n = z.shape[0]
    mask = torch.eye(n, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    labels = torch.cat([torch.arange(n // 2) + n // 2, torch.arange(n // 2)])
    return F.cross_entropy(sim, labels)


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        X, y = load_digits(return_X_y=True)
        X = torch.tensor(X / 16.0, dtype=torch.float32)
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)

        enc = Encoder()
        proj = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))
        opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)

        for step in range(700):
            idx = torch.randint(0, len(X), (128,), generator=g)
            x = X[idx]
            v1, v2 = augment(x, g), augment(x, g)
            z = proj(torch.cat([enc(v1), enc(v2)]))
            loss = nt_xent(z)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            feats = enc(X).numpy()
        collapse = float(feats.std(axis=0).mean())

        Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.3, random_state=0)
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        probe = float(clf.score(Xte, yte))

        Xtr2, Xte2, ytr2, yte2 = train_test_split(X.numpy(), y, test_size=0.3, random_state=0)
        raw = float(LogisticRegression(max_iter=2000).fit(Xtr2, ytr2).score(Xte2, yte2))

        with torch.no_grad():
            a = F.normalize(enc(augment(X[:400], g)), dim=1)
            b = F.normalize(enc(augment(X[:400], g)), dim=1)
            pos = (a * b).sum(1).mean().item()
            neg = (a * F.normalize(enc(X[400:800]), dim=1)).sum(1).mean().item()

        res.check("frozen features support a strong linear probe", probe > 0.80,
                  f"probe accuracy={probe:.3f} (raw pixels={raw:.3f}, chance=0.10)")
        res.check("representation did not collapse", collapse > 0.05,
                  f"mean feature std={collapse:.3f}")
        res.check("positive views are closer than negative views", pos > neg + 0.05,
                  f"cos(pos)={pos:.3f}, cos(neg)={neg:.3f}")

        res.metrics.update({"probe_acc": probe, "raw_pixel_acc": raw,
                            "feature_std": collapse, "pos_cos": pos, "neg_cos": neg})
    return run_experiment("e24_jepa", body)
