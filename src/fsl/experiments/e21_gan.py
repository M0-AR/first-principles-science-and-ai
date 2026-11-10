"""E21: GAN — adversarial training and mode coverage.

Train a small GAN on UCI handwritten digits (8x8) and test whether it
covers all ten digit classes, stays diverse, and keeps the discriminator
away from trivial collapse.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

from fsl.common import ExperimentResult, run_experiment


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        X, y = load_digits(return_X_y=True)
        X = torch.tensor(X / 8.0 - 1.0, dtype=torch.float32)
        y = torch.tensor(y)
        rng = np.random.default_rng(0)
        torch.manual_seed(0)

        G = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.Tanh())
        D = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128),
                          nn.LeakyReLU(0.2), nn.Linear(128, 1))
        og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        od = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        for step in range(4000):
            idx = rng.integers(0, len(X), size=128)
            real = X[idx]
            z = torch.randn(128, 8)
            fake = G(z).detach()
            d_loss = F.binary_cross_entropy_with_logits(D(real), torch.ones(128, 1)) \
                + F.binary_cross_entropy_with_logits(D(fake), torch.zeros(128, 1))
            od.zero_grad(); d_loss.backward(); od.step()

            z = torch.randn(128, 8)
            g_loss = F.binary_cross_entropy_with_logits(D(G(z)), torch.ones(128, 1))
            og.zero_grad(); g_loss.backward(); og.step()

        with torch.no_grad():
            z = torch.randn(3000, 8)
            fake = G(z).numpy()
            d_real = torch.sigmoid(D(X)).mean().item()
            d_fake = torch.sigmoid(D(torch.tensor(fake, dtype=torch.float32))).mean().item()

        centers = torch.stack([X[y == k].mean(dim=0) for k in range(10)]).numpy()
        d2 = ((fake[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        cls = np.array([np.mean(nearest == k) for k in range(10)])
        diversity = float(fake.std(axis=0).mean() / (X.numpy().std(axis=0).mean() + 1e-9))

        res.check("generator covers all ten digit classes (each >1% of samples)", bool((cls > 0.01).all()),
                  f"class fractions={np.round(cls, 3).tolist()}")
        res.check("no mode collapse (diversity > 50% of real)", diversity > 0.5,
                  f"diversity ratio={diversity:.3f}")
        res.check("discriminator is not saturated", 0.2 < d_real < 0.95 and 0.05 < d_fake < 0.8,
                  f"D(real)={d_real:.3f}, D(fake)={d_fake:.3f}")

        res.metrics.update({"class_fractions": cls.tolist(),
                            "diversity": diversity,
                            "D_real": d_real, "D_fake": d_fake})
    return run_experiment("e21_gan", body)
