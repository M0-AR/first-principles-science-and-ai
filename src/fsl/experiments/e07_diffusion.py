"""E07: Diffusion models from scratch — denoising score matching.

Train a small epsilon-predicting MLP on an 8-mode Gaussian mixture with a
variance-preserving diffusion process, then sample by ancestral DDPM.
Checks that the learned reverse process covers every mode of the true
distribution (no mode collapse).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fsl.common import ExperimentResult, FIGURES, run_experiment

T = 400
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHAS = 1 - BETAS
ABAR = torch.cumprod(ALPHAS, dim=0)


def data(n: int, rng: np.random.Generator) -> np.ndarray:
    k = rng.integers(0, 8, size=n)
    ang = k * np.pi / 4
    centers = np.stack([2.0 * np.cos(ang), 2.0 * np.sin(ang)], axis=1)
    return centers + rng.normal(scale=0.12, size=(n, 2))


class EpsNet(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(16) * 2.0, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(2 + 32, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def forward(self, x, t):
        ang = t[:, None] * self.freq[None, :] * 6.283
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.net(torch.cat([x, emb], dim=1))


def train(steps: int = 3500, batch: int = 256, seed: int = 0) -> EpsNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EpsNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps):
        x0 = torch.tensor(data(batch, rng), dtype=torch.float32)
        t = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        ab = ABAR[t][:, None]
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        loss = nn.functional.mse_loss(net(xt, t.float() / T), eps)
        opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def sample(net: EpsNet, n: int = 2000) -> np.ndarray:
    net.eval()
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        tt = torch.full((n,), t, dtype=torch.float32) / T
        eps = net(x, tt)
        alpha, ab = ALPHAS[t], ABAR[t]
        beta = BETAS[t]
        mean = (x - beta / torch.sqrt(1 - ab) * eps) / torch.sqrt(alpha)
        x = mean + (torch.sqrt(beta) * torch.randn_like(x) if t > 0 else 0)
    return x.numpy()


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        net = train()
        x = sample(net)
        centers = 2.0 * np.stack([np.cos(np.arange(8) * np.pi / 4),
                                  np.sin(np.arange(8) * np.pi / 4)], axis=1)
        d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        frac = np.array([np.mean(nearest == k) for k in range(8)])
        covered = np.mean(frac > 0.03)
        res.check("all 8 modes are covered", covered == 1.0,
                  f"mode fractions={np.round(frac, 3).tolist()}")
        res.check("each mode has a generated sample within 0.5",
                  bool(np.all(d.min(axis=0) < 0.5)),
                  f"max nearest distance={d.min(axis=0).max():.3f}")
        spread = float(np.mean(np.std(x, axis=0)))
        res.check("generated cloud is not collapsed", spread > 0.9,
                  f"per-axis std={spread:.3f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.4, label="generated")
        ax.scatter(centers[:, 0], centers[:, 1], c="r", s=60, marker="x", label="modes")
        ax.legend(); ax.set_aspect("equal"); ax.set_title("DDPM samples, 8-Gaussian")
        fig.tight_layout(); fig.savefig(FIGURES / "e07_diffusion.png", dpi=140)
        plt.close(fig)

        res.metrics.update({"mode_fractions": frac.tolist(), "spread": spread,
                            "n_samples": int(len(x)), "diffusion_steps": T})
    return run_experiment("e07_diffusion", body)
