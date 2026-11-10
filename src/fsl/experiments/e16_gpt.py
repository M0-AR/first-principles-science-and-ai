"""E16: GPT from scratch — decoder-only Transformer language model.

Train a 2-layer, 4-head character-level Transformer on Tiny Shakespeare
and compare its held-out bits/char against the best Markov chain from
E12. Also sample text to show grammatical structure has been learned.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsl.common import DATA, ExperimentResult, run_experiment

CTX = 96
D = 128
STEPS = 2500


class CharGPT(nn.Module):
    def __init__(self, vocab: int, ctx: int = CTX, d: int = D, heads: int = 4, layers: int = 2):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        n = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.blocks(h, mask=mask)
        return self.head(self.ln(h))


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        torch.manual_seed(0)
        raw = np.frombuffer((DATA / "tinyshakespeare.txt").read_bytes(), dtype=np.uint8)
        raw = raw[raw < 128].astype(np.int64)
        split = int(0.9 * len(raw))
        train = torch.tensor(raw[:split])
        test = torch.tensor(raw[split:])
        V = int(raw.max()) + 1

        model = CharGPT(V)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(0)
        losses = []
        model.train()
        for step in range(STEPS):
            idx = torch.tensor(rng.integers(0, len(train) - CTX - 1, size=32))
            x = torch.stack([train[i:i + CTX] for i in idx])
            y = torch.stack([train[i + 1:i + CTX + 1] for i in idx])
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
            if step == 300:
                opt.param_groups[0]["lr"] = 1e-3

        model.eval()
        with torch.no_grad():
            idx = torch.arange(0, len(test) - CTX - 1, CTX)[:80]
            x = torch.stack([test[i:i + CTX] for i in idx])
            y = torch.stack([test[i + 1:i + CTX + 1] for i in idx])
            loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        bits = float(loss / np.log(2))

        with torch.no_grad():
            seed = torch.tensor(raw[:CTX])[None, :]
            out = model.generate if hasattr(model, "generate") else None
            seq = seed.clone()
            for _ in range(300):
                logits = model(seq[:, -CTX:])[:, -1] / 0.9
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                seq = torch.cat([seq, nxt], dim=1)
        text = bytes(seq[0, CTX:].tolist()).decode("ascii", errors="ignore")

        n_chars = len(set(text))
        space_ratio = text.count(" ") / max(len(text), 1)
        res.check("beats the best Markov chain from E12 (<2.408 bits/char)",
                  bits < 2.408, f"Transformer={bits:.3f} bits/char")
        res.check("learned a diverse character distribution", n_chars > 15,
                  f"{n_chars} distinct chars in 300 generated")
        res.check("generated text has word structure", 0.08 < space_ratio < 0.30,
                  f"space ratio={space_ratio:.3f}")

        res.metrics.update({"bits_per_char": bits, "steps": STEPS,
                            "params": int(sum(p.numel() for p in model.parameters())),
                            "final_train_loss": float(np.mean(losses[-100:])),
                            "sample": text[:200]})
    return run_experiment("e16_gpt", body)
