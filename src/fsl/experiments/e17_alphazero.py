"""E17: AlphaZero — self-play reinforcement learning on tic-tac-toe.

A tiny policy/value network trained only by self-play with PUCT
Monte-Carlo tree search. No human games and no heuristics: every target
comes from the search itself. Verify the trained agent beats random play
and that the policy/value losses fall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsl.common import ExperimentResult, run_experiment

WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)]


def winner(s: tuple) -> int:
    for a, b, c in WINS:
        if s[a] != 0 and s[a] == s[b] == s[c]:
            return s[a]
    return 0


def terminal(s: tuple) -> bool:
    return winner(s) != 0 or all(v != 0 for v in s)


def encode(s: tuple, player: int) -> np.ndarray:
    v = np.zeros(27, dtype=np.float32)
    for i, x in enumerate(s):
        base = 0 if x == 0 else (1 if x == player else 2)
        v[base * 9 + i] = 1.0
    return v


class AZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(27, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.policy = nn.Linear(128, 9)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), torch.tanh(self.value(h)).squeeze(-1)


@dataclass
class Node:
    P: np.ndarray
    N: np.ndarray
    W: np.ndarray
    legal: list | None = None
    children: dict = field(default_factory=dict)


def evaluate(net: AZNet, states: list[tuple], player: int) -> tuple[np.ndarray, float]:
    if not states:
        return np.zeros(0), 0.0
    x = torch.tensor(np.stack([encode(s, player) for s in states]))
    with torch.no_grad():
        logits, val = net(x)
    p = F.softmax(logits, dim=1).numpy()
    return p, float(val[0])


def node_arrays(legal):
    return np.zeros(9), np.zeros(9), np.zeros(9)


def mcts(net: AZNet, state: tuple, player: int, sims: int, rng,
         noise: float = 0.0) -> tuple[np.ndarray, Node]:
    root = Node(*node_arrays(None), legal=[i for i, v in enumerate(state) if v == 0])
    p, _ = evaluate(net, [state], player)
    for i in root.legal:
        root.P[i] = p[0, i]
    root.P /= root.P.sum() + 1e-9
    if noise > 0:
        d = rng.dirichlet(np.ones(len(root.legal)) * 0.3)
        mask = np.zeros(9)
        mask[root.legal] = d
        root.P = (1 - noise) * root.P + noise * mask
        root.P /= root.P.sum()

    for _ in range(sims):
        s, pl = state, player
        node = root
        path = []
        while True:
            if terminal(s):
                v = 0.0 if winner(s) == 0 else -1.0
                break
            if node.legal is None:
                p, v = evaluate(net, [s], pl)
                node.legal = [i for i, x in enumerate(s) if x == 0]
                q = p[0].copy()
                legal_mask = np.zeros(9, dtype=bool)
                legal_mask[node.legal] = True
                q[~legal_mask] = 0.0
                node.P = q / (q.sum() + 1e-9)
                break
            n = node.N.sum()
            qv = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
            u = 1.5 * node.P * math.sqrt(n + 1) / (1 + node.N)
            a = int(np.argmax(qv + u))
            path.append((node, a))
            s = s[:a] + (pl,) + s[a + 1:]
            pl = -pl
            child = node.children.get(a)
            if child is None:
                child = Node(*node_arrays(None))
                node.children[a] = child
            node = child
        for nd, a in reversed(path):
            v = -v
            nd.N[a] += 1
            nd.W[a] += v
    visits = root.N / (root.N.sum() + 1e-9)
    return visits, root


SYMMETRIES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [6, 7, 8, 3, 4, 5, 0, 1, 2],
    [0, 3, 6, 1, 4, 7, 2, 5, 8],
    [8, 5, 2, 7, 4, 1, 6, 3, 0],
]


def augment_data(data: list[tuple]) -> list[tuple]:
    out = []
    for x, pi, z in data:
        xb = x.reshape(3, 9)
        for perm in SYMMETRIES:
            out.append((xb[:, perm].reshape(27).copy(), pi[perm], z))
    return out


def self_play(net: AZNet, games: int, sims: int, rng) -> list[tuple]:
    data = []
    for _ in range(games):
        s = (0,) * 9
        pl = 1
        history = []
        for move in range(9):
            visits, _ = mcts(net, s, pl, sims, rng, noise=0.25)
            pi = visits.copy()
            if pi.sum() <= 0:
                pi = np.ones(9) / 9
            pi /= pi.sum()
            a = int(np.argmax(pi)) if move >= 4 else int(rng.choice(9, p=pi))
            history.append((encode(s, pl), pi, pl))
            s = s[:a] + (pl,) + s[a + 1:]
            pl = -pl
            if terminal(s):
                break
        w = winner(s)
        for x, pi, mover in history:
            z = 0.0 if w == 0 else (1.0 if w == mover else -1.0)
            data.append((x, pi, z))
    return data


def train(net, data, epochs: int = 4, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    X = torch.tensor(np.stack([d[0] for d in data]))
    PI = torch.tensor(np.stack([d[1] for d in data]), dtype=torch.float32)
    Z = torch.tensor(np.array([d[2] for d in data]), dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        ep = 0.0
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            logits, v = net(X[idx])
            loss = F.cross_entropy(logits, PI[idx]) + F.mse_loss(v, Z[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            ep += float(loss.detach())
        losses.append(ep / max(1, len(X) // 128))
    return losses


def vs_random(net: AZNet, games: int, sims: int, rng) -> dict:
    score = 0.0
    wins = draws = 0
    for g in range(games):
        agent = 1 if g % 2 == 0 else -1
        s = (0,) * 9
        pl = 1
        while not terminal(s):
            if pl == agent:
                visits, _ = mcts(net, s, pl, sims, rng)
                a = int(np.argmax(visits))
            else:
                legal = [i for i, x in enumerate(s) if x == 0]
                a = int(rng.choice(legal))
            s = s[:a] + (pl,) + s[a + 1:]
            pl = -pl
        w = winner(s)
        if w == agent:
            score += 1; wins += 1
        elif w == 0:
            score += 0.5; draws += 1
    return {"score": score / games, "wins": wins, "draws": draws}


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            net = AZNet()
            data = augment_data(self_play(net, games=150, sims=30, rng=rng))
            losses1 = train(net, data, epochs=6)
            data = augment_data(self_play(net, games=150, sims=30, rng=rng))
            losses = train(net, data, epochs=6)
            net.eval()

            result = vs_random(net, games=100, sims=40, rng=rng)
            res.check("trained agent beats random play (win=1, draw=0.5)", result["score"] > 0.85,
                      f"score={result['score']:.3f} ({result['wins']} wins, {result['draws']} draws / 100)")
            res.check("training loss decreases", losses[-1] < losses[0],
                      f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

            res.metrics.update({"selfplay_positions": len(data), "losses": losses,
                                "losses_iter1": losses1, "vs_random": result,
                                "sims_selfplay": 30, "sims_eval": 40})
        finally:
            torch.set_num_threads(prev_threads)
    return run_experiment("e17_alphazero", body)
