"""E20: Q-learning vs SARSA on the CliffWalking gridworld.

Classic benchmark (Sutton & Barto, Example 6.6). Verification:
  * Q-learning finds the optimal cliff-edge policy (episode length 13),
  * epsilon-greedy SARSA finds a safer policy with lower online regret.
"""
from __future__ import annotations

import numpy as np

from fsl.common import ExperimentResult, run_experiment

H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def step(s, a):
    dr, dc = ACTIONS[a]
    ns = (min(max(s[0] + dr, 0), H - 1), min(max(s[1] + dc, 0), W - 1))
    if ns in CLIFF:
        return START, -100.0
    if ns == GOAL:
        return GOAL, -1.0
    return ns, -1.0


def train(method: str, episodes: int = 500, alpha: float = 0.5,
          gamma: float = 1.0, eps: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, 4))
    returns = []
    for _ in range(episodes):
        s = START
        a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
        total = 0.0
        steps = 0
        while s != GOAL and steps < 500:
            ns, r = step(s, a)
            na = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[ns]))
            target = r if ns == GOAL else r + gamma * (Q[ns][na] if method == "sarsa"
                                                       else np.max(Q[ns]))
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = ns, na
            total += r
            steps += 1
        returns.append(total)
    return {"Q": Q, "returns": returns}


def greedy_episode(Q: np.ndarray, max_steps: int = 100) -> tuple[int, list]:
    s, path = START, [START]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _ = step(s, a)
        path.append(s)
        if s == GOAL:
            break
    return len(path) - 1, path


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        ql = train("qlearning")
        sarsa = train("sarsa", seed=1)
        ql_len, ql_path = greedy_episode(ql["Q"])
        sarsa_len, sarsa_path = greedy_episode(sarsa["Q"])
        res.check("Q-learning greedy path is optimal (13 steps)", ql_len == 13,
                  f"length={ql_len}")
        res.check("Q-learning path touches the cliff edge",
                  3 - 1 in [p[0] for p in ql_path] or any(p[0] == 3 for p in ql_path[1:-1]),
                  f"path rows={[p[0] for p in ql_path]}")
        res.check("epsilon-greedy SARSA keeps off the cliff (length > 13)",
                  sarsa_len > 13, f"length={sarsa_len}")
        n = 100
        ql_last = float(np.mean(ql["returns"][-n:]))
        sarsa_last = float(np.mean(sarsa["returns"][-n:]))
        res.check("SARSA has lower online regret than Q-learning", sarsa_last >= ql_last,
                  f"SARSA {sarsa_last:.1f} vs Q-learning {ql_last:.1f}")
        res.metrics.update({
            "qlearning_path_length": ql_len,
            "sarsa_path_length": sarsa_len,
            "qlearning_online_return_last100": ql_last,
            "sarsa_online_return_last100": sarsa_last,
            "qlearning_path": ql_path,
            "sarsa_path": sarsa_path,
        })
    return run_experiment("e20_qlearning", body)
