"""Experiment registry."""
from __future__ import annotations

import importlib
from typing import Callable

EXPERIMENTS: dict[str, str] = {
    "e01_least_action": "e01_least_action",
    "e02_path_integral_g2": "e02_path_integral_g2",
    "e03_quantum_grover": "e03_quantum_grover",
    "e04_order_book": "e04_order_book",
    "e05_kv_cache": "e05_kv_cache",
    "e06_ceres": "e06_ceres",
    "e07_diffusion": "e07_diffusion",
    "e08_fourier_dtmf": "e08_fourier_dtmf",
    "e09_quantization": "e09_quantization",
    "e10_boltzmann": "e10_boltzmann",
    "e11_gps": "e11_gps",
    "e12_markov_shakespeare": "e12_markov_shakespeare",
    "e13_moe": "e13_moe",
    "e14_riemann_gue": "e14_riemann_gue",
    "e15_rag": "e15_rag",
    "e16_gpt": "e16_gpt",
    "e17_alphazero": "e17_alphazero",
    "e18_paged_attention": "e18_paged_attention",
    "e19_resnet": "e19_resnet",
    "e20_qlearning": "e20_qlearning",
    "e21_gan": "e21_gan",
    "e22_double_descent": "e22_double_descent",
    "e23_crypto": "e23_crypto",
    "e24_jepa": "e24_jepa",
}


def load(name: str) -> Callable:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name}")
    mod = importlib.import_module(f"fsl.experiments.{EXPERIMENTS[name]}")
    return mod.run
