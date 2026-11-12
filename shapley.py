"""Точні значення Шеплі для груп ознак нейромережі.

Питання, на яке це відповідає: *«чому модель очікувала саме таке виконання?»*
Пояснення розкладає очікуваний індекс на внески 5 груп (``features.GROUPS``):
власна історія, календар (сезон), зовнішні події, колеги, ідентичність (посада/відділення).

Метод
-----
Значення Шеплі гравця i в грі з n гравцями::

    φ_i = Σ_{S ⊆ N∖{i}}  |S|!·(n−|S|−1)! / n!  ·  [ v(S ∪ {i}) − v(S) ]

Для n = 5 — лише 2^5 = 32 коаліції, тож **точний** розрахунок (без MC-наближень
SHAP) коштує 32 прямих проходи мережі. Функція виграшу ``v(S)`` — очікуваний
індекс виконання моделі, коли присутні лише групи з S, а решта вимкнені.

Чому це коректно саме тут: модель навчалась із груповим dropout
(``train.py``), тож «вимкнена група» — штатний вхід, а не вихід за межі
розподілу (головна проблема Шеплі-пояснень для звичайних мереж).

Властивість ефективності: Σ_i φ_i = v(N) − v(∅). Перевіряється тестом.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, List, Tuple

import numpy as np
import torch

from .features import GROUPS, N_GROUPS
from .model import KpiForecaster


def _all_masks() -> np.ndarray:
    """Усі 2^n бінарних масок присутності груп, [2^n, n]."""
    n = N_GROUPS
    return np.array([[(m >> i) & 1 for i in range(n)] for m in range(2 ** n)], dtype=np.float32)


@torch.no_grad()
def coalition_values(model: KpiForecaster, batch: Dict[str, torch.Tensor]) -> np.ndarray:
    """v(S) для всіх коаліцій і всіх прикладів: масив [2^n, B]."""
    model.eval()
    masks = torch.tensor(_all_masks(), device=batch["hq"].device)
    B = batch["hq"].shape[0]
    out = []
    for m in masks:
        present = m.unsqueeze(0).expand(B, -1).contiguous()
        _, _, _, mu_i, _ = model(batch, present)
        out.append(mu_i.cpu().numpy())
    return np.stack(out)


def shapley_from_values(values: np.ndarray) -> np.ndarray:
    """Точні значення Шеплі за таблицею виграшів ``values[2^n, B]`` → [B, n]."""
    n = N_GROUPS
    phi = np.zeros((values.shape[1], n))
    for i in range(n):
        for size in range(n):
            coef = factorial(size) * factorial(n - size - 1) / factorial(n)
            for others in combinations([j for j in range(n) if j != i], size):
                s = sum(1 << j for j in others)
                phi[:, i] += coef * (values[s | (1 << i)] - values[s])
    return phi


def explain(model: KpiForecaster, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Повне пояснення: база v(∅), внески груп φ, підсумок v(N)."""
    values = coalition_values(model, batch)
    phi = shapley_from_values(values)
    return {"base": values[0], "phi": phi, "full": values[-1], "groups": np.array(GROUPS)}
