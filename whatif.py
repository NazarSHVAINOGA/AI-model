"""Prescriptive what-if: «що мінімально треба зробити, щоб досягти цільової оцінки».

Задача як лінійна програма
--------------------------
Нехай q_k — поточне виконання KPI k (вирівняне: більше = краще), w_k — вага,
σ_k — типове коливання KPI саме цього працівника (його власна «одиниця зусиль»).
Оцінка = Σ w_k · min(100, q_k). Шукаємо підвищення Δ_k ≥ 0::

    minimize    Σ_k  Δ_k / σ_k                      (зусилля в «власних σ»)
    subject to  Σ_k  w_k · Δ_k  ≥  T − S₀ + z_δ·s   (досягти цілі T із надійністю 1−δ)
                0 ≤ Δ_k ≤ min(100 − q_k,  m · σ_k)   (стеля 100 і реалістичність: ≤ m σ)

де S₀ — поточна оцінка, s — стандартне відхилення передбаченої оцінки, z_δ — квантиль
N(0,1) (chance-constraint: ціль досягається з імовірністю ≥ 1−δ у гаусовому наближенні).

Структура розв'язку
-------------------
Це дробова задача про рюкзак: «ціна» KPI k за одиницю виграшу в оцінці дорівнює
1/(w_k σ_k). Оптимум — жадібно за спаданням ``w_k·σ_k``: спочатку заповнюємо KPI з
найбільшим виграшем на одиницю зусиль до його ліміту, потім наступний. Тому
розв'язок (а) завжди має ≤ 1 «частково заповнений» KPI і (б) легко пояснюється людині.
Реалізовано і через LP (``scipy.optimize.linprog``), і жадібно; тест перевіряє збіг.

Ідея «акційності» (actionable recourse) — Ustun, Spangher, Liu, 2019; вартість у
власних σ користувача замість абсолютних одиниць робить план порівнянним між KPI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


@dataclass
class WhatIfPlan:
    feasible: bool
    deltas: np.ndarray             # [K] підвищення KPI у пунктах виконання
    effort_sigma: float            # сумарні зусилля у σ
    start_score: float
    expected_score: float          # оцінка після виконання плану (без запасу надійності)
    required_gain: float           # потрібний приріст з урахуванням запасу
    max_reachable: float           # максимум, досяжний за обмеженнями


def _caps(q: np.ndarray, sigma: np.ndarray, max_sigma: float) -> np.ndarray:
    return np.maximum(0.0, np.minimum(100.0 - np.minimum(q, 100.0), max_sigma * sigma))


def current_score(q: Sequence[float], w: Sequence[float]) -> float:
    q, w = np.asarray(q, float), np.asarray(w, float)
    return float((w * np.clip(q, 0, 100)).sum())


def solve_lp(
    q: Sequence[float], w: Sequence[float], sigma: Sequence[float], target: float,
    max_sigma: float = 2.0, score_std: float = 0.0, confidence: float = 0.5,
) -> WhatIfPlan:
    q, w, sigma = (np.asarray(x, float) for x in (q, w, sigma))
    sigma = np.maximum(sigma, 1e-6)
    s0 = current_score(q, w)
    need = target - s0 + (norm.ppf(confidence) * score_std if score_std > 0 else 0.0)
    caps = _caps(q, sigma, max_sigma)
    max_gain = float((w * caps).sum())
    if need <= 0:
        return WhatIfPlan(True, np.zeros_like(q), 0.0, s0, s0, need, s0 + max_gain)
    if need > max_gain + 1e-9:
        return WhatIfPlan(False, caps, float((caps / sigma).sum()), s0, s0 + max_gain, need, s0 + max_gain)
    res = linprog(
        c=1.0 / sigma, A_ub=[-w], b_ub=[-need],
        bounds=[(0.0, float(c)) for c in caps], method="highs",
    )
    d = res.x
    return WhatIfPlan(True, d, float((d / sigma).sum()), s0, s0 + float((w * d).sum()), need, s0 + max_gain)


def solve_greedy(
    q: Sequence[float], w: Sequence[float], sigma: Sequence[float], target: float,
    max_sigma: float = 2.0, score_std: float = 0.0, confidence: float = 0.5,
) -> WhatIfPlan:
    """Те саме, але жадібним алгоритмом дробового рюкзака (еталон для тесту)."""
    q, w, sigma = (np.asarray(x, float) for x in (q, w, sigma))
    sigma = np.maximum(sigma, 1e-6)
    s0 = current_score(q, w)
    need = target - s0 + (norm.ppf(confidence) * score_std if score_std > 0 else 0.0)
    caps = _caps(q, sigma, max_sigma)
    max_gain = float((w * caps).sum())
    d = np.zeros_like(q)
    if need > 0 and need <= max_gain + 1e-9:
        left = need
        for k in np.argsort(-(w * sigma)):
            if w[k] <= 0 or caps[k] <= 0:
                continue
            take = min(caps[k], left / w[k])
            d[k], left = take, left - w[k] * take
            if left <= 1e-12:
                break
    feasible = need <= max_gain + 1e-9
    if not feasible:
        d = caps
    return WhatIfPlan(feasible, d, float((d / sigma).sum()), s0, s0 + float((w * d).sum()), need, s0 + max_gain)
