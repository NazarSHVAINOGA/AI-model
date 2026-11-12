"""Conformal-калібрування «сюрпризу» та виявлення аномалій з контролем FDR.

Проблема, яку це розв'язує
--------------------------
У поточній системі пороги аномалій (|mZ| > 3.5, відхилення прогнозу > 20,
падіння відділення > 15) задані вручну й не мають жодної статистичної гарантії:
невідомо, скільки хибних тривог вони видають, і вони не залежать від того,
наскільки «шумна» конкретна посада.

Що робимо натомість
-------------------
1. Нейромережа дає очікування ŷ та масштаб σ̂ індексу виконання.
2. Міра неконформності (nonconformity score): ``s = |y − ŷ| / σ̂``.
3. **Зважений Mondrian split-conformal**: p-значення нового спостереження — це
   (зважена) частка калібрувальних сюрпризів, що не менші за поточний.
   * *Mondrian* — окремо для кожної посади (умовне покриття за стратою);
   * *зважений* з експоненційним загасанням за віком — послаблює припущення
     про обмінюваність у часових рядах (Barber, Candès, Ramdas, Tibshirani, 2023).
4. Аномалія = p-значення, що пройшло процедуру **Бенджаміні–Хохберга** (BH).
   Для conformal p-значень (які є PRDS) BH контролює FDR на рівні q
   (Bates, Candès, Lei, Romano, Sesia, 2023, *Testing for outliers with
   conformal p-values*, Ann. Statist.). На практиці це означає: серед усіх
   позначених у місяці працівників частка хибних тривог у середньому ≤ q.

Усе — чиста math на numpy, без Django і torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np


def decay_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    """Експоненційні ваги ``w = 2^(-age / half_life)``; ``half_life=inf`` → рівні ваги."""
    ages = np.asarray(ages, dtype=float)
    if not np.isfinite(half_life):
        return np.ones_like(ages)
    return np.power(2.0, -ages / half_life)


def conformal_pvalues(
    cal_scores: np.ndarray,
    test_scores: np.ndarray,
    cal_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Зважені conformal p-значення для кожного test-сюрпризу.

    p(s) = ( Σ_i w_i · 1[s_i ≥ s]  +  w_new ) / ( Σ_i w_i + w_new ),   w_new = 1.

    «+ w_new» — власна маса тестової точки; саме вона дає скінченновибіркову
    валідність (p-значення ніколи не дорівнює 0 і є консервативним).
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    test_scores = np.atleast_1d(np.asarray(test_scores, dtype=float))
    if len(cal_scores) == 0:
        return np.ones_like(test_scores)
    w = np.ones_like(cal_scores) if cal_weights is None else np.asarray(cal_weights, dtype=float)
    order = np.argsort(cal_scores)
    s_sorted, w_sorted = cal_scores[order], w[order]
    # tail[j] = Σ_{i>=j} w_sorted[i]
    tail = np.concatenate([np.cumsum(w_sorted[::-1])[::-1], [0.0]])
    pos = np.searchsorted(s_sorted, test_scores, side="left")  # перший індекс із s_i >= s
    w_new = 1.0
    return (tail[pos] + w_new) / (w.sum() + w_new)


def weighted_quantile(
    scores: np.ndarray, level: float, weights: Optional[np.ndarray] = None
) -> float:
    """Зважений (1 − α)-квантиль із масою w_new на +∞ (консервативна поправка split-conformal).

    Повертає найменше s таке, що Σ_{i: s_i ≤ s} w_i ≥ level · (Σ w + w_new).
    Якщо вимога недосяжна (замало калібрувальних даних), повертає +∞.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float("inf")
    w = np.ones_like(scores) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(scores)
    s, ww = scores[order], w[order]
    target = level * (w.sum() + 1.0)
    cum = np.cumsum(ww)
    k = np.searchsorted(cum, target, side="left")
    return float(s[k]) if k < len(s) else float("inf")


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.1) -> np.ndarray:
    """Процедура BH: булева маска відхилень гіпотез «норма» при FDR ≤ q."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k = np.max(np.nonzero(below)[0])
        reject[order[: k + 1]] = True
    return reject


@dataclass
class MondrianConformal:
    """Калібрувальна множина сюрпризів, розбита за стратами (посадами).

    ``half_life`` — період напіврозпаду ваг у місяцях. ``min_stratum`` —
    мінімум точок у страті; якщо менше, страта доповнюється глобальною множиною.
    """

    half_life: float = 8.0
    min_stratum: int = 50
    _scores: Dict[int, np.ndarray] = field(default_factory=dict)
    _weights: Dict[int, np.ndarray] = field(default_factory=dict)
    _global_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _global_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, scores: np.ndarray, strata: np.ndarray, ages: np.ndarray) -> "MondrianConformal":
        scores = np.asarray(scores, dtype=float)
        strata = np.asarray(strata)
        w = decay_weights(ages, self.half_life)
        self._global_scores, self._global_weights = scores, w
        self._scores, self._weights = {}, {}
        for s in np.unique(strata):
            sel = strata == s
            self._scores[int(s)] = scores[sel]
            self._weights[int(s)] = w[sel]
        return self

    def _pool(self, stratum: int):
        sc, w = self._scores.get(int(stratum)), self._weights.get(int(stratum))
        if sc is None or len(sc) < self.min_stratum:
            return self._global_scores, self._global_weights
        return sc, w

    def pvalues(self, scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        strata = np.asarray(strata)
        out = np.ones(len(scores))
        for s in np.unique(strata):
            sel = strata == s
            sc, w = self._pool(int(s))
            out[sel] = conformal_pvalues(sc, scores[sel], w)
        return out

    def quantile(self, alpha: float, stratum: int) -> float:
        sc, w = self._pool(stratum)
        return weighted_quantile(sc, 1.0 - alpha, w)

    # --- серіалізація (JSON) для збереження поряд з вагами моделі ---
    def to_json(self) -> dict:
        return {
            "half_life": self.half_life,
            "min_stratum": self.min_stratum,
            "scores": {str(k): v.tolist() for k, v in self._scores.items()},
            "weights": {str(k): v.tolist() for k, v in self._weights.items()},
            "global_scores": self._global_scores.tolist(),
            "global_weights": self._global_weights.tolist(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "MondrianConformal":
        obj = cls(half_life=d["half_life"], min_stratum=d["min_stratum"])
        obj._scores = {int(k): np.array(v) for k, v in d["scores"].items()}
        obj._weights = {int(k): np.array(v) for k, v in d["weights"].items()}
        obj._global_scores = np.array(d["global_scores"])
        obj._global_weights = np.array(d["global_weights"])
        return obj
