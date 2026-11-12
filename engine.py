"""Рушій аналізу Copilot: збирає модель, очищення, conformal, Шеплі й what-if докупи.

Працює з «голими» масивами (без Django), тому однаково годиться і для симуляції,
і для даних із бази (див. ``apps/copilot/service.py``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import whatif
from .cleaning import sequential_predict
from .conformal import MondrianConformal
from .evaluate import adjust_common_shock
from .features import GROUPS, build_dataset
from .model import KpiForecaster, to_tensors
from .shapley import explain
from .simulator import KMAX, use_domain, weight_matrix
from .train import ARTIFACTS, device, load_model

GROUP_UA = {
    "HIST": "власна історія виконання",
    "CAL": "сезонність",
    "EVENTS": "зовнішні події",
    "PEERS": "результати колег",
    "IDENT": "посада та відділення",
}


@dataclass
class Analysis:
    emp: int
    t: int
    actual_index: float
    expected_index: float
    sigma: float
    signed_surprise: float       # (факт − очікування) / σ̂, після поправки на спільний шок
    p_value: float
    flagged: bool
    shap: Dict[str, float]       # внески груп в очікування (у пунктах індексу)
    shap_base: float
    kpi_q: List[float]
    kpi_sigma: List[float]
    weights: List[float]
    systemic_shift: float = 0.0  # медіанний зсув групи (посада × місяць), у σ


class Engine:
    def __init__(self, model: KpiForecaster, conformal: MondrianConformal, window: int = 12) -> None:
        self.model = model.to(device()).eval()
        self.conformal = conformal
        self.window = window

    @classmethod
    def load(cls, art_dir: Path = ARTIFACTS) -> "Engine":
        model, cfg = load_model(art_dir / "forecaster.pt", map_location=device())
        # Якщо є калібрування на власній історії бази — воно точніше за симуляційне.
        own = art_dir / "conformal_db.json"
        src = own if own.exists() else art_dir / "conformal.json"
        conf = MondrianConformal.from_json(json.loads(src.read_text(encoding="utf-8")))
        eng = cls(model, conf, cfg.window)
        meta = art_dir / "meta.json"
        if meta.exists():  # інша предметна область (напр., Rossmann) зі своїми вагами KPI
            eng.domain_weights = np.array(json.loads(meta.read_text(encoding="utf-8"))["weights"], dtype=np.float32)
        eng.calibrated_on = "db" if own.exists() else "simulation"
        return eng

    calibrated_on = "simulation"
    domain_weights = None

    def calibrate(self, q, pos, branch, events, month, active, last_t: int, months: int = 12,
                  half_life: float = 8.0, min_history: int = 3) -> int:
        """Перекалібровує conformal на власній історії: сюрпризи за ``months`` місяців, що передують ``last_t``.

        Для кожного місяця t' < last_t проганяється той самий конвеєр (очищення + поправка на спільний
        шок), збираються |сюрпризи| з їхнім віком — на них будується нова калібрувальна множина.
        Повертає кількість використаних спостережень. Поточний місяць last_t у калібрування не входить.
        """
        scores, strata, ages = [], [], []
        for t in range(max(min_history, last_t - months), last_t):
            for e, a in self.analyse(q, pos, branch, events, month, active, t).items():
                scores.append(abs(a.signed_surprise))
                strata.append(int(pos[e]))
                ages.append(last_t - t)
        if scores:
            self.conformal = MondrianConformal(half_life=half_life).fit(
                np.array(scores), np.array(strata), np.array(ages))
            self.calibrated_on = "db"
        return len(scores)

    def analyse(self, *a, **kw) -> Dict[int, Analysis]:
        with use_domain(self.domain_weights):
            return self._analyse(*a, **kw)

    def _analyse(
        self,
        q: np.ndarray, pos: np.ndarray, branch: np.ndarray, events: np.ndarray,
        month: np.ndarray, active: np.ndarray, t: int, alpha: float = 0.01,
    ) -> Dict[int, Analysis]:
        """Аналіз усіх працівників за місяць з індексом ``t`` (0-базований у межах панелі)."""
        ds = build_dataset(q, pos, branch, events, month, active, window=self.window)
        t_from = max(0, t - self.window)
        ds = ds.time_range(t_from, t + 1)
        if not (ds.t == t).any():
            return {}
        seq = sequential_predict(self.model, ds, t_from, clean=True)
        adj = adjust_common_shock(ds, seq["signed"])
        p = self.conformal.pvalues(np.abs(adj), ds.pos)
        cur = np.nonzero(ds.t == t)[0]
        # Пояснення Шеплі — лише для цільового місяця.
        sub = ds.select(cur)
        ex = explain(self.model, to_tensors(sub, device()))
        W = weight_matrix()
        out: Dict[int, Analysis] = {}
        for j, i in enumerate(cur):
            e = int(ds.emp[i])
            # Типове коливання KPI працівника (σ у пунктах) — з його історії.
            hist = q[e, max(0, t - 24):t]
            sig_k = np.nan_to_num(np.nanstd(hist, axis=0), nan=5.0) if hist.size else np.full(KMAX, 5.0)
            sig_k = np.maximum(sig_k, 2.0)
            out[e] = Analysis(
                emp=e, t=t,
                actual_index=float(seq["y_idx"][i]), expected_index=float(seq["mu_idx"][i]),
                sigma=float(seq["sig_idx"][i]), signed_surprise=float(adj[i]),
                p_value=float(p[i]), flagged=bool(p[i] <= alpha),
                shap={g: float(ex["phi"][j, k]) for k, g in enumerate(GROUPS)},
                shap_base=float(ex["base"][j]),
                kpi_q=[float(v) for v in np.nan_to_num(q[e, t], nan=0.0)],
                kpi_sigma=[float(v) for v in sig_k],
                weights=[float(v) for v in W[pos[e]]],
                systemic_shift=float(np.nanmedian(seq["signed"][ds.t == t][ds.pos[ds.t == t] == pos[e]])),
            )
        return out

    @staticmethod
    def plan(a: Analysis, target: float, confidence: float = 0.8, max_sigma: float = 2.0) -> whatif.WhatIfPlan:
        k = sum(1 for w in a.weights if w > 0)
        return whatif.solve_lp(
            a.kpi_q[:k], a.weights[:k], a.kpi_sigma[:k], target,
            max_sigma=max_sigma, score_std=a.sigma, confidence=confidence,
        )
