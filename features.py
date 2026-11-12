"""Перетворення панелі в тензори для нейромережі.

Ця функція навмисно приймає «голі» масиви (а не симулятор), щоб ту саму
побудову ознак можна було застосувати і до синтетичних даних, і до даних
із бази Django (див. ``service.py``).

Групи ознак (вони ж «гравці» у Shapley-поясненнях, див. ``shapley.py``):

    0  HIST    — власна історія виконання KPI за попередні W місяців
    1  CAL     — календар (синус/косинус місяця) → навчена сезонність
    2  EVENTS  — зовнішні події (типи × підписана величина), у вікні й у цільовому місяці
    3  PEERS   — одночасні результати колег (медіана відділення/банку, без самого працівника)
    4  IDENT   — ідентичність: посада та відділення
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .simulator import KMAX, N_EVENT_TYPES

GROUPS = ["HIST", "CAL", "EVENTS", "PEERS", "IDENT"]
N_GROUPS = len(GROUPS)

Q_CENTER, Q_SCALE = 100.0, 20.0     # стандартизація q
EV_SCALE = 2.0


def _month_cal(month: np.ndarray) -> np.ndarray:
    ang = 2 * np.pi * (month.astype(np.float32) - 1) / 12
    return np.stack([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


def performance_index(q: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Необрізаний індекс виконання Σ w_k q_k / Σ w_k [N, T] (NaN, де немає даних)."""
    from .simulator import weight_matrix
    w = weight_matrix()[pos][:, None, :]
    valid = ~np.isnan(q)
    num = (w * valid * np.nan_to_num(q)).sum(-1)
    den = (w * valid).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _loo_median(vals: np.ndarray, i: int, fallback: float) -> float:
    """Медіана групи без елемента i (NaN ігноруються)."""
    others = np.delete(vals, i)
    others = others[~np.isnan(others)]
    return float(np.median(others)) if len(others) else fallback


def peer_context(
    idx: np.ndarray, pos: np.ndarray, branch: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """Одночасні результати колег, ВИКЛЮЧАЮЧИ самого працівника (leave-one-out).

    Повертає [N, T, 4]:
      0. медіана *зміни* індексу (t відносно t-1) серед колег відділення;
      1. медіана *зміни* індексу серед колег ТІЄЇ САМОЇ ПОСАДИ по всьому банку;
      2. медіана рівня індексу серед колег відділення;
      3. медіана рівня індексу серед колег тієї самої посади.

    Чому саме так: період фіналізується для всіх одразу, тож результати колег за
    той самий місяць відомі. «Зміна» (а не рівень) не залежить від різниці
    базових рівнів посад. Посадова медіана ловить спільний шок, який чіпляє
    лише одну посаду (наприклад, регуляторну зміну для бухгалтерів), — його
    не видно в середньому по відділенню. Медіана (не середнє) робить ознаку
    стійкою до самих аномалій колег.
    """
    N, T = idx.shape
    out = np.zeros((N, T, 4), dtype=np.float32)
    x = np.where(active, idx, np.nan)
    delta = np.full_like(x, np.nan)
    delta[:, 1:] = x[:, 1:] - x[:, :-1]
    for t in range(T):
        alive = np.nonzero(~np.isnan(x[:, t]))[0]
        if len(alive) == 0:
            continue
        groups = {("b", int(b)): alive[branch[alive] == b] for b in np.unique(branch[alive])}
        groups.update({("p", int(p)): alive[pos[alive] == p] for p in np.unique(pos[alive])})
        for (kind, gid), members in groups.items():
            for j, e in enumerate(members):
                d = _loo_median(delta[members, t], j, 0.0)
                lv = _loo_median(x[members, t], j, 100.0)
                if kind == "b":
                    out[e, t, 0] = d / 10.0
                    out[e, t, 2] = (lv - 100.0) / 10.0
                else:
                    out[e, t, 1] = d / 10.0
                    out[e, t, 3] = (lv - 100.0) / 10.0
    return out


@dataclass
class Dataset:
    """Тензори в numpy (перенос на GPU робить код навчання)."""

    hq: np.ndarray       # [M, W, KMAX] стандартизоване q (0 де немає)
    hm: np.ndarray       # [M, W, KMAX] маска наявності
    hv: np.ndarray       # [M, W] місяць має хоч одне значення
    hcal: np.ndarray     # [M, W, 2]
    hev: np.ndarray      # [M, W, N_EVENT_TYPES]
    tcal: np.ndarray     # [M, 2]
    tev: np.ndarray      # [M, N_EVENT_TYPES]
    peers: np.ndarray    # [M, 4]
    pos: np.ndarray      # [M]
    branch: np.ndarray   # [M]
    y: np.ndarray        # [M, KMAX] «сире» q цільового місяця (0 де немає)
    ym: np.ndarray       # [M, KMAX] маска цілей
    emp: np.ndarray      # [M] індекс працівника
    t: np.ndarray        # [M] індекс місяця

    def __len__(self) -> int:
        return len(self.emp)

    def select(self, sel: np.ndarray) -> "Dataset":
        return Dataset(**{k: getattr(self, k)[sel] for k in self.__dataclass_fields__})

    def time_range(self, lo: int, hi: int) -> "Dataset":
        """Підмножина з lo <= t < hi."""
        return self.select((self.t >= lo) & (self.t < hi))


def build_dataset(
    q: np.ndarray,
    pos: np.ndarray,
    branch: np.ndarray,
    events: np.ndarray,
    month: np.ndarray,
    active: np.ndarray,
    window: int = 12,
    targets: Optional[np.ndarray] = None,
    min_history: int = 1,
) -> Dataset:
    """Будує вікна (працівник, місяць-ціль).

    ``targets`` — необов'язкова булева [N, T]: для яких пар будувати приклад
    (за замовчуванням усі активні місяці з >= ``min_history`` попередніх спостережень).
    """
    N, T, _ = q.shape
    peers_all = peer_context(performance_index(q, pos), pos, branch, active)
    cal = _month_cal(month)  # [T,2]
    ev = (events / EV_SCALE).astype(np.float32)

    valid_obs = ~np.isnan(q)
    q_std = np.where(valid_obs, (q - Q_CENTER) / Q_SCALE, 0.0).astype(np.float32)
    month_has = valid_obs.any(-1)  # [N,T]

    if targets is None:
        targets = active.copy()
        cum = np.cumsum(month_has, axis=1) - month_has  # скільки місяців історії до t
        targets &= cum >= min_history

    es, ts = np.nonzero(targets & valid_obs.any(-1))
    M = len(es)
    W = window
    hq = np.zeros((M, W, KMAX), np.float32)
    hm = np.zeros((M, W, KMAX), np.float32)
    hv = np.zeros((M, W), np.float32)
    hcal = np.zeros((M, W, 2), np.float32)
    hev = np.zeros((M, W, N_EVENT_TYPES), np.float32)
    for i, (e, t) in enumerate(zip(es, ts)):
        lo = max(0, t - W)
        n = t - lo
        # Вікно вирівняне праворуч: останній крок = місяць t-1.
        hq[i, W - n:] = q_std[e, lo:t]
        hm[i, W - n:] = valid_obs[e, lo:t]
        hv[i, W - n:] = month_has[e, lo:t]
        hcal[i, W - n:] = cal[lo:t]
        hev[i, W - n:] = ev[e, lo:t] if ev.ndim == 3 else ev[lo:t]
    y = np.where(valid_obs[es, ts], q[es, ts], 0.0).astype(np.float32)
    ym = valid_obs[es, ts].astype(np.float32)
    return Dataset(
        hq=hq, hm=hm, hv=hv, hcal=hcal, hev=hev,
        tcal=cal[ts], tev=(ev[es, ts] if ev.ndim == 3 else ev[ts]), peers=peers_all[es, ts],
        pos=pos[es].astype(np.int64), branch=branch[es].astype(np.int64),
        y=y, ym=ym, emp=es.astype(np.int64), t=ts.astype(np.int64),
    )


def dataset_from_panel(panel, window: int = 12) -> Dataset:
    return build_dataset(
        panel.q, panel.pos, panel.branch, panel.events, panel.month,
        panel.active(), window=window,
    )
