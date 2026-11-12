"""Генератор синтетичної панелі «працівник × місяць × KPI» з відомою істиною.

Навіщо він потрібен
-------------------
У реальній базі проєкту є лише кілька місяців даних, а аномалії в них ніхто
не розмічав. Тому довести, що модель знаходить *справжні* аномалії, на
реальних даних неможливо. Симулятор дає те, чого немає в житті:

* ми **знаємо**, де саме вставлена аномалія (мітка ``anomaly``);
* ми **знаємо**, які відхилення пояснюються сезоном чи зовнішньою подією
  (це НЕ аномалії, мітка 0) — ним керує приховане «справжнє» очікування ``truth``;
* можемо рахувати точність/повноту/FDR і покриття інтервалів.

Усе, що модель бачить на вході, — це лише ``q`` (спостережені виконання),
події та календар. Приховані величини (``truth``, ``anomaly``) використовуються
виключно для оцінювання.

Генеративна модель (для кожного працівника e, KPI k, місяця t)::

    q[e,k,t] = 100 + a_e + s_{e,k} + slope_e * t           # рівень працівника
               + l_k * u[e,t]                              # AR(1) «форма» місяця
               + D[pos(e), k, month(t)]                    # справжня сезонність
               + sum_j  beta[type_j, pos(e)] * m_j * rho^{lag}   # події
               + b[branch(e), t]                           # шок відділення
               + sigma_{pos,k} * c_e * Student_t(nu=5)     # шум із важкими хвостами
               + anomaly[e,k,t]                            # вставлена аномалія

``q`` — виконання плану **вже вирівняне за напрямком** (більше = краще) і
сезонно скориговане номінальною таблицею зі специфікації. Тому «справжня
сезонність» ``D`` — це те, що номінальна таблиця **не** врахувала; саме її
має навчитися модель (ручна таблиця ``SeasonalProfile`` неточна за означенням).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# --- Предметна область (дзеркалить seed_demo; узгодженість перевіряє тест) -----

EVENT_TYPES: List[str] = [
    "RATE_CHANGE",
    "CURRENCY_SHOCK",
    "LOCKDOWN",
    "MARTIAL_LAW",
    "COMPETITOR_EVENT",
    "REGULATION_CHANGE",
    "MACRO_OTHER",
]
N_EVENT_TYPES = len(EVENT_TYPES)

# Номінальні сезонні коефіцієнти зі специфікації §4.1.
NOMINAL_SEASON: Dict[int, float] = {
    1: 0.80, 2: 0.88, 3: 1.00, 4: 1.10, 5: 1.10, 6: 1.00,
    7: 0.90, 8: 0.87, 9: 1.05, 10: 1.12, 11: 1.17, 12: 1.25,
}

KMAX = 5  # максимум KPI на посаду


@dataclass(frozen=True)
class PositionSpec:
    key: str
    name: str
    weights: Tuple[float, ...]
    directions: Tuple[str, ...]  # "higher_better" | "lower_better"

    @property
    def k(self) -> int:
        return len(self.weights)


H, L = "higher_better", "lower_better"
POSITIONS: List[PositionSpec] = [
    PositionSpec("cashier", "Касир", (0.25, 0.20, 0.25, 0.15, 0.15), (H, L, L, H, L)),
    PositionSpec("credit_officer", "Кредитний спеціаліст", (0.20, 0.20, 0.30, 0.15, 0.15), (H, H, L, L, H)),
    PositionSpec("client_manager", "Менеджер по роботі з клієнтами", (0.25, 0.25, 0.20, 0.20, 0.10), (H, H, H, L, H)),
    PositionSpec("branch_manager", "Керівник відділення", (0.30, 0.20, 0.20, 0.15, 0.15), (H, H, H, L, L)),
    PositionSpec("hr_specialist", "Кадровий спеціаліст", (0.25, 0.25, 0.25, 0.25), (H, L, L, L)),
    PositionSpec("accountant", "Бухгалтер / Фінансовий спеціаліст", (0.30, 0.25, 0.25, 0.20), (H, L, L, L)),
    PositionSpec("fx_specialist", "Спеціаліст з валютних операцій", (0.30, 0.30, 0.20, 0.20), (H, H, H, L)),
]
N_POS = len(POSITIONS)


_WEIGHTS_OVERRIDE = None


def set_domain(weights: "np.ndarray | None") -> None:
    """Підміняє матрицю ваг на іншу предметну область (напр., реальні дані Rossmann).

    ``weights`` має форму [n_груп, KMAX]; ``None`` повертає банківські ваги за замовчуванням.
    """
    global _WEIGHTS_OVERRIDE
    _WEIGHTS_OVERRIDE = None if weights is None else np.asarray(weights, dtype=np.float32)


import contextlib
import threading

_DOMAIN_LOCK = threading.RLock()


@contextlib.contextmanager
def use_domain(weights):
    """Тимчасово переключає предметну область (ваги KPI); потокобезпечно, після виходу відновлює попередню."""
    global _WEIGHTS_OVERRIDE
    with _DOMAIN_LOCK:
        prev = _WEIGHTS_OVERRIDE
        set_domain(weights)
        try:
            yield
        finally:
            _WEIGHTS_OVERRIDE = prev


def weight_matrix() -> np.ndarray:
    """Матриця ваг [N_POS, KMAX] (нулі для неіснуючих слотів KPI)."""
    if _WEIGHTS_OVERRIDE is not None:
        out = np.zeros((N_POS, KMAX), dtype=np.float32)
        out[: len(_WEIGHTS_OVERRIDE)] = _WEIGHTS_OVERRIDE
        return out
    w = np.zeros((N_POS, KMAX), dtype=np.float32)
    for i, p in enumerate(POSITIONS):
        w[i, : p.k] = p.weights
    return w


def kpi_slot_mask() -> np.ndarray:
    """Булева матриця [N_POS, KMAX]: який слот KPI існує для посади."""
    m = np.zeros((N_POS, KMAX), dtype=bool)
    for i, p in enumerate(POSITIONS):
        m[i, : p.k] = True
    return m


def composite_score(q: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Композитна оцінка 0–100 за формулою специфікації.

    ``q`` має форму [..., KMAX] (NaN там, де KPI немає/не введено), ``pos`` — [...].
    Нормалізація одного KPI: ``min(100, max(0, q))`` — як у ``spec_normalize_one``.
    Пропущені слоти ігноруються з перенормуванням ваг.
    """
    w = weight_matrix()[pos]
    valid = ~np.isnan(q)
    capped = np.clip(np.nan_to_num(q, nan=0.0), 0.0, 100.0)
    wsum = (w * valid).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (w * valid * capped).sum(-1) / wsum
    out = np.where(wsum > 0, out, np.nan)
    return out


# --- Параметри симуляції --------------------------------------------------------

@dataclass
class SimConfig:
    n_employees: int = 400
    n_months: int = 96          # 8 років
    start_year: int = 2018
    n_branches: int = 8
    seed: int = 7
    anomaly_rate: float = 0.025  # ймовірність початку аномалії для працівника-місяця
    event_rate: float = 0.4      # середня кількість нових подій за місяць (~38 за 8 років)
    ar_phi: float = 0.75
    t_df: float = 5.0
    event_decay: float = 0.5
    hire_frac: float = 0.18      # частка працівників, прийнятих посеред панелі (cold-start)
    term_frac: float = 0.10


@dataclass
class Panel:
    """Результат симуляції. Усі масиви — numpy."""

    cfg: SimConfig
    q: np.ndarray            # [N, T, KMAX] спостережене виконання (NaN = немає)
    pos: np.ndarray          # [N] індекс посади
    branch: np.ndarray       # [N] індекс відділення
    hire_t: np.ndarray       # [N]
    term_t: np.ndarray       # [N] (T = не звільнений)
    events: np.ndarray       # [T, N_EVENT_TYPES] підписані величини подій (зсув у місяці)
    month: np.ndarray        # [T] 1..12
    year: np.ndarray         # [T]
    truth: np.ndarray        # [N, T, KMAX] справжнє очікування без шуму й аномалій
    anomaly: np.ndarray      # [N, T] bool: є вставлена аномалія
    anomaly_kind: np.ndarray  # [N, T] int8: 0 нема, 1 просадка, 2 стрибок, 3 колапс одного KPI
    score: np.ndarray = field(default=None)       # [N, T] композитна оцінка
    truth_score: np.ndarray = field(default=None)  # [N, T] «чисте» очікування композитної оцінки

    @property
    def n(self) -> int:
        return self.q.shape[0]

    @property
    def t(self) -> int:
        return self.q.shape[1]

    def active(self) -> np.ndarray:
        """Булева [N, T]: працівник працює в цей місяць."""
        t_idx = np.arange(self.t)[None, :]
        return (t_idx >= self.hire_t[:, None]) & (t_idx < self.term_t[:, None])


def _event_sensitivity(rng: np.random.Generator) -> np.ndarray:
    """Чутливість посад до типів подій beta[type, pos] (розріджена, зі знаками)."""
    beta = rng.normal(0.0, 1.0, size=(N_EVENT_TYPES, N_POS))
    keep = rng.random(size=beta.shape) < 0.55
    return (beta * keep).astype(np.float32)


def simulate(cfg: SimConfig | None = None) -> Panel:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_employees, cfg.n_months

    # --- Календар
    t_idx = np.arange(T)
    month = (t_idx % 12) + 1
    year = cfg.start_year + t_idx // 12

    # --- Склад працівників
    pos = rng.choice(N_POS, size=N, p=[0.26, 0.17, 0.20, 0.07, 0.07, 0.10, 0.13])
    branch = rng.integers(0, cfg.n_branches, size=N)
    hire_t = np.zeros(N, dtype=int)
    is_hire = rng.random(N) < cfg.hire_frac
    hire_t[is_hire] = rng.integers(1, T - 6, size=int(is_hire.sum()))
    term_t = np.full(N, T, dtype=int)
    is_term = (rng.random(N) < cfg.term_frac) & ~is_hire
    term_t[is_term] = rng.integers(max(12, T // 3), T - 1, size=int(is_term.sum()))

    # --- Зовнішні події: підписана величина у місяці настання
    events = np.zeros((T, N_EVENT_TYPES), dtype=np.float32)
    for t in range(T):
        for _ in range(rng.poisson(cfg.event_rate)):
            j = rng.integers(0, N_EVENT_TYPES)
            sign = rng.choice([-1.0, 1.0])
            events[t, j] += sign * rng.uniform(0.5, 2.0)
    beta = _event_sensitivity(rng)

    # Ефект подій на (посада, місяць) із загасанням за лагами 0..3.
    ev_effect = np.zeros((T, N_POS), dtype=np.float32)
    for lag in range(4):
        shifted = np.zeros_like(events)
        if lag == 0:
            shifted = events
        else:
            shifted[lag:] = events[:-lag]
        ev_effect += (cfg.event_decay ** lag) * (shifted @ beta)
    ev_effect *= 7.0  # масштаб у відсоткових пунктах

    # --- Справжня сезонність (відхилення від номінальної таблиці)
    amp = rng.uniform(3.0, 8.0, size=(N_POS, KMAX))
    phase = rng.uniform(0, 2 * np.pi, size=(N_POS, KMAX))
    m_arr = np.arange(1, 13)
    season_dev = amp[:, :, None] * np.sin(2 * np.pi * m_arr[None, None, :] / 12 + phase[:, :, None])  # [P,K,12]

    # --- Індивідуальні параметри
    ability = rng.normal(0, 6.0, size=N)
    skill = rng.normal(0, 5.0, size=(N, KMAX))
    slope = rng.normal(0, 0.04, size=N)
    loading = rng.uniform(0.7, 1.3, size=KMAX)
    noise_scale = rng.lognormal(0, 0.3, size=N)
    sigma_pk = rng.uniform(3.0, 6.5, size=(N_POS, KMAX))

    # --- Динамічні компоненти
    u = np.zeros((N, T))
    u[:, 0] = rng.normal(0, 3.5, size=N)
    for t in range(1, T):
        u[:, t] = cfg.ar_phi * u[:, t - 1] + rng.normal(0, 3.0, size=N)

    b = np.zeros((cfg.n_branches, T))
    for t in range(1, T):
        b[:, t] = 0.3 * b[:, t - 1] + rng.normal(0, 2.5, size=cfg.n_branches)

    # --- «Справжнє» очікування (без шуму й аномалій)
    truth = np.zeros((N, T, KMAX), dtype=np.float32)
    for e in range(N):
        p = pos[e]
        for t in range(T):
            m = month[t] - 1
            truth[e, t] = (
                100.0
                + ability[e]
                + skill[e]
                + slope[e] * t
                + loading * u[e, t]
                + season_dev[p, :, m]
                + ev_effect[t, p]
                + b[branch[e], t]
            )

    # --- Шум із важкими хвостами
    noise = rng.standard_t(cfg.t_df, size=(N, T, KMAX)) * sigma_pk[pos][:, None, :] * noise_scale[:, None, None]
    q = truth + noise

    # --- Вставлені аномалії
    anomaly = np.zeros((N, T), dtype=bool)
    kind = np.zeros((N, T), dtype=np.int8)
    delta = np.zeros((N, T, KMAX), dtype=np.float32)
    starts = rng.random((N, T)) < cfg.anomaly_rate
    for e in range(N):
        kmax_e = POSITIONS[pos[e]].k
        for t in np.nonzero(starts[e])[0]:
            if t < hire_t[e] + 1 or t >= term_t[e]:
                continue
            r = rng.random()
            if r < 0.45:   # просадка всіх KPI, 1–3 місяці
                dur, mag, kd = rng.integers(1, 4), -rng.uniform(14, 32), 1
                vec = np.full(KMAX, mag) * rng.uniform(0.7, 1.2, size=KMAX)
            elif r < 0.70:  # стрибок
                dur, mag, kd = 1, rng.uniform(12, 24), 2
                vec = np.full(KMAX, mag) * rng.uniform(0.7, 1.2, size=KMAX)
            else:           # колапс одного KPI
                dur, mag, kd = rng.integers(1, 3), -rng.uniform(30, 60), 3
                vec = np.zeros(KMAX)
                vec[rng.integers(0, kmax_e)] = mag
            for d in range(dur):
                tt = t + d
                if tt >= min(T, term_t[e]):
                    break
                delta[e, tt] += vec
                anomaly[e, tt] = True
                kind[e, tt] = kd
    q = q + delta

    # --- Виключаємо слоти, яких немає в посади, та неактивні місяці
    slot = kpi_slot_mask()[pos]  # [N,KMAX]
    t_grid = np.arange(T)[None, :]
    active = (t_grid >= hire_t[:, None]) & (t_grid < term_t[:, None])
    keep = slot[:, None, :] & active[:, :, None]
    q = np.where(keep, np.clip(q, 0.0, 250.0), np.nan).astype(np.float32)
    truth = np.where(keep, np.clip(truth, 0.0, 250.0), np.nan).astype(np.float32)
    anomaly &= active
    kind = np.where(active, kind, 0).astype(np.int8)

    panel = Panel(
        cfg=cfg, q=q, pos=pos, branch=branch, hire_t=hire_t, term_t=term_t,
        events=events, month=month, year=year, truth=truth,
        anomaly=anomaly, anomaly_kind=kind,
    )
    panel.score = composite_score(q, pos[:, None].repeat(T, 1)).astype(np.float32)
    panel.truth_score = composite_score(truth, pos[:, None].repeat(T, 1)).astype(np.float32)
    return panel
