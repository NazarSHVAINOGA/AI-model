"""Реальний бенчмарк: Rossmann Store Sales (Kaggle) → панель «одиниця × тиждень × KPI».

Чому цей датасет
---------------
Реальних даних про KPI працівників банку у відкритому доступі немає. Rossmann —
реальна панель 1115 магазинів з щоденними продажами (2013-01 … 2015-07), де є ті
самі структурні елементи, що й у нашій задачі:

=====================  ==========================================================
Наша система           Rossmann
=====================  ==========================================================
працівник/відділення   магазин
посада                 тип магазину (a, b, c, d)  → страта Mondrian-conformal
відділення             асортимент (a, b, c)       → група колег
KPI                    продажі/відкритий день, клієнти/день, середній чек, частка відкритих днів
план                   середнє значення магазину за навчальний період
сезонність             тиждень року
зовнішні події         промо-дні, державні й шкільні свята (СВОЇ для кожного магазину)
=====================  ==========================================================

Чого тут НЕМАЄ: розмічених аномалій. Тому
1. прогнозування (MAE, покриття інтервалів) оцінюємо на *чистих реальних* даних;
2. для детекції вставляємо контрольовані аномалії в реальні ряди (semi-synthetic):
   мітки точні, а фон — реальний. У дипломі це треба так і називати.

Завантаження (потрібен токен Kaggle, див. README)::

    kaggle datasets download -d pratyushakar/rossmann-store-sales -p data/rossmann --unzip
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .simulator import KMAX, N_EVENT_TYPES, Panel, SimConfig, composite_score, set_domain

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "rossmann"
# Ваги чотирьох KPI: продажі, клієнти, середній чек, частка відкритих днів.
ROSSMANN_WEIGHTS = np.array([[0.40, 0.25, 0.20, 0.15, 0.0]], dtype=np.float32).repeat(4, axis=0)
SPLITS_WEEKLY = {"train": (0, 80), "val": (80, 88), "calib": (88, 104), "test": (104, 134)}


def load_weekly(data_dir: Path = DATA_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Повертає (q_raw, pos, branch, events, phase).

    q_raw  [N, T, KMAX] — значення KPI (NaN для закритих тижнів);
    events [N, T, 7]    — подієві ознаки для кожного магазину;
    phase  [T]          — «псевдомісяць» 1..13 (частка року × 12 + 1) для сезонних ознак.
    """
    df = pd.read_csv(data_dir / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"])
    store = pd.read_csv(data_dir / "store.csv")
    df["week"] = ((df["Date"] - pd.Timestamp("2012-12-31")).dt.days // 7).astype(int)
    df["open"] = (df["Open"] == 1) & (df["Sales"] > 0)
    df["sales_open"] = np.where(df["open"], df["Sales"], 0.0)
    df["cust_open"] = np.where(df["open"], df["Customers"], 0.0)
    df["promo_open"] = np.where(df["open"], df["Promo"], 0)
    df["state_hol"] = (df["StateHoliday"].fillna("0") != "0").astype(int)
    g = df.groupby(["Store", "week"]).agg(
        open_days=("open", "sum"), sales=("sales_open", "sum"), cust=("cust_open", "sum"),
        promo=("promo_open", "sum"), state_hol=("state_hol", "sum"), school=("SchoolHoliday", "sum"),
        days=("open", "size"),
    ).reset_index()
    # Перші/останні неповні тижні відкидаємо.
    g = g[g["days"] == 7]
    stores = np.sort(store["Store"].unique())
    sid = {s: i for i, s in enumerate(stores)}
    T = int(g["week"].max()) + 1
    N = len(stores)
    q = np.full((N, T, KMAX), np.nan, np.float32)
    ev = np.zeros((N, T, N_EVENT_TYPES), np.float32)
    i = g["Store"].map(sid).to_numpy()
    t = g["week"].to_numpy()
    od = g["open_days"].to_numpy()
    ok = od >= 3                       # тижні з ремонтом/закриттям вважаємо «неактивними»
    with np.errstate(invalid="ignore", divide="ignore"):
        s_pd = g["sales"].to_numpy() / od
        c_pd = g["cust"].to_numpy() / od
        basket = g["sales"].to_numpy() / g["cust"].to_numpy()
        share = od / 6.0
    q[i[ok], t[ok], 0] = s_pd[ok]
    q[i[ok], t[ok], 1] = c_pd[ok]
    q[i[ok], t[ok], 2] = basket[ok]
    q[i[ok], t[ok], 3] = share[ok]
    # Події: частка промо-днів, державні свята, шкільні свята (масштаб ~[0, 2]).
    ev[i, t, 0] = g["promo"].to_numpy() / 7.0 * 2.0
    ev[i, t, 1] = g["state_hol"].to_numpy().astype(float)
    ev[i, t, 2] = g["school"].to_numpy() / 7.0 * 2.0
    # Рівні «позицій» і «відділень».
    stype = store.set_index("Store").loc[stores, "StoreType"].map({"a": 0, "b": 1, "c": 2, "d": 3}).to_numpy()
    assort = store.set_index("Store").loc[stores, "Assortment"].map({"a": 0, "b": 1, "c": 2}).to_numpy()
    week_start = pd.Timestamp("2012-12-31") + pd.to_timedelta(np.arange(T) * 7 + 3, unit="D")
    phase = (week_start.dayofyear.to_numpy() - 1) / 365.0 * 12 + 1
    return q, stype.astype(int), assort.astype(int), ev, phase.astype(np.float32)


def to_attainment(q_raw: np.ndarray, plan_until: int) -> np.ndarray:
    """Нормалізація «факт/план»: план = середнє магазину за тижні < ``plan_until``, ×100."""
    plan = np.nanmean(q_raw[:, :plan_until, :], axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 100.0 * q_raw / plan
    return np.clip(q, 0.0, 250.0).astype(np.float32)


def inject_anomalies(q: np.ndarray, rate: float, seed: int):
    """Вставляє контрольовані аномалії в реальні ряди; повертає (q', мітки, тип)."""
    rng = np.random.default_rng(seed)
    N, T, _ = q.shape
    out = q.copy()
    lab = np.zeros((N, T), bool)
    kind = np.zeros((N, T), np.int8)
    active = ~np.isnan(q).all(-1)
    starts = rng.random((N, T)) < rate
    for e in range(N):
        for t in np.nonzero(starts[e])[0]:
            if t < 13 or not active[e, t]:
                continue
            r = rng.random()
            if r < 0.45:
                dur, kd = int(rng.integers(1, 4)), 1
                vec = -rng.uniform(15, 35) * rng.uniform(0.8, 1.2, size=KMAX)
            elif r < 0.70:
                dur, kd = 1, 2
                vec = rng.uniform(15, 30) * rng.uniform(0.8, 1.2, size=KMAX)
            else:
                dur, kd = int(rng.integers(1, 3)), 3
                vec = np.zeros(KMAX)
                vec[int(rng.integers(0, 3))] = -rng.uniform(35, 60)
            for d in range(dur):
                tt = t + d
                if tt >= T or not active[e, tt]:
                    break
                out[e, tt] = np.clip(out[e, tt] + vec, 0.0, 250.0)
                lab[e, tt] = True
                kind[e, tt] = kd
    out = np.where(np.isnan(q), np.nan, out)
    return out.astype(np.float32), lab, kind


def build_panel(inject: bool, rate: float = 0.025, seed: int = 0, data_dir: Path = DATA_DIR) -> Panel:
    """Збирає Panel (формат, який розуміє решта коду) з реальних даних Rossmann."""
    set_domain(ROSSMANN_WEIGHTS)
    q_raw, pos, branch, ev, phase = load_weekly(data_dir)
    q = to_attainment(q_raw, plan_until=SPLITS_WEEKLY["train"][1])
    N, T, _ = q.shape
    if inject:
        q_obs, lab, kind = inject_anomalies(q, rate, seed)
    else:
        q_obs, lab, kind = q, np.zeros((N, T), bool), np.zeros((N, T), np.int8)
    active = ~np.isnan(q_obs).all(-1)
    hire = np.array([int(np.argmax(a)) if a.any() else T for a in active])
    panel = Panel(
        cfg=SimConfig(n_employees=N, n_months=T), q=q_obs, pos=pos, branch=branch,
        hire_t=hire, term_t=np.full(N, T), events=ev, month=phase,
        year=np.zeros(T, int), truth=q, anomaly=lab, anomaly_kind=kind,
    )
    panel.score = composite_score(q_obs, np.repeat(pos[:, None], T, 1)).astype(np.float32)
    panel.truth_score = composite_score(q, np.repeat(pos[:, None], T, 1)).astype(np.float32)
    # «Активність» для Panel.active(): тижні, де є хоч одне значення.
    panel.active = lambda a=active: a  # type: ignore[method-assign]
    return panel
