"""Контекст Copilot для РЕАЛЬНОГО датасету Rossmann (магазини замість працівників)."""
from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .core.rossmann import DATA_DIR, ROSSMANN_WEIGHTS, SPLITS_WEEKLY, build_panel
from .core.rossmann_model import ROSS_DIR

TAIL_WEEKS = 26     # скільки останніх тижнів потрібно для аналізу (12 історії + запас на очищення)
KPI_NAMES = ["Продажі за відкритий день", "Клієнти за відкритий день", "Середній чек", "Частка відкритих днів"]


def available() -> bool:
    return (ROSS_DIR / "forecaster.pt").exists() and (DATA_DIR / "train.csv").exists()


@functools.lru_cache(maxsize=1)
def _arrays() -> Dict[str, Any]:
    panel = build_panel(inject=False)
    T = panel.t
    lo = T - TAIL_WEEKS
    q = panel.q[:, lo:]
    active = ~np.isnan(q).all(-1)
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=["Date"], parse_dates=["Date"])
    last_day = df["Date"].max()
    store_ids = np.sort(pd.read_csv(DATA_DIR / "store.csv")["Store"].unique())
    # Останній тиждень, де відкрито достатньо магазинів (щоб порівняння з колегами мало сенс).
    t = TAIL_WEEKS - 1
    while t > 12 and active[:, t].sum() < 0.5 * active.sum(0).max():
        t -= 1
    return {
        "q": q, "pos": panel.pos, "branch": panel.branch, "events": panel.events[:, lo:],
        "month": panel.month[lo:], "active": active, "t": t,
        "emp_index": {int(s): i for i, s in enumerate(store_ids)},
        "label": f"тиждень, що закінчується {(last_day - pd.Timedelta(days=7 * (TAIL_WEEKS - 1 - t))).date()}",
        "entity": "Магазин", "cache_key": ("rossmann", t),
    }


class RossmannContext:
    """Усі автентифіковані (крім адміністратора) бачать усі магазини: даних про людей тут немає."""

    def __init__(self, user) -> None:
        self.user = user

    def period_arrays(self, year, month):
        return _arrays()

    def can_access(self, emp_id: int) -> bool:
        return not self.user.is_system_admin

    def default_employee(self) -> Optional[int]:
        return None


def example_store() -> Optional[int]:
    a = _arrays()
    idx = np.nonzero(a["active"][:, a["t"]])[0]
    ids = {i: s for s, i in a["emp_index"].items()}
    return ids[int(idx[0])] if len(idx) else None


def kpi_names(_store_id: int) -> List[str]:
    return KPI_NAMES
