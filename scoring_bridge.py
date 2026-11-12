"""Місток до чинних правил системи (``apps.analytics.scoring``) без залежності від Django.

``apps/analytics/scoring.py`` — чисті функції, але пакет ``apps.analytics`` при імпорті
не тягне Django, тож їх можна використати напряму. Це гарантує, що базлайни в
експерименті — *саме ті формули*, що працюють у системі, а не їхні перекази.
"""
from __future__ import annotations

import statistics
from typing import Sequence

from apps.analytics.scoring import holt_winters_fit, holt_winters_forecast


def modified_z_current(current: float, history: Sequence[float]) -> float:
    """Модифікований Z поточного значення відносно історії — копія логіки
    ``_robust_individual_anomalies`` (потрібно ≥ 4 точки історії; MAD·1.4826)."""
    if len(history) < 4:
        return 0.0
    med = float(statistics.median(history))
    mad = float(statistics.median([abs(v - med) for v in history])) * 1.4826
    if mad <= 0:
        return 0.0
    return (current - med) / mad


def hw_forecast_next(history: Sequence[float]) -> float:
    """Прогноз Holt-Winters на 1 крок уперед (як ``_forecast_score``: ≥ 24 точки, сезон 12)."""
    state, _ = holt_winters_fit(list(history), season_length=12)
    return holt_winters_forecast(state, horizon=1, season_length=12)[0]
