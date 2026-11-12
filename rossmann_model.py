"""Навчання Copilot на РЕАЛЬНИХ даних Rossmann і збереження моделі.

На відміну від ``evaluate.py`` (бенчмарк із вставленими аномаліями), тут модель навчається
на справжніх даних БЕЗ будь-яких вставок і зберігається в ``artifacts/rossmann/`` — її потім
використовує асистент (датасет «rossmann»).

Запуск::  python manage.py copilot_train_rossmann
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cleaning import sequential_predict
from .conformal import MondrianConformal, weighted_quantile, decay_weights
from .evaluate import CAL_WINDOW, adjust_common_shock, hgb_predict, y_index
from .features import dataset_from_panel
from .rossmann import ROSSMANN_WEIGHTS, SPLITS_WEEKLY, build_panel
from .simulator import set_domain
from .train import ARTIFACTS, TrainConfig, save_model, train_model

ROSS_DIR = ARTIFACTS / "rossmann"


def train_and_save(log=print, epochs: int = 30, seed: int = 0) -> str:
    set_domain(ROSSMANN_WEIGHTS)
    panel = build_panel(inject=False)           # реальні дані, нічого не вставляємо
    ds = dataset_from_panel(panel, 12)
    lo, hi = SPLITS_WEEKLY["test"]
    parts = {k: ds.time_range(*v) for k, v in SPLITS_WEEKLY.items()}
    log(f"Rossmann: {panel.n} магазинів, {panel.t} тижнів, навчальних вікон {len(parts['train'])}")

    cfg = TrainConfig(seed=seed, epochs=epochs, patience=6)
    model = train_model(
    parts["train"],
    parts["val"],
    cfg,
    log=log,
    checkpoint_path=ROSS_DIR / "training_checkpoint.pt",
)

    seq_from = SPLITS_WEEKLY["calib"][0] - CAL_WINDOW
    ev = ds.time_range(seq_from, hi)
    seq = sequential_predict(model, ev, seq_from, clean=True)
    adj = adjust_common_shock(ev, seq["signed"])
    cal_sel = ev.t >= lo - CAL_WINDOW
    last = int(ev.t.max())
    mc = MondrianConformal(half_life=8.0).fit(np.abs(adj[cal_sel]), ev.pos[cal_sel], last - ev.t[cal_sel])

    ROSS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, cfg, ROSS_DIR / "forecaster.pt")
    (ROSS_DIR / "conformal.json").write_text(json.dumps(mc.to_json()), encoding="utf-8")
    (ROSS_DIR / "meta.json").write_text(json.dumps({
        "weights": ROSSMANN_WEIGHTS.tolist(), "dataset": "Rossmann Store Sales (Kaggle)",
        "n_stores": int(panel.n), "n_weeks": int(panel.t), "train_weeks": list(SPLITS_WEEKLY["train"]),
        "kpis": ["Продажі за відкритий день", "Клієнти за відкритий день", "Середній чек", "Частка відкритих днів"],
    }, ensure_ascii=False), encoding="utf-8")

    # --- Чесна оцінка на реальних тижнях, яких модель не бачила (тест) ---
    test = ev.t >= lo
    y = y_index(ev)
    mu = seq["mu_idx"]
    mae_nn = float(np.mean(np.abs(y[test] - mu[test])))
    # Наївний сезонний базлайн: значення того ж тижня рік тому (52 тижні), якщо є.
    from .evaluate import _history_series  # noqa: F401  (імпорт для узгодженості)
    idx = {(int(e), int(t)): float(v) for e, t, v in zip(ev.emp, ev.t, y)}
    naive = [(abs(idx[(int(e), int(t))] - idx[(int(e), int(t) - 52)]))
             for e, t in zip(ev.emp[test], ev.t[test]) if (int(e), int(t) - 52) in idx]
    prev = {(int(e), int(t)): float(v) for e, t, v in zip(ev.emp, ev.t, y)}
    naive_last = [abs(prev[(int(e), int(t))] - prev[(int(e), int(t) - 1)])
                  for e, t in zip(ev.emp[test], ev.t[test]) if (int(e), int(t) - 1) in prev]
    mu_h, sig_h = hgb_predict(parts["train"], ev, seed)
    mae_h = float(np.mean(np.abs(y[test] - mu_h[test])))

    # Покриття 90% інтервалу (conformal на попередніх 12 тижнях) на реальних тижнях тесту.
    surprise = np.abs(adj)
    cover = []
    for t in range(lo, hi):
        cal = (ev.t >= t - CAL_WINDOW) & (ev.t < t)
        q = weighted_quantile(surprise[cal], 0.9, decay_weights(t - ev.t[cal], 8.0))
        cur = ev.t == t
        cover.append((surprise[cur] <= q).mean())
    report = (
        "## Реальні дані Rossmann: прогноз індексу виконання магазину (тест — 30 тижнів, яких модель не бачила)\n\n"
        "| Метод | MAE індексу (п.п.) |\n|---|---|\n"
        f"| «Як минулого тижня» | {np.mean(naive_last):.2f} |\n"
        f"| «Як рік тому» (52 тижні) | {np.mean(naive) if naive else float('nan'):.2f} |\n"
        f"| Бустинг (ті самі ознаки) | {mae_h:.2f} |\n"
        f"| **Нейромережа Copilot** | **{mae_nn:.2f}** |\n\n"
        f"Покриття 90 % conformal-інтервалу на реальних тижнях: **{np.mean(cover):.3f}** (ціль 0,900).\n"
    )
    (ROSS_DIR / "report.md").write_text(report, encoding="utf-8")
    log(report)
    set_domain(None)
    return report
