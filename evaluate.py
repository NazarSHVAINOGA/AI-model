"""Експериментальна перевірка: чи справді модель краща за чинні правила.

Порівнюються методи виявлення аномалій індивідуальної продуктивності:

* ``modz_deployed``   — modified Z-score (|mZ|>3.5) по історії *обрізаної* оцінки, як у ``services.py``;
* ``modz_uncapped``   — той самий метод, але по необрізаному індексу (чесніший варіант);
* ``hw_deployed``     — правило «відхилення від Holt-Winters > 20» зі ``services.py``;
* ``hgb_conformal``   — сильний ML-базлайн: градієнтний бустинг з УСІМА тими ж ознаками
                         (календар, події, колеги) + те саме conformal-калібрування;
* ``nn_*``            — наша нейромережа: повна та абляції (без подій / сезону / колег / історії)
                         + conformal; ``nn_full_bh`` — із контролем FDR (BH).

Чесність експерименту
---------------------
* Дані синтетичні (``simulator.py``): мітки аномалій відомі. Це перевірка на
  *змодельованому* світі, а не доказ продуктивності на реальному банку.
* Модель навчається на «забруднених» даних (аномалії всередині навчальної вибірки, без міток).
* Тест — останні 18 місяців, які модель не бачила; калібрування — ковзне вікно 12 місяців
  перед кожним місяцем (online), тобто лише минуле.
* Базлайни без екзогенних ознак (modZ, HW) за побудовою в гіршому становищі —
  тому додано ``hgb_conformal`` з тими самими ознаками, що й у нейромережі.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from .conformal import MondrianConformal, benjamini_hochberg, decay_weights, weighted_quantile
from .features import Dataset, N_GROUPS
from .model import KpiForecaster, slice_batch, to_tensors
from .cleaning import sequential_predict
from .scoring_bridge import hw_forecast_next, modified_z_current
from .simulator import SimConfig, composite_score, weight_matrix
from .train import ARTIFACTS, SPLITS, TrainConfig, device, prepare, train_model

CAL_WINDOW = 12   # місяців минулого, які йдуть у калібрування
SEQ_FROM_DEFAULT = 54  # (для симуляції) з якого місяця починається послідовний (очищувальний) прохід
TEST_LO, TEST_HI = SPLITS["test"]


# --- Прогнози та сюрпризи ------------------------------------------------------

def y_index(ds: Dataset) -> np.ndarray:
    w = weight_matrix()[ds.pos] * ds.ym
    return (w * ds.y).sum(-1) / np.maximum(w.sum(-1), 1e-6)


@torch.no_grad()
def nn_predict(model: KpiForecaster, ds: Dataset, disabled: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    dev = device()
    b = to_tensors(ds, dev)
    model.to(dev).eval()
    mus, sigs = [], []
    n = len(ds)
    present = torch.ones(n, N_GROUPS, device=dev)
    for g in disabled:
        present[:, g] = 0.0
    for i in range(0, n, 4096):
        idx = torch.arange(i, min(n, i + 4096), device=dev)
        _, _, _, mu_i, sig_i = model(slice_batch(b, idx), present[idx])
        mus.append(mu_i.cpu().numpy())
        sigs.append(sig_i.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def hgb_features(ds: Dataset) -> np.ndarray:
    n = len(ds)
    return np.concatenate(
        [
            ds.hq.reshape(n, -1), ds.hm.reshape(n, -1), ds.hv, ds.hev.reshape(n, -1),
            ds.tcal, ds.tev, ds.peers, ds.pos[:, None], ds.branch[:, None],
        ],
        axis=1,
    )


def hgb_predict(ds_train: Dataset, ds_all: Dataset, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Бустинг: μ та σ̂ (друга модель на |залишку|). Ті самі ознаки, що й у нейромережі."""
    Xtr, ytr = hgb_features(ds_train), y_index(ds_train)
    cat = np.zeros(Xtr.shape[1], dtype=bool)
    cat[-2:] = True
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, categorical_features=cat, random_state=seed)
    m.fit(Xtr, ytr)
    res = np.abs(ytr - m.predict(Xtr))
    ms = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, categorical_features=cat, random_state=seed)
    ms.fit(Xtr, res)
    X = hgb_features(ds_all)
    return m.predict(X), np.maximum(ms.predict(X), 0.5)


def adjust_common_shock(ds: Dataset, signed: np.ndarray, min_group: int = 8) -> np.ndarray:
    """Віднімає медіану сюрпризів групи (посада × місяць) → «індивідуальна» частина.

    Спільний шок (подія, що зачепила всю посаду) зсуває сюрпризи всіх колег в один бік;
    медіана — робастна оцінка цього зсуву (до 50 % аномальних колег її не зрушують).
    Після віднімання лишається те, чим працівник відрізняється від колег тієї ж посади
    в тому ж місяці. Для малих груп (< ``min_group``) поправка не застосовується.
    """
    out = signed.copy()
    for t in np.unique(ds.t):
        for pos in np.unique(ds.pos[ds.t == t]):
            sel = np.nonzero((ds.t == t) & (ds.pos == pos))[0]
            if len(sel) >= min_group:
                out[sel] = signed[sel] - np.nanmedian(signed[sel])
    return out


def rolling_conformal(
    ds: Dataset, surprise: np.ndarray, half_life: float = 8.0,
    test_lo: int = TEST_LO, test_hi: int = TEST_HI,
) -> np.ndarray:
    """Online p-значення для кожного місяця тесту: калібрування = попередні CAL_WINDOW місяців."""
    p = np.ones(len(ds))
    for t in range(test_lo, test_hi):
        cal = (ds.t >= t - CAL_WINDOW) & (ds.t < t)
        cur = ds.t == t
        if cur.sum() == 0:
            continue
        mc = MondrianConformal(half_life=half_life).fit(surprise[cal], ds.pos[cal], t - ds.t[cal])
        p[cur] = mc.pvalues(surprise[cur], ds.pos[cur])
    return p


def conformal_band_coverage(
    ds: Dataset, y: np.ndarray, mu: np.ndarray, sig: np.ndarray, alpha: float, half_life: float = 8.0,
    test_lo: int = TEST_LO, test_hi: int = TEST_HI,
) -> Dict[str, np.ndarray]:
    """Покриття інтервалів μ ± q̂σ̂ (conformal) та μ ± z·σ̂ (наївний гаусів)."""
    from scipy.stats import norm
    surprise = np.abs(y - mu) / sig
    cov_conf = np.full(len(ds), np.nan)
    for t in range(test_lo, test_hi):
        cal = (ds.t >= t - CAL_WINDOW) & (ds.t < t)
        cur = np.nonzero(ds.t == t)[0]
        for pos in np.unique(ds.pos[cur]):
            c = cal & (ds.pos == pos)
            q = weighted_quantile(surprise[c], 1 - alpha, decay_weights(t - ds.t[c], half_life)) if c.sum() >= 50 \
                else weighted_quantile(surprise[cal], 1 - alpha, decay_weights(t - ds.t[cal], half_life))
            sel = cur[ds.pos[cur] == pos]
            cov_conf[sel] = (surprise[sel] <= q).astype(float)
    cov_gauss = (surprise <= norm.ppf(1 - alpha / 2)).astype(float)
    return {"conformal": cov_conf, "gaussian": cov_gauss}


# --- Метрики детекції ------------------------------------------------------------

def detection_metrics(label: np.ndarray, flag: np.ndarray, score: Optional[np.ndarray]) -> Dict[str, float]:
    tp = int((flag & label).sum())
    fp = int((flag & ~label).sum())
    fn = int((~flag & label).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    out = {
        "flagged": int(flag.sum()), "precision": prec, "recall": rec, "f1": f1,
        "fdp": 1 - prec if tp + fp else float("nan"),
        "false_alarm_rate": fp / max(1, int((~label).sum())),
    }
    if score is not None and 0 < label.sum() < len(label):
        out["auroc"] = float(roc_auc_score(label, score))
        out["auprc"] = float(average_precision_score(label, score))
    return out


def _bh_by_month(ds: Dataset, p: np.ndarray, q: float) -> np.ndarray:
    flag = np.zeros(len(ds), dtype=bool)
    for t in np.unique(ds.t):
        sel = np.nonzero(ds.t == t)[0]
        flag[sel] = benjamini_hochberg(p[sel], q)
    return flag


def _history_series(panel, ds: Dataset, capped: bool) -> List[np.ndarray]:
    """Для кожного прикладу ds: ряд оцінок працівника за попередні 24 місяці."""
    if capped:
        series = panel.score
    else:
        w = weight_matrix()[panel.pos][:, None, :]
        qq = np.where(np.isnan(panel.q), 0.0, panel.q)
        valid = ~np.isnan(panel.q)
        series = (w * valid * qq).sum(-1) / np.maximum((w * valid).sum(-1), 1e-6)
        series = np.where(valid.any(-1), series, np.nan)
    out = []
    for e, t in zip(ds.emp, ds.t):
        s = series[e, max(0, t - 24):t]
        out.append(s[~np.isnan(s)])
    return out


# --- Основний експеримент ------------------------------------------------------------

def run_experiment(
    seed: int = 0,
    sim_cfg: Optional[SimConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    log=print,
    skip_ablations: bool = False,
    save_main: bool = False,
) -> Dict:
    sim_cfg = replace(sim_cfg or SimConfig(), seed=100 + seed)
    train_cfg = replace(train_cfg or TrainConfig(), seed=seed)
    panel, ds, parts = prepare(sim_cfg, train_cfg.window)
    return run_on_panel(panel, ds, SPLITS, seed, train_cfg, log, skip_ablations, save_main)


def run_on_panel(
    panel, ds: Dataset, splits: Dict[str, tuple], seed: int, train_cfg: TrainConfig,
    log=print, skip_ablations: bool = False, save_main: bool = False,
) -> Dict:
    """Повний експеримент на будь-якій панелі (синтетичній або реальній з вставленими аномаліями)."""
    t0 = time.time()
    parts = {name: ds.time_range(lo, hi) for name, (lo, hi) in splits.items()}
    TEST_LO, TEST_HI = splits["test"]
    SEQ_FROM = splits["calib"][0] - CAL_WINDOW
    ds_eval = ds.time_range(SEQ_FROM, TEST_HI)   # історія для очищення + калібрування + тест
    y = y_index(ds_eval)
    truth_idx = _truth_index(panel, ds_eval)
    in_test = ds_eval.t >= TEST_LO
    label_all = panel.anomaly[ds_eval.emp, ds_eval.t]
    results: Dict[str, Dict] = {}
    log(f"[seed {seed}] дані готові за {time.time() - t0:.0f}s; приклади: "
        f"train={len(parts['train'])} test={int(in_test.sum())}; частка аномалій у тесті = "
        f"{label_all[in_test].mean():.3%}")

    prev_anom = np.array([panel.anomaly[e, t - 1] if t > 0 else False for e, t in zip(ds_eval.emp, ds_eval.t)])
    first_month = label_all & ~prev_anom
    cont_month = label_all & prev_anom
    # Події, що «пояснюють» відхилення: |подія| >= 1 у цільовому місяці або в 3 попередніх.
    ev_mag = np.abs(panel.events).sum(-1)
    if ev_mag.ndim == 2:
        ev_mag = ev_mag.mean(0)
    ev_mag = ev_mag if ev_mag.ndim == 1 else ev_mag.mean(0)
    ev_recent = np.array([ev_mag[max(0, t - 3): t + 1].max() for t in range(panel.t)])
    event_month = ev_recent[ds_eval.t] >= 1.0

    def split_auroc(entry: Dict, surprise: np.ndarray) -> None:
        """AUROC окремо для першого місяця епізоду та його продовження."""
        for tag, pos_mask in (("first", first_month), ("cont", cont_month)):
            mm = in_test & (~label_all | pos_mask)
            entry[f"auroc_{tag}"] = float(roc_auc_score(pos_mask[mm], surprise[mm]))

    def evaluate(name: str, mu: np.ndarray, sig: np.ndarray, signed: np.ndarray, with_bh: bool = False) -> None:
        surprise = np.abs(signed)
        p = rolling_conformal(ds_eval, surprise, test_lo=TEST_LO, test_hi=TEST_HI)
        m, lab = in_test, label_all[in_test]
        ok = in_test & ~label_all
        entry = {
            "alpha_0.01": detection_metrics(lab, (p <= 0.01)[m], surprise[m]),
            "alpha_0.05": detection_metrics(lab, (p <= 0.05)[m], surprise[m]),
            "mae_vs_truth": float(np.mean(np.abs(mu[ok] - truth_idx[ok]))),
            "fa_rate_event_months": float((p <= 0.01)[ok & event_month].mean()),
            "fa_rate_calm_months": float((p <= 0.01)[ok & ~event_month].mean()),
        }
        split_auroc(entry, surprise)
        cov = conformal_band_coverage(ds_eval, y, mu, sig, alpha=0.1, test_lo=TEST_LO, test_hi=TEST_HI)
        entry["coverage_90_conformal"] = float(np.nanmean(cov["conformal"][ok]))
        entry["coverage_90_gaussian"] = float(np.nanmean(cov["gaussian"][ok]))
        if with_bh:
            for q in (0.05, 0.10):
                flag = _bh_by_month(ds_eval.select(m), p[m], q)
                entry[f"bh_q{q}"] = detection_metrics(lab, flag, None)
        results[name] = entry

    # --- 1. Нейромережі: повна (з очищенням і без) та абляції
    variants = {"nn_full": ()}
    if not skip_ablations:
        variants.update({"nn_no_events": (2,), "nn_no_peers": (3,), "nn_no_calendar": (1,), "nn_no_history": (0,)})
    for name, disabled in variants.items():
        cfg = replace(train_cfg, disabled_groups=disabled)
        model = train_model(parts["train"], parts["val"], cfg, log=None)
        seq = sequential_predict(model, ds_eval, SEQ_FROM, clean=True, disabled=disabled)
        mu, sig, signed = seq["mu_idx"], seq["sig_idx"], seq["signed"]
        evaluate(name, mu, sig, adjust_common_shock(ds_eval, signed), with_bh=(name == "nn_full"))
        if name == "nn_full":
            raw = sequential_predict(model, ds_eval, SEQ_FROM, clean=False)
            evaluate("nn_raw_no_cleaning", raw["mu_idx"], raw["sig_idx"], adjust_common_shock(ds_eval, raw["signed"]))
            evaluate("nn_no_group_adjust", mu, sig, signed, with_bh=True)
            if save_main:
                _save_main(model, train_cfg, panel, ds_eval, seq, adjust_common_shock(ds_eval, signed), TEST_LO, TEST_HI)
        log(f"[seed {seed}] {name} ({time.time() - t0:.0f}s)")

    # --- 2. Бустинг із тими ж ознаками (з тією самою поправкою на спільний шок)
    mu_h, sig_h = hgb_predict(parts["train"], ds_eval, seed)
    signed_h = (y - mu_h) / sig_h
    evaluate("hgb_conformal", mu_h, sig_h, adjust_common_shock(ds_eval, signed_h), with_bh=True)
    log(f"[seed {seed}] hgb_conformal")

    # --- 3. Правила з чинної системи
    cur_capped = panel.score[ds_eval.emp, ds_eval.t]
    for name, capped, cur in (("modz_deployed", True, cur_capped), ("modz_uncapped", False, y)):
        hist = _history_series(panel, ds_eval, capped)
        mz = np.array([modified_z_current(float(c), h[-12:]) for c, h in zip(cur, hist)])
        flag = np.abs(mz) > 3.5
        m, ok = in_test, in_test & ~label_all
        entry = {"rule_3.5": detection_metrics(label_all[m], flag[m], np.abs(mz)[m]),
                 "fa_rate_event_months": float(flag[ok & event_month].mean()),
                 "fa_rate_calm_months": float(flag[ok & ~event_month].mean())}
        split_auroc(entry, np.abs(mz))
        results[name] = entry
    hist = _history_series(panel, ds_eval, True)
    dev = np.array([abs(c - hw_forecast_next(h)) if len(h) >= 24 else 0.0 for c, h in zip(cur_capped, hist)])
    ok = in_test & ~label_all
    results["hw_deployed"] = {
        "rule_20": detection_metrics(label_all[in_test], (dev > 20)[in_test], dev[in_test]),
        "fa_rate_event_months": float((dev > 20)[ok & event_month].mean()),
        "fa_rate_calm_months": float((dev > 20)[ok & ~event_month].mean()),
    }
    log(f"[seed {seed}] завершено за {time.time() - t0:.0f}s")
    return results


def _truth_index(panel, ds: Dataset) -> np.ndarray:
    w = weight_matrix()[ds.pos]
    tr = panel.truth[ds.emp, ds.t]
    valid = ~np.isnan(tr)
    return (w * valid * np.nan_to_num(tr)).sum(-1) / np.maximum((w * valid).sum(-1), 1e-6)


def _save_main(model, train_cfg, panel, ds_eval, seq, adj_signed, TEST_LO=TEST_LO, TEST_HI=TEST_HI) -> None:
    """Зберігає основну модель і калібрувальну множину для продакшн-використання."""
    from .train import save_model
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    save_model(model, train_cfg, ARTIFACTS / "forecaster.pt")
    sel = (ds_eval.t >= TEST_LO - CAL_WINDOW) & (ds_eval.t < TEST_HI)
    last = ds_eval.t[sel].max()
    mc = MondrianConformal(half_life=8.0).fit(np.abs(adj_signed[sel]), ds_eval.pos[sel], last - ds_eval.t[sel])
    (ARTIFACTS / "conformal.json").write_text(json.dumps(mc.to_json()), encoding="utf-8")


# --- Агрегація за seed-ами та звіт -----------------------------------------------------

def _flatten(d: Dict, prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out


def aggregate(runs: List[Dict]) -> Dict[str, Dict[str, tuple]]:
    """Середнє ± стандартне відхилення за seed-ами для кожної метрики."""
    flat = [_flatten(r) for r in runs]
    keys = sorted({k for f in flat for k in f})
    out: Dict[str, tuple] = {}
    for k in keys:
        vals = np.array([f[k] for f in flat if k in f], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out[k] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals))
    return out


def markdown_report(agg: Dict[str, tuple], n_seeds: int) -> str:
    def g(key: str, nd: int = 3) -> str:
        if key not in agg:
            return "—"
        m, s, _ = agg[key]
        return f"{m:.{nd}f} ± {s:.{nd}f}"

    rows = [
        ("modified Z-score (як у системі, обрізана оцінка)", "modz_deployed.rule_3.5"),
        ("modified Z-score (необрізаний індекс)", "modz_uncapped.rule_3.5"),
        ("Holt-Winters, відхилення > 20 (як у системі)", "hw_deployed.rule_20"),
        ("Бустинг + conformal (α = 0.01)", "hgb_conformal.alpha_0.01"),
        ("Бустинг + conformal + BH (q = 0.10)", "hgb_conformal.bh_q0.1"),
        ("**Нейромережа + conformal (α = 0.01)**", "nn_full.alpha_0.01"),
        ("**Нейромережа + conformal + BH (q = 0.10)**", "nn_full.bh_q0.1"),
    ]
    L = [f"## Виявлення аномалій (тест на невидимих даних, {n_seeds} незалежних seed, середнє ± ст. відхилення)\n",
         "| Метод | Precision | Recall | F1 | FDP (частка хибних) | AUROC |", "|---|---|---|---|---|---|"]
    for title, key in rows:
        L.append(f"| {title} | {g(key + '.precision')} | {g(key + '.recall')} | {g(key + '.f1')} | "
                 f"{g(key + '.fdp')} | {g(key + '.auroc')} |")
    L += ["", "## Хибні тривоги серед НЕ-аномалій: «місяці з подіями» vs «спокійні місяці»\n",
          "| Метод | у місяцях з подіями | у спокійних місяцях |", "|---|---|---|"]
    for title, key in [
        ("modified Z-score (як у системі)", "modz_deployed"),
        ("modified Z-score (необрізаний)", "modz_uncapped"),
        ("Бустинг + conformal α=0.01", "hgb_conformal"),
        ("Нейромережа + conformal α=0.01", "nn_full"),
    ]:
        L.append(f"| {title} | {g(key + '.fa_rate_event_months', 4)} | {g(key + '.fa_rate_calm_months', 4)} |")
    L += ["", "## Покриття 90 % інтервалу на нормальних спостереженнях (ціль = 0.900)\n",
          "| Модель | Наївний гаусів інтервал | Conformal-інтервал | MAE до «істини» |", "|---|---|---|---|"]
    for title, key in [("Бустинг", "hgb_conformal"), ("Нейромережа", "nn_full"),
                       ("НМ без подій", "nn_no_events"), ("НМ без колег", "nn_no_peers"),
                       ("НМ без календаря", "nn_no_calendar"), ("НМ без історії", "nn_no_history")]:
        L.append(f"| {title} | {g(key + '.coverage_90_gaussian')} | {g(key + '.coverage_90_conformal')} | "
                 f"{g(key + '.mae_vs_truth')} |")
    L += ["", "## Абляція: внесок груп ознак (AUROC / F1 при α=0.01)\n", "| Варіант | AUROC | F1 |", "|---|---|---|"]
    for title, key in [("повна модель", "nn_full"), ("без подій", "nn_no_events"), ("без колег", "nn_no_peers"),
                       ("без календаря", "nn_no_calendar"), ("без власної історії", "nn_no_history")]:
        L.append(f"| {title} | {g(key + '.alpha_0.01.auroc')} | {g(key + '.alpha_0.01.f1')} |")
    return "\n".join(L)


def run_all(seeds: int = 3, log=print, out_dir: Path = ARTIFACTS) -> str:
    runs = []
    for s in range(seeds):
        runs.append(run_experiment(seed=s, log=log, save_main=(s == 0)))
    agg = aggregate(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_raw.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    md = markdown_report(agg, seeds)
    (out_dir / "experiment_report.md").write_text(md, encoding="utf-8")
    return md
