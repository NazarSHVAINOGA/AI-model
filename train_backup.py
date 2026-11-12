"""Навчання KpiForecaster на симульованій панелі.

Запуск (з кореня проєкту, у активованому .venv)::

    python manage.py copilot_train            # повний цикл: симуляція → навчання → калібрування

Часові розбиття (щоб не було «підглядання в майбутнє»)::

    train  [0, 60)   — навчання
    val    [60, 66)  — ранній зупин
    calib  [66, 78)  — калібрування conformal-методу
    test   [78, 96)  — фінальна оцінка (модель цих місяців не бачила)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .features import N_GROUPS, Dataset, dataset_from_panel
from .model import KpiForecaster, index_nll, observed_index, slice_batch, student_t_nll, to_tensors
from .simulator import Panel, SimConfig, simulate

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

SPLITS = {"train": (0, 60), "val": (60, 66), "calib": (66, 78), "test": (78, 96)}


@dataclass
class TrainConfig:
    window: int = 12
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-2
    batch_size: int = 512
    epochs: int = 40
    patience: int = 6
    group_drop: float = 0.2       # ймовірність вимкнути кожну групу ознак
    group_drop_batch: float = 0.7  # частка батчів, де застосовується груповий dropout
    seed: int = 0
    # Групи, які взагалі не подаються на вхід (для ablation-експериментів).
    disabled_groups: tuple = ()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _present_mask(B: int, cfg: TrainConfig, dev: torch.device, training: bool) -> torch.Tensor:
    present = torch.ones(B, N_GROUPS, device=dev)
    if training:
        use = (torch.rand(B, 1, device=dev) < cfg.group_drop_batch).float()
        drop = (torch.rand(B, N_GROUPS, device=dev) < cfg.group_drop).float()
        present = 1.0 - drop * use
    for g in cfg.disabled_groups:
        present[:, g] = 0.0
    return present


def train_model(
    ds_train: Dataset,
    ds_val: Dataset,
    cfg: TrainConfig,
    log: Optional[Callable[[str], None]] = print,
) -> KpiForecaster:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = device()
    model = KpiForecaster(
        window=cfg.window, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, dropout=cfg.dropout,
    ).to(dev)
    tr, va = to_tensors(ds_train, dev), to_tensors(ds_val, dev)
    n = len(ds_train)
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15
    )
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            b = slice_batch(tr, idx)
            present = _present_mask(len(idx), cfg, dev, training=True)
            mu, sigma, nu, mu_i, sig_i = model(b, present)
            loss = student_t_nll(b["y"], mu, sigma, nu, b["ym"]) + index_nll(
                observed_index(b, model.wmat), mu_i, sig_i, nu
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        val = evaluate_nll(model, va, cfg)
        if log:
            log(f"epoch {epoch + 1:02d}  train_nll={tot / n:.4f}  val_nll={val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                if log:
                    log("ранній зупин")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_nll(model: KpiForecaster, b: Dict[str, torch.Tensor], cfg: TrainConfig) -> float:
    model.eval()
    n = b["y"].shape[0]
    tot = 0.0
    for i in range(0, n, 2048):
        idx = torch.arange(i, min(n, i + 2048), device=b["y"].device)
        bb = slice_batch(b, idx)
        present = _present_mask(len(idx), cfg, b["y"].device, training=False)
        mu, sigma, nu, mu_i, sig_i = model(bb, present)
        loss = student_t_nll(bb["y"], mu, sigma, nu, bb["ym"]) + index_nll(
            observed_index(bb, model.wmat), mu_i, sig_i, nu
        )
        tot += loss.item() * len(idx)
    return tot / n


def save_model(model: KpiForecaster, cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": asdict(cfg)}, path)


def load_model(path: Path, map_location: str | torch.device = "cpu") -> tuple[KpiForecaster, TrainConfig]:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = TrainConfig(**{**asdict(TrainConfig()), **blob["cfg"]})
    cfg.disabled_groups = tuple(cfg.disabled_groups)
    model = KpiForecaster(
        window=cfg.window, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, dropout=cfg.dropout,
    )
    model.load_state_dict(blob["state"])
    return model.to(map_location).eval(), cfg


def prepare(sim_cfg: SimConfig | None = None, window: int = 12):
    panel = simulate(sim_cfg)
    ds = dataset_from_panel(panel, window=window)
    parts = {name: ds.time_range(lo, hi) for name, (lo, hi) in SPLITS.items()}
    return panel, ds, parts
