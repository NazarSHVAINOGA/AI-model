"""PyTorch models and preprocessing helpers for employee performance."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

TARGET_REG = "Productivity_Index"
TARGET_CLS = "Performance_Label"
DROP_COLUMNS = {"Employee_ID", TARGET_REG, TARGET_CLS}
CLASSES = ["Low", "Medium", "High"]

def load_dataset(path: str | Path):
    df = pd.read_csv(path)
    required = {TARGET_REG, TARGET_CLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"У датасеті немає колонок: {sorted(missing)}")
    features = [c for c in df.select_dtypes(include="number").columns if c not in DROP_COLUMNS]
    if not features:
        raise ValueError("Не знайдено числових ознак.")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    medians = df[features].median()
    X = df[features].fillna(medians).to_numpy(dtype=np.float32)
    y_reg = df[TARGET_REG].to_numpy(dtype=np.float32)
    unknown = set(df[TARGET_CLS].dropna().unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"Невідомі класи: {unknown}")
    y_cls = df[TARGET_CLS].map({name:i for i, name in enumerate(CLASSES)})
    if y_cls.isna().any():
        raise ValueError("Performance_Label містить порожні або невідомі значення.")
    return X, y_reg, y_cls.to_numpy(dtype=np.int64), features, medians.to_dict()

class EmployeeMultiTaskNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.BatchNorm1d(hidden // 2), nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden // 2, 1)
        self.classification_head = nn.Linear(hidden // 2, len(CLASSES))

    def forward(self, x):
        z = self.shared(x)
        return self.regression_head(z).squeeze(1), self.classification_head(z)

def save_metadata(path, metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
