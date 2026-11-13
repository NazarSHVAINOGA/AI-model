"""Evaluate Employee V2 on the held-out test partition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, classification_report, confusion_matrix)
from employee_model import load_dataset, EmployeeMultiTaskNet, CLASSES

ART = Path("apps/copilot/artifacts/employee_v2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    with open(ART/"metadata.json", encoding="utf-8") as f: meta = json.load(f)
    data_path = Path("data/employee_activity/employee_performance_dataset.csv")
    X, yr, yc, features, _ = load_dataset(data_path)
    if features != meta["features"]:
        raise ValueError("Список ознак відрізняється від того, з яким навчали модель.")
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scale = np.array(meta["scaler_scale"], dtype=np.float32)
    X = ((X-mean)/scale).astype(np.float32)
    test_idx = np.array(meta["test_indices"], dtype=int)
    model = EmployeeMultiTaskNet(len(features)).to(DEVICE)
    model.load_state_dict(torch.load(ART/"employee_model.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        pr, pc = model(torch.tensor(X[test_idx], device=DEVICE))
        pred_reg = pr.cpu().numpy()
        pred_cls = pc.argmax(1).cpu().numpy()
    true_reg, true_cls = yr[test_idx], yc[test_idx]
    print("REGRESSION")
    print(json.dumps({
        "MAE": float(mean_absolute_error(true_reg, pred_reg)),
        "RMSE": float(np.sqrt(mean_squared_error(true_reg, pred_reg))),
        "R2": float(r2_score(true_reg, pred_reg))
    }, indent=2))
    print("\nCLASSIFICATION")
    print(json.dumps({
        "accuracy": float(accuracy_score(true_cls, pred_cls)),
        "macro_f1": float(f1_score(true_cls, pred_cls, average="macro", zero_division=0)),
        "report": classification_report(true_cls, pred_cls, labels=[0,1,2],
             target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(true_cls, pred_cls, labels=[0,1,2]).tolist()
    }, indent=2, ensure_ascii=False))
if __name__ == "__main__": main()
