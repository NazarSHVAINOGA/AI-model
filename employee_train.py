"""Train Employee V2 multi-task model with stratified splits and class weighting."""
from __future__ import annotations
import copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from employee_model import load_dataset, EmployeeMultiTaskNet, CLASSES, save_metadata

DATA = Path("data/employee_activity/employee_performance_dataset.csv")
OUT = Path("apps/copilot/artifacts/employee_v2")
SEED, EPOCHS, BATCH, PATIENCE = 42, 150, 128, 18
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_loader(X, yr, yc, indices, shuffle):
    ds = TensorDataset(torch.tensor(X[indices]), torch.tensor(yr[indices]), torch.tensor(yc[indices]))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=False)

def main():
    start = time.time(); seed_everything(SEED)
    if not DATA.exists():
        raise FileNotFoundError(f"Не знайдено {DATA.resolve()}")
    X, yr, yc, features, medians = load_dataset(DATA)
    idx = np.arange(len(X))
    train_idx, temp_idx = train_test_split(idx, test_size=0.35, random_state=SEED, stratify=yc)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.60, random_state=SEED, stratify=yc[temp_idx])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_all = scaler.transform(X).astype(np.float32)
    print(f"Дані: {len(X)} рядків; пристрій: {DEVICE}")
    print("Розподіл класів:", {c:int((yc==i).sum()) for i,c in enumerate(CLASSES)})
    print(f"Розбиття: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    model = EmployeeMultiTaskNet(X.shape[1]).to(DEVICE)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASSES)), y=yc[train_idx])
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    reg_loss = nn.SmoothL1Loss()
    cls_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    train_loader = make_loader(X_all, yr, yc, train_idx, True)
    val_loader = make_loader(X_all, yr, yc, val_idx, False)
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(1, EPOCHS+1):
        model.train(); train_total = 0.0
        for xb, yb_reg, yb_cls in train_loader:
            xb, yb_reg, yb_cls = xb.to(DEVICE), yb_reg.to(DEVICE), yb_cls.to(DEVICE)
            optimizer.zero_grad()
            pred_reg, pred_cls = model(xb)
            loss = reg_loss(pred_reg, yb_reg) / 15.0 + cls_loss(pred_cls, yb_cls)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            train_total += loss.item() * len(xb)
        model.eval(); val_total = 0.0
        with torch.no_grad():
            for xb, yb_reg, yb_cls in val_loader:
                xb, yb_reg, yb_cls = xb.to(DEVICE), yb_reg.to(DEVICE), yb_cls.to(DEVICE)
                pr, pc = model(xb)
                val_total += (reg_loss(pr, yb_reg)/15.0 + cls_loss(pc, yb_cls)).item()*len(xb)
        val_loss = val_total / len(val_idx)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:03d}: train={train_total/len(train_idx):.4f} val={val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping: epoch {epoch}"); break

    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT/"employee_model.pt")
    metadata = {
        "seed": SEED, "features": features, "classes": CLASSES,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "imputation_medians": medians, "test_indices": test_idx.tolist(),
        "best_validation_loss": best, "device_used": str(DEVICE)
    }
    save_metadata(OUT/"metadata.json", metadata)
    print(f"Модель і metadata збережено в: {OUT.resolve()}")
    print(f"Час: {time.time()-start:.1f} с")
    print("Наступний крок: python -m apps.copilot.core.employee_evaluate")

if __name__ == "__main__": main()
