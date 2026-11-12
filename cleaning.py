"""Послідовне очищення історії від аномалій («anti-self-masking»).

Проблема самомаскування
-----------------------
Будь-який детектор, що порівнює поточний місяць із власною історією працівника,
має вроджену вразливість. Нехай у місяці t-1 сталась просадка на −22 пункти.
Для місяця t історія вже «забруднена»: модель (і медіана/MAD у чинному
modified Z-score) бачить новий низький рівень, вважає його нормою й **не
сигналізує, хоча просадка триває**. На симульованих даних (див. звіт
експерименту) AUROC на *першому* місяці епізоду 0.91, а на місяцях його
*продовження* лише 0.72.

Метод
-----
Обробляємо місяці в хронологічному порядку. Якщо для клітинки (працівник, t) сюрприз
``|y − μ_idx| / σ_idx > c``, то в усіх майбутніх вікнах, де ця клітинка
з'являється як історія, її KPI замінюються на *очікування моделі* μ_k, тобто на
контрфактичне «як було б без аномалії». Наступні місяці порівнюються з
очищеною історією, тому епізод залишається видимим, доки не завершиться.

Це послідовна процедура — імпутація, яка використовує лише минуле, тож її
можна застосовувати онлайн (у продакшні — при фіналізації кожного періоду).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import torch

from .features import N_GROUPS, Q_CENTER, Q_SCALE, Dataset
from .model import KpiForecaster, slice_batch, to_tensors
from .train import device


@torch.no_grad()
def sequential_predict(
    model: KpiForecaster,
    ds: Dataset,
    t_from: int,
    clean: bool = True,
    c: float = 3.0,
    disabled: tuple = (),
) -> Dict[str, np.ndarray]:
    """Прогноз для всіх прикладів із ``t >= t_from`` у хронологічному порядку.

    Повертає масиви довжини ``len(ds)`` (для ``t < t_from`` — NaN):
    ``mu_idx``, ``sig_idx``, ``surprise`` (з урахуванням знака → ``signed``), ``flagged``.
    ``clean=False`` вимикає очищення (для порівняння; поведінка = звичайний прогноз).
    """
    dev = device()
    model.to(dev).eval()
    W = ds.hq.shape[1]
    hq = ds.hq.copy()
    key = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(ds.emp, ds.t))}
    n = len(ds)
    mu_idx = np.full(n, np.nan, np.float32)
    sig_idx = np.full(n, np.nan, np.float32)
    flagged = np.zeros(n, dtype=bool)

    from .simulator import weight_matrix
    wmat = weight_matrix()
    y_idx = (wmat[ds.pos] * ds.ym * ds.y).sum(-1) / np.maximum((wmat[ds.pos] * ds.ym).sum(-1), 1e-6)

    for t in np.unique(ds.t[ds.t >= t_from]):
        sel = np.nonzero(ds.t == t)[0]
        part = replace(ds.select(sel), hq=hq[sel])
        b = to_tensors(part, dev)
        present = torch.ones(len(sel), N_GROUPS, device=dev)
        for g in disabled:
            present[:, g] = 0.0
        mu_k, _, _, mi, si = model(b, present)
        mi, si, mu_k = mi.cpu().numpy(), si.cpu().numpy(), mu_k.cpu().numpy()
        mu_idx[sel], sig_idx[sel] = mi, si
        if not clean:
            continue
        bad = np.abs(y_idx[sel] - mi) / si > c
        flagged[sel] = bad
        for j in np.nonzero(bad)[0]:
            i = sel[j]
            e = int(ds.emp[i])
            vec = np.where(ds.ym[i] > 0, (mu_k[j] - Q_CENTER) / Q_SCALE, 0.0).astype(np.float32)
            # Підміняємо цю клітинку в історії наступних W вікон цього працівника.
            for d in range(1, W + 1):
                k = key.get((e, int(t) + d))
                if k is not None:
                    hq[k, W - d] = vec
    surprise = np.abs(y_idx - mu_idx) / sig_idx
    return {"mu_idx": mu_idx, "sig_idx": sig_idx, "surprise": surprise,
            "signed": (y_idx - mu_idx) / sig_idx, "flagged": flagged, "y_idx": y_idx}
