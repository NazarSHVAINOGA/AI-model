"""Власний NLU: символьний Transformer, навчений з нуля для розпізнавання наміру.

Чому не LLM: звіти на фінансових даних мають бути перевірюваними. Тому розбір
запиту (клас наміру) робить маленька власна мережа (~100 тис. параметрів, навчання
секунди), а всі числа у відповіді береться лише з обчислень ``engine.py`` — див.
``assistant.py``. Модель не може «вигадати» цифру, бо не генерує текст.

Намірі::

    explain  — чому така оцінка / що на неї вплинуло (Шеплі)
    anomaly  — чи є аномалія / що незвичайного
    forecast — очікування, прогноз
    whatif   — що зробити, щоб досягти цілі (LP)
    help     — що ти вмієш

Дані для навчання генеруються шаблонами з перефразуванням, суржиком/русизмами та
друкарськими помилками. Для чесної оцінки узагальнення частина шаблонів
**повністю виключається з навчання** й використовується лише в тесті.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INTENTS = ["explain", "anomaly", "forecast", "whatif", "help"]
MAX_LEN = 72

SUBJECTS = ["працівника", "працівника №{n}", "співробітника {n}", "працівника з id {n}", "мене", "нашого касира", "цього працівника", "працівника {n}"]
TEMPLATES: Dict[str, List[str]] = {
    "explain": [
        "чому в {s} така оцінка", "поясни оцінку {s}", "що вплинуло на результат {s}",
        "чому результат {s} саме такий", "від чого залежить оцінка {s}", "розкажи, чому оцінка {s} змінилась",
        "які фактори вплинули на показник {s}", "чому в {s} впала оцінка", "поясни, чому в {s} зросла оцінка",
        "чому показники {s} такі", "чому у {s} {m} оцінка менша ніж очікувалось",
    ],
    "anomaly": [
        "чи є аномалія в {s}", "перевір {s} на аномалії", "чи є щось незвичайне у {s}",
        "чи виглядає результат {s} підозріло", "знайди відхилення в {s}", "чи це випадковий збій у {s}",
        "покажи аномальні результати {s}", "чи нормальний результат {s} за {m}", "є проблеми у {s}?",
    ],
    "forecast": [
        "який прогноз для {s}", "яка очікувана оцінка {s}", "чого очікувати від {s}",
        "скільки мав би набрати {s}", "яким буде результат {s} за {m}", "дай прогноз по {s}",
        "який нормальний рівень для {s}", "яка очікувана оцінка {s} з урахуванням сезону",
    ],
    "whatif": [
        "що зробити, щоб {s} набрав {t} балів", "як {s} досягти {t}", "скільки треба підтягнути, щоб отримати {t}",
        "складіть план, щоб {s} вийшов на {t}", "що змінити {s} аби оцінка стала {t}", "як підвищити оцінку {s} до {t}",
        "який мінімальний план для {s} щоб дійти до {t} балів", "як дотягнути до {t} {s}",
    ],
    "help": [
        "що ти вмієш", "допоможи", "які в тебе функції", "що можна в тебе запитати", "help", "з чого почати",
        "як з тобою працювати", "привіт, що ти можеш", "поясни, як користуватись помічником",
    ],
}
# Шаблони, які НЕ беруть участі в навчанні (тест узагальнення):
HELD_OUT = {
    "explain": ["а чому оцінка {s} така, а не інша", "що за цим результатом {s}"],
    "anomaly": ["щось не так із результатом {s}?", "чи не вибивається {s} із норми"],
    "forecast": ["чого ми чекали від {s}", "що передбачає модель для {s}"],
    "whatif": ["підкажи, що робити {s} щоб було {t}", "який шлях до {t} балів для {s}"],
    "help": ["розкажи про свої можливості", "що тут взагалі можна робити"],
}
MONTHS = ["січні", "лютому", "березні", "квітні", "травні", "червні", "липні", "серпні", "вересні", "жовтні", "листопаді", "грудні", "минулому місяці"]
RUSIFY = [("чому", "почему"), ("оцінка", "оценка"), ("результат", "результат"), ("працівника", "сотрудника"), ("що", "что")]
CHARS = list("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя0123456789 ?!.,-№#:") + list("abcdefghijklmnopqrstuvwxyz")
PAD, UNK = 0, 1
VOCAB = {c: i + 2 for i, c in enumerate(CHARS)}


def encode(text: str) -> List[int]:
    ids = [VOCAB.get(c, UNK) for c in text.lower()[:MAX_LEN]]
    return ids + [PAD] * (MAX_LEN - len(ids))


def _noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.2:
        for a, b in RUSIFY:
            if rng.random() < 0.5:
                text = text.replace(a, b)
    if rng.random() < 0.25 and len(text) > 6:  # друкарська помилка: видалення або подвоєння літери
        i = rng.randrange(1, len(text) - 1)
        text = text[:i] + text[i + 1:] if rng.random() < 0.5 else text[:i] + text[i] + text[i:]
    if rng.random() < 0.15:
        text = text.upper()
    return text


def _fill(tpl: str, rng: random.Random) -> str:
    s = rng.choice(SUBJECTS).format(n=rng.randint(1, 400))
    return tpl.format(s=s, t=rng.choice([75, 80, 85, 90, 95]), m=rng.choice(MONTHS))


def make_dataset(n_per_intent: int, seed: int, held_out: bool = False) -> Tuple[List[str], List[int]]:
    rng = random.Random(seed)
    X, y = [], []
    src = HELD_OUT if held_out else TEMPLATES
    for k, intent in enumerate(INTENTS):
        for _ in range(n_per_intent):
            text = _fill(rng.choice(src[intent]), rng)
            X.append(text if held_out else _noise(text, rng))
            y.append(k)
    return X, y


class IntentNet(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 2, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, len(INTENTS))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids == PAD
        h = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        h = self.enc(h, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        return self.out((h * keep).sum(1) / keep.sum(1).clamp(min=1))


def train_nlu(epochs: int = 25, seed: int = 0, log=print) -> Tuple[IntentNet, Dict[str, float]]:
    torch.manual_seed(seed)
    X, y = make_dataset(1500, seed)
    Xh, yh = make_dataset(300, seed + 1, held_out=True)
    enc = lambda xs: torch.tensor([encode(x) for x in xs])
    Xt, yt, Xht, yht = enc(X), torch.tensor(y), enc(Xh), torch.tensor(yh)
    model = IntentNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xt).argmax(-1) == yt).float().mean().item()
        ho_acc = (model(Xht).argmax(-1) == yht).float().mean().item()
    if log:
        log(f"NLU: точність на навчальних шаблонах {tr_acc:.3f}; на ВИКЛЮЧЕНИХ шаблонах {ho_acc:.3f}")
    return model, {"train_acc": tr_acc, "heldout_template_acc": ho_acc}


@torch.no_grad()
def predict_intent(model: IntentNet, text: str, min_conf: float = 0.5) -> Tuple[str, float]:
    model.eval()
    p = F.softmax(model(torch.tensor([encode(text)])), -1)[0]
    k = int(p.argmax())
    return (INTENTS[k], float(p[k])) if float(p[k]) >= min_conf else ("help", float(p[k]))


# --- Витяг слотів (правила; значення перевіряються проти бази, а не вгадуються) ----

MONTH_STEMS = {"січ": 1, "лют": 2, "берез": 3, "квіт": 4, "трав": 5, "черв": 6, "лип": 7, "серп": 8, "верес": 9, "жовт": 10, "листоп": 11, "груд": 12}


def extract_slots(text: str) -> Dict[str, Optional[float]]:
    low = text.lower()
    slots: Dict[str, Optional[float]] = {"employee_id": None, "target": None, "month": None, "year": None}
    m = re.search(r"(?:працівник\w*|співробітник\w*|сотрудник\w*|id|№|#)\s*(?:з\s+id\s*)?№?\s*(\d{1,6})", low)
    if m:
        slots["employee_id"] = int(m.group(1))
    m = re.search(r"(?:набрав|на|до|стала|було|отримати|вийшов на)\s*(\d{2,3})\s*(?:бал|%)?", low) or re.search(r"(\d{2,3})\s*бал", low)
    if m and 40 <= int(m.group(1)) <= 100:
        slots["target"] = float(m.group(1))
    for stem, num in MONTH_STEMS.items():
        if stem in low:
            slots["month"] = num
            break
    m = re.search(r"(20\d{2})", low)
    if m:
        slots["year"] = int(m.group(1))
    if "минул" in low:
        slots["month"] = -1  # відносний: попередній місяць
    return slots


def save_nlu(model: IntentNet, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_nlu(path: Path) -> IntentNet:
    m = IntentNet()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()
