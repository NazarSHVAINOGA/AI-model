"""Нейромережа прогнозу очікуваного виконання KPI: ``KpiForecaster``.

Ідея
----
Для пари (працівник, місяць t) модель відповідає на питання: *«якби цей
працівник працював у своєму звичайному режимі, яким мало б бути виконання
кожного KPI у цьому місяці з огляду на сезон, зовнішні події та те, як
відпрацювали колеги?»*. Різниця між фактом і цим очікуванням — це «сюрприз»,
який потім калібрується conformal-методом (див. ``conformal.py``) і
перетворюється на p-значення аномалії.

Архітектура
-----------
* Токени: W токенів історії (місяці t-W..t-1), 1 цільовий токен (місяць t, значення
  KPI приховане), 1 контекстний токен (колеги + посада + відділення).
* Transformer-енкодер (pre-LN) із маскою «порожніх» місяців — коректно працює
  для нових працівників (cold-start).
* Голови: для кожного з KMAX KPI — середнє μ_k, масштаб σ_k > 0; спільне число
  ступенів свободи ν_pos > 2 для кожної посади (важкі хвости).
* Втрата — від'ємне логарифмічне правдоподібство Student-t. Воно **робастне**:
  модель навчається на «забруднених» даних (аномалії не розмічені), і
  вони не псують оцінку очікування так, як це робить MSE.

Груповий dropout (ключова деталь)
---------------------------------
Під час навчання кожна з 5 груп ознак (``features.GROUPS``) незалежно
«вимикається» (обнуляється), а модель отримує бінарні прапорці присутності груп.
Наслідки:
1. Модель вчиться працювати за будь-якої підмножини груп — тому вимкнена група в
   Shapley-аналізі (``shapley.py``) не виводить мережу за межі розподілу даних;
2. це регуляризація — мережа не зациклюється на одній групі.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_GROUPS, Dataset
from .simulator import KMAX, N_EVENT_TYPES, N_POS, kpi_slot_mask, weight_matrix

MAX_BRANCH = 64


class KpiForecaster(nn.Module):
    def __init__(
        self,
        window: int = 12,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_pos: int = N_POS,
    ) -> None:
        super().__init__()
        self.window = window
        self.d = d_model
        tok_in = KMAX + KMAX + 1 + 2 + N_EVENT_TYPES  # q, маска, valid, календар, події
        self.tok_proj = nn.Linear(tok_in, d_model)
        self.step_emb = nn.Embedding(window + 2, d_model)   # позиція токена
        self.kind_emb = nn.Embedding(3, d_model)            # 0 історія, 1 ціль, 2 контекст
        self.ctx_proj = nn.Linear(4 + N_GROUPS, d_model)
        self.pos_emb = nn.Embedding(n_pos, d_model)
        self.branch_emb = nn.Embedding(MAX_BRANCH, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2 * KMAX + 2))
        self.nu_raw = nn.Parameter(torch.zeros(n_pos))  # ν_pos = 2.1 + softplus(...)
        self.register_buffer("slot", torch.tensor(kpi_slot_mask(), dtype=torch.bool))
        self.register_buffer("wmat", torch.tensor(weight_matrix()))

    # ---- допоміжне ---------------------------------------------------------

    def nu(self, pos: torch.Tensor) -> torch.Tensor:
        return 2.1 + F.softplus(self.nu_raw)[pos]

    @staticmethod
    def apply_group_mask(b: Dict[str, torch.Tensor], present: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Обнуляє вимкнені групи. ``present``: [B, N_GROUPS] у {0,1}."""
        p = present
        hist, cal, ev, peers, ident = (p[:, i] for i in range(N_GROUPS))
        out = dict(b)
        out["hq"] = b["hq"] * hist[:, None, None]
        out["hm"] = b["hm"] * hist[:, None, None]
        out["hv"] = b["hv"] * hist[:, None]
        out["hcal"] = b["hcal"] * cal[:, None, None]
        out["tcal"] = b["tcal"] * cal[:, None]
        out["hev"] = b["hev"] * ev[:, None, None]
        out["tev"] = b["tev"] * ev[:, None]
        out["peers"] = b["peers"] * peers[:, None]
        out["ident"] = ident
        return out

    # ---- прямий прохід ---------------------------------------------------------

    def forward(
        self, b: Dict[str, torch.Tensor], present: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Повертає (μ, σ, ν, μ_idx, σ_idx) у «сирих» одиницях q (відсотки виконання).

        ``μ_idx, σ_idx`` — розподіл *індексу виконання* ``Σ w_k q_k`` (необрізаної
        композитної оцінки). Окрема голова потрібна, бо помилки різних KPI одного
        працівника корельовані (спільний AR-компонент), і припущення про
        незалежність KPI занижувало б невизначеність індексу.
        """
        B = b["hq"].shape[0]
        dev = b["hq"].device
        if present is None:
            present = torch.ones(B, N_GROUPS, device=dev)
        x = self.apply_group_mask(b, present)

        hist_in = torch.cat([x["hq"], x["hm"], x["hv"].unsqueeze(-1), x["hcal"], x["hev"]], dim=-1)
        z = torch.zeros(B, 1, KMAX, device=dev)
        tgt_in = torch.cat(
            [z, z, torch.zeros(B, 1, 1, device=dev), x["tcal"].unsqueeze(1), x["tev"].unsqueeze(1)], dim=-1
        )
        toks = self.tok_proj(torch.cat([hist_in, tgt_in], dim=1))  # [B, W+1, d]
        steps = torch.arange(self.window + 1, device=dev)
        toks = toks + self.step_emb(steps)[None]
        kind = torch.cat([torch.zeros(self.window, dtype=torch.long), torch.ones(1, dtype=torch.long)]).to(dev)
        toks = toks + self.kind_emb(kind)[None]

        ident = x["ident"].unsqueeze(-1)
        ctx = self.ctx_proj(torch.cat([x["peers"], present], dim=-1))
        ctx = ctx + ident * (self.pos_emb(b["pos"]) + self.branch_emb(b["branch"].clamp(max=MAX_BRANCH - 1)))
        ctx = (ctx + self.kind_emb.weight[2] + self.step_emb.weight[self.window + 1]).unsqueeze(1)

        seq = torch.cat([toks, ctx], dim=1)  # [B, W+2, d]
        # Маска ключів: порожні місяці історії ігноруються; ціль і контекст — завжди видимі.
        pad = torch.cat(
            [x["hv"] < 0.5, torch.zeros(B, 2, dtype=torch.bool, device=dev)], dim=1
        )
        h = self.encoder(seq, src_key_padding_mask=pad)
        h = self.norm(h[:, self.window])  # цільовий токен
        out = self.head(h)
        mu = Q_CENTER_T + Q_SCALE_T * out[:, :KMAX]
        sigma = (F.softplus(out[:, KMAX:2 * KMAX]) + 1e-3) * Q_SCALE_T * 0.5
        w = self.wmat[b["pos"]]
        mu_idx = (w * mu).sum(-1) + Q_SCALE_T * 0.25 * out[:, 2 * KMAX]
        sig_idx = (F.softplus(out[:, 2 * KMAX + 1]) + 1e-3) * Q_SCALE_T * 0.5
        return mu, sigma, self.nu(b["pos"]), mu_idx, sig_idx


Q_CENTER_T, Q_SCALE_T = 100.0, 20.0


def student_t_nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Середнє від'ємне лог-правдоподібство Student-t за наявними KPI."""
    nu = nu.unsqueeze(-1)
    z = (y - mu) / sigma
    nll = (
        torch.log(sigma)
        + 0.5 * (nu + 1) * torch.log1p(z * z / nu)
        + torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2)
        + 0.5 * torch.log(nu * math.pi)
    )
    return (nll * mask).sum() / mask.sum().clamp(min=1)


# --- Утиліти: dataset → тензори, прогноз композитної оцінки -------------------------

TENSOR_KEYS = ["hq", "hm", "hv", "hcal", "hev", "tcal", "tev", "peers", "pos", "branch", "y", "ym"]


def to_tensors(ds: Dataset, device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k in TENSOR_KEYS:
        arr = getattr(ds, k)
        t = torch.from_numpy(arr)
        out[k] = t.to(device)
    return out


def slice_batch(b: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in b.items()}


@torch.no_grad()
def predict_composite(
    model: KpiForecaster,
    b: Dict[str, torch.Tensor],
    present: Optional[torch.Tensor] = None,
    n_samples: int = 256,
    seed: int = 0,
    return_kpi: bool = False,
):
    """Передбачувальний розподіл композитної оцінки Monte-Carlo.

    Оцінка = Σ w_k · min(100, max(0, q_k)) — **нелінійна** (обрізання на 100), тому
    середнє й розкид композиту не можна отримати простою алгеброю з μ_k, σ_k;
    семплюємо q_k = μ_k + σ_k · T_ν і пропускаємо через ту саму формулу, що й у ``scoring``.
    Повертає (mean, std) композитної оцінки; опційно ще μ, σ по KPI.
    """
    model.eval()
    mu, sigma, nu, _, _ = model(b, present)
    B = mu.shape[0]
    # Детермінізм: фіксуємо глобальний seed (torch._standard_gamma не приймає Generator).
    torch.manual_seed(seed)
    # Student-t через ратіо нормальної та χ²: T = Z / sqrt(V/ν)
    z = torch.randn(n_samples, B, KMAX, device=mu.device)
    chi2 = 2.0 * torch._standard_gamma(
        (nu / 2).expand(n_samples, B).contiguous()
    ).unsqueeze(-1)
    t = z / torch.sqrt(chi2 / nu[None, :, None])
    samples = mu[None] + sigma[None] * t                              # [S,B,K]
    capped = samples.clamp(0.0, 100.0)
    w = model.wmat[b["pos"]]                                          # [B,K]
    comp = (capped * w[None]).sum(-1)                                 # [S,B]
    mean, std = comp.mean(0), comp.std(0)
    if return_kpi:
        return mean, std, mu, sigma
    return mean, std


def index_nll(y_idx: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Student-t NLL для індексу виконання (скаляр на приклад)."""
    z = (y_idx - mu) / sigma
    nll = (
        torch.log(sigma) + 0.5 * (nu + 1) * torch.log1p(z * z / nu)
        + torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2) + 0.5 * torch.log(nu * math.pi)
    )
    return nll.mean()


def observed_index(b: Dict[str, torch.Tensor], wmat: torch.Tensor) -> torch.Tensor:
    """Фактичний необрізаний індекс виконання Σ w_k q_k (з перенормуванням на наявні KPI)."""
    w = wmat[b["pos"]] * b["ym"]
    return (w * b["y"]).sum(-1) / w.sum(-1).clamp(min=1e-6)


def observed_composite(b: Dict[str, torch.Tensor], wmat: torch.Tensor) -> torch.Tensor:
    """Фактична композитна оцінка цільового місяця (з перенормуванням ваг для відсутніх KPI)."""
    w = wmat[b["pos"]] * b["ym"]
    capped = b["y"].clamp(0.0, 100.0)
    return (w * capped).sum(-1) / w.sum(-1).clamp(min=1e-6)
