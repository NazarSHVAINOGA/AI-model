"""Copilot-асистент: питання природною мовою → перевірювана відповідь.

Конвеєр (нейро-символічний)::

    текст ──NLU(власна мережа)──► намір + слоти
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
                 engine.analyse()            whatif (LP) / Шеплі
                         └────────────┬─────────────┘
                                      ▼
                     NLG: шаблони, у які підставлено ЛИШЕ обчислені числа

Гарантія «без галюцинацій»: жодна цифра у відповіді не генерується мовною
моделлю — кожна береться з ``Analysis``/``WhatIfPlan`` і дублюється в полі
``evidence``, яке можна перевірити. Якщо даних бракує — асистент прямо це каже.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np

from .engine import GROUP_UA, Analysis, Engine
from .nlu import IntentNet, extract_slots, predict_intent

HELP_TEXT = (
    "Я вмію відповідати на питання про результати працівників за обраний місяць:\n"
    "• «Чому в працівника №17 така оцінка?» — розкладаю очікування за чинниками (Шеплі);\n"
    "• «Чи є аномалія у працівника 17?» — статистична перевірка з контрольованою частотою хибних тривог;\n"
    "• «Який прогноз для працівника 17?» — очікуваний рівень з 90 % інтервалом;\n"
    "• «Як працівнику 17 набрати 90 балів?» — мінімальний план зусиль (лінійне програмування).\n"
    "Уточніть номер працівника та, за потреби, місяць."
)


class Context(Protocol):
    """Джерело даних (у продакшні — база Django; у тестах — симулятор)."""

    def period_arrays(self, year: Optional[int], month: Optional[int]) -> Optional[Dict[str, Any]]: ...
    def can_access(self, emp_id: int) -> bool: ...
    def default_employee(self) -> Optional[int]: ...


MAX_QUESTION_LEN = 200
MIN_INTENT_CONF = 0.6   # нижче — питання не схоже на жодне з підтримуваних

# Питання мають стосуватися оцінок/аномалій/прогнозів/планів працівників (магазинів). Усе інше — відмова.
_DOMAIN = re.compile(
    r"оцінк|аномал|прогноз|очіку|план|бал|результат|показник|kpi|виконан|працівник|співробітник|магазин|"
    r"відділен|чому|поясни|вплин|підозр|відхилен|помічник|вмієш|вмієте|допоможи|help|функці|можеш|привіт|"
    r"набрат|досягт|підвищ|покращ|нормальн|проблем|збій|сезон",
    re.IGNORECASE,
)
REFUSAL = (
    "Це питання поза моєю темою. Я відповідаю лише про результати працівників (магазинів) за обраний період:\n"
    "• пояснити оцінку — «Чому в працівника №17 така оцінка?»\n"
    "• перевірити на аномалію — «Чи є аномалія у працівника 17?»\n"
    "• дати прогноз — «Який прогноз для працівника 17?»\n"
    "• скласти план — «Як працівнику 17 набрати 90 балів?»"
)


def is_in_scope(text: str) -> bool:
    """Питання допустиме, якщо воно містить лексику предметної області (оцінка, аномалія, прогноз, план…)."""
    return bool(_DOMAIN.search(text)) or bool(re.search(r"(?:№|#|id)\s*\d+", text, re.IGNORECASE))


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _shap_sentence(a: Analysis) -> str:
    items = sorted(a.shap.items(), key=lambda kv: -abs(kv[1]))
    parts = []
    for g, v in items:
        if abs(v) < 0.3:
            continue
        parts.append(f"{GROUP_UA[g]} ({'+' if v >= 0 else '−'}{_fmt(abs(v))})")
    if not parts:
        return "Жоден чинник не дав помітного внеску відносно базового рівня."
    return "Очікування сформували: " + ", ".join(parts[:4]) + f"; базовий рівень моделі — {_fmt(a.shap_base)}."


def _json_safe(x):
    """Нескінченні/NaN-числа (напр., квантиль при замалій калібрувальній вибірці) → None: JSON їх не приймає."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


@dataclass
class Copilot:
    engine: Engine
    nlu: IntentNet
    kpi_names: Optional[Callable[[int], List[str]]] = None
    _cache: Dict[Any, Any] = field(default_factory=dict)

    def answer(self, text: str, ctx: Context, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text or len(text) > MAX_QUESTION_LEN:
            return {"intent": "rejected", "confidence": 0.0, "evidence": {},
                    "text": f"Питання має містити від 1 до {MAX_QUESTION_LEN} символів."}
        if not is_in_scope(text):
            return {"intent": "out_of_scope", "confidence": 0.0, "evidence": {}, "text": REFUSAL}
        # Інші предметні області (магазини) нормалізуємо до «працівника» — NLU навчався на цьому слові.
        norm = re.sub(r"магазин\w*", "працівник", text, flags=re.IGNORECASE)
        intent, conf = predict_intent(self.nlu, norm, min_conf=MIN_INTENT_CONF)
        slots = extract_slots(norm)
        if intent == "help":
            return {"intent": intent, "confidence": conf, "text": HELP_TEXT, "evidence": {}}

        emp_id = slots["employee_id"] or ctx.default_employee()
        if emp_id is None:
            return {"intent": intent, "confidence": conf, "evidence": {},
                    "text": "Вкажіть, про якого працівника йдеться, наприклад: «працівник №17»."}
        emp_id = int(emp_id)
        if not ctx.can_access(emp_id):
            return {"intent": intent, "confidence": conf, "evidence": {},
                    "text": "У вас немає доступу до даних цього працівника."}

        y = int(slots["year"]) if slots["year"] else year
        m = month
        if slots["month"] and slots["month"] > 0:
            m = int(slots["month"])
        arrays = ctx.period_arrays(y, m)
        if arrays is None or emp_id not in arrays["emp_index"]:
            return {"intent": intent, "confidence": conf, "evidence": {},
                    "text": "Для цього працівника немає введених даних за обраний період."}
        e = arrays["emp_index"][emp_id]
        key = arrays.get("cache_key")      # для великих наборів (Rossmann) аналіз рахуємо один раз
        if key is not None and key in self._cache:
            res = self._cache[key]
        else:
            res = self.engine.analyse(
                arrays["q"], arrays["pos"], arrays["branch"], arrays["events"],
                arrays["month"], arrays["active"], arrays["t"],
            )
            if key is not None:
                self._cache[key] = res
        a = res.get(e)
        if a is None:
            return {"intent": intent, "confidence": conf, "evidence": {},
                    "text": "Недостатньо даних для аналізу цього працівника в обраному місяці."}
        noun = arrays.get("entity", "Працівник")
        head = f"{noun} №{emp_id}, {arrays['label']}. "
        n_hist = int(np.isfinite(arrays["q"][e, :arrays["t"]]).any(-1).sum())
        caveat = (f" Увага: історії лише {n_hist} міс., висновки менш надійні." if n_hist < 6 else "")

        if intent == "explain":
            body = (f"Фактичний індекс виконання {_fmt(a.actual_index)}, модель очікувала {_fmt(a.expected_index)} "
                    f"(типове коливання ±{_fmt(a.sigma)}), тобто факт {'вищий' if a.signed_surprise >= 0 else 'нижчий'} "
                    f"за очікуване на {_fmt(abs(a.signed_surprise), 2)}σ. {_shap_sentence(a)}")
            ev = {"actual": a.actual_index, "expected": a.expected_index, "sigma": a.sigma, "shapley": a.shap, "base": a.shap_base}
        elif intent == "anomaly":
            lo = self.engine.conformal.quantile(0.01, int(arrays["pos"][e]))
            verdict = ("ПОЗНАЧЕНО як аномалію" if a.flagged
                       else "на межі норми (підозріло, p < 0,05, але поріг аномалії не перейдено)" if a.p_value <= 0.05
                       else "у межах норми")
            direction = "нижче" if a.signed_surprise < 0 else "вище"
            body = (f"Результат {verdict}: факт {_fmt(a.actual_index)} проти очікуваних {_fmt(a.expected_index)}, "
                    f"тобто {direction} на {_fmt(abs(a.signed_surprise), 2)}σ; p-значення {a.p_value:.4f} "
                    f"(поріг 0,01 → очікувана частка хибних тривог не більше 1 %).")
            if abs(a.systemic_shift) > 1.5:
                body += (f" Але вся посада в цьому місяці відхилилась у середньому на {_fmt(a.systemic_shift, 2)}σ — "
                         f"імовірно, це спільний чинник, а не індивідуальна проблема.")
            ev = {"p_value": a.p_value, "flagged": a.flagged, "signed_surprise": a.signed_surprise,
                  "systemic_shift": a.systemic_shift, "conformal_quantile_alpha_0.01": lo}
        elif intent == "forecast":
            qh = self.engine.conformal.quantile(0.10, int(arrays["pos"][e]))
            half = qh * a.sigma if np.isfinite(qh) else float("nan")
            interval = (f"90 % інтервал [{_fmt(a.expected_index - half)}; {_fmt(a.expected_index + half)}] (conformal-калібрування)"
                        if np.isfinite(half) else "інтервал не обчислено: замало даних для калібрування")
            body = f"Очікуваний індекс виконання {_fmt(a.expected_index)}; {interval}. {_shap_sentence(a)}"
            ev = {"expected": a.expected_index, "interval_half_width": half, "shapley": a.shap}
        else:  # whatif
            target = slots["target"] or 90.0
            plan = self.engine.plan(a, target)
            names = (self.kpi_names(emp_id) if self.kpi_names else None) or [f"KPI {i + 1}" for i in range(len(plan.deltas))]
            lines = [f"• {names[i]}: покращити виконання на {_fmt(d)} п.п. ({_fmt(d / a.kpi_sigma[i])} власних σ)"
                     for i, d in enumerate(plan.deltas) if d > 1e-6]
            if plan.required_gain <= 0:
                body = f"Поточна оцінка {_fmt(plan.start_score)} вже не нижча за ціль {_fmt(target)} (з запасом надійності)."
            elif not plan.feasible:
                mean_plan = self.engine.plan(a, target, confidence=0.5)
                if mean_plan.feasible and mean_plan.required_gain <= 0:
                    body = (f"Поточна оцінка {_fmt(plan.start_score)} вже вища за ціль {_fmt(target)}, тож додаткових зусиль "
                            f"у середньому не потрібно. Гарантувати це з надійністю 80 % неможливо: при коливанні "
                            f"±{_fmt(a.sigma)} стеля оцінки {_fmt(plan.max_reachable)} не лишає достатнього запасу.")
                elif mean_plan.feasible:
                    lines = [f"• {names[i]}: покращити виконання на {_fmt(d)} п.п. ({_fmt(d / a.kpi_sigma[i])} власних σ)"
                             for i, d in enumerate(mean_plan.deltas) if d > 1e-6]
                    body = (f"Ціль {_fmt(target)} досяжна лише «в середньому» (надійність 50 %), але не з надійністю 80 %: "
                            f"стеля оцінки {_fmt(plan.max_reachable)} не дає запасу на випадкові коливання "
                            f"(±{_fmt(a.sigma)}). План для 50 % (сумарно {_fmt(mean_plan.effort_sigma)} σ):\n" + "\n".join(lines))
                    plan = mean_plan
                else:
                    body = (f"Ціль {_fmt(target)} недосяжна в межах реалістичного плану (≤ 2σ на KPI): "
                            f"максимум {_fmt(plan.max_reachable)}.")
            else:
                body = (f"Щоб вийти з {_fmt(plan.start_score)} на {_fmt(target)} з надійністю 80 %, потрібен приріст "
                        f"оцінки на {_fmt(plan.required_gain)} п. Мінімальний за зусиллями план "
                        f"(сумарно {_fmt(plan.effort_sigma)} σ):\n" + "\n".join(lines))
            ev = {"target": target, "feasible": plan.feasible, "deltas": plan.deltas.tolist(),
                  "effort_sigma": plan.effort_sigma, "required_gain": plan.required_gain}
        return {"intent": intent, "confidence": conf, "text": head + body + caveat, "evidence": _json_safe(ev)}
