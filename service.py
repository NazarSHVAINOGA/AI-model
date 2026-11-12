"""Міст між базою Django та ядром Copilot (``core/``): збирає масиви «працівник × місяць × KPI»."""
from __future__ import annotations

import functools
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

from apps.kpi.models import KPIDefinition, KPIFact, KPIPlan
from apps.seasonal.models import ExternalEvent, SeasonalProfile
from apps.structure.models import Employee

from .core.simulator import EVENT_TYPES, KMAX, POSITIONS

HISTORY_MONTHS = 24
POS_INDEX = {p.name: i for i, p in enumerate(POSITIONS)}


def _months_back(year: int, month: int, n: int) -> List[tuple]:
    out = []
    for k in range(n - 1, -1, -1):
        y, m = year, month - k
        while m < 1:
            m, y = m + 12, y - 1
        out.append((y, m))
    return out


def latest_period() -> Optional[tuple]:
    """Останній *заповнений* період: де фактів не менше половини від типової місячної кількості.

    Просто «найновіший місяць» не годиться: у ньому часто лише кілька щойно введених фактів.
    """
    from django.db.models import Count

    rows = list(
        KPIFact.objects.values_list("year", "month").annotate(n=Count("id")).order_by("-year", "-month")
    )
    if not rows:
        return None
    typical = float(np.median([n for _, _, n in rows]))
    for y, m, n in rows:
        if n >= 0.5 * typical:
            return (y, m)
    return (rows[0][0], rows[0][1])


def _seasonal(position_id: int, month: int) -> float:
    qs = SeasonalProfile.objects.filter(month=month)
    row = qs.filter(position_id=position_id).first() or qs.filter(position__isnull=True).first()
    return float(row.coefficient) if row else 1.0


def build_arrays(year: Optional[int], month: Optional[int]) -> Optional[Dict[str, Any]]:
    """Панель за HISTORY_MONTHS місяців, що закінчується заданим періодом.

    q — виконання плану, вирівняне за напрямком (як у ``scoring._simple_normalize``, але без обрізання).
    """
    if year is None or month is None:
        last = latest_period()
        if last is None:
            return None
        year, month = last
    periods = _months_back(year, month, HISTORY_MONTHS)
    emps = list(Employee.objects.filter(is_active=True).select_related("position", "branch").order_by("id"))
    if not emps:
        return None
    emp_index = {e.id: i for i, e in enumerate(emps)}
    N, T = len(emps), len(periods)
    pidx = {p: t for t, p in enumerate(periods)}
    q = np.full((N, T, KMAX), np.nan, np.float32)

    defs: Dict[int, List[KPIDefinition]] = {}
    for d in KPIDefinition.objects.order_by("position_id", "id"):
        defs.setdefault(d.position_id, []).append(d)
    slot = {d.id: k for ds in defs.values() for k, d in enumerate(ds)}
    plans = {
        (p.kpi_definition_id, p.branch_id, p.year, p.month): float(p.plan_value)
        for p in KPIPlan.objects.filter(year__gte=periods[0][0])
    }
    seas_cache: Dict[tuple, float] = {}
    emp_by_id = {e.id: e for e in emps}
    for f in KPIFact.objects.filter(employee_id__in=emp_index).select_related("kpi_definition"):
        t = pidx.get((f.year, f.month))
        if t is None or f.kpi_definition_id not in slot:
            continue
        e = emp_by_id[f.employee_id]
        plan = plans.get((f.kpi_definition_id, e.branch_id, f.year, f.month), 0.0)
        key = (e.position_id, f.month)
        if key not in seas_cache:
            seas_cache[key] = _seasonal(*key)
        adj = plan * seas_cache[key]
        if adj <= 0:
            continue
        r = float(f.fact_value) / adj * 100.0
        val = r if f.kpi_definition.direction == "higher_better" else 200.0 - r
        q[emp_index[e.id], t, slot[f.kpi_definition_id]] = np.clip(val, 0.0, 250.0)

    events = np.zeros((T, len(EVENT_TYPES)), np.float32)
    for ev in ExternalEvent.objects.filter(event_date__gte=date(periods[0][0], periods[0][1], 1)):
        t = pidx.get((ev.event_date.year, ev.event_date.month))
        if t is None or ev.event_type not in EVENT_TYPES:
            continue
        v = ev.quantitative_value
        mag = 1.0 if v is None else float(np.clip(abs(v) / 5.0, 0.3, 2.0)) * (1 if v >= 0 else -1)
        events[t, EVENT_TYPES.index(ev.event_type)] += mag

    pos = np.array([POS_INDEX.get(e.position.name, 0) for e in emps], dtype=np.int64)
    branch = np.array([e.branch_id % 63 for e in emps], dtype=np.int64)
    return {
        "q": q, "pos": pos, "branch": branch, "events": events,
        "month": np.array([m for _, m in periods]), "active": ~np.isnan(q).all(-1),
        "t": T - 1, "emp_index": emp_index, "label": f"{month:02d}.{year}",
        "periods": periods,
    }


class DbContext:
    """Контекст запиту з урахуванням ролі користувача (RBAC як в ``ai_reports``)."""

    def __init__(self, user) -> None:
        self.user = user

    def period_arrays(self, year, month):
        return build_arrays(year, month)

    def can_access(self, emp_id: int) -> bool:
        u = self.user
        if u.is_system_admin:
            return False
        if u.is_top_management or u.is_auditor:
            return True
        emp = Employee.objects.filter(id=emp_id).first()
        if emp is None:
            return False
        if u.is_branch_manager:
            return u.branch_id == emp.branch_id
        prof = getattr(u, "employee_profile", None)
        return bool(prof and prof.id == emp.id)

    def default_employee(self) -> Optional[int]:
        prof = getattr(self.user, "employee_profile", None)
        return prof.id if prof else None


def example_employee(user) -> Optional[int]:
    """Номер працівника для прикладів у інтерфейсі: доступний користувачу І з даними в останньому періоді.

    (Власний профіль керівника не годиться: у нього зазвичай немає KPI-фактів.)
    """
    if user.is_system_admin:
        return None
    period = latest_period()
    if period is None:
        return None
    ctx = DbContext(user)
    with_data = list(
        KPIFact.objects.filter(year=period[0], month=period[1])
        .values_list("employee_id", flat=True).distinct().order_by("employee_id")
    )
    own = ctx.default_employee()
    ordered = ([own] if own in with_data else []) + [i for i in with_data if i != own]
    for emp_id in ordered:
        if ctx.can_access(emp_id):
            return emp_id
    return None


def kpi_names_for(emp_id: int) -> List[str]:
    emp = Employee.objects.filter(id=emp_id).select_related("position").first()
    if emp is None:
        return []
    return [d.name for d in KPIDefinition.objects.filter(position=emp.position).order_by("id")]


@functools.lru_cache(maxsize=2)
def get_copilot(dataset: str = "bank"):
    """Ледаче завантаження моделей. ``dataset``: «bank» (БД проєкту) або «rossmann» (реальний датасет)."""
    from .core.assistant import Copilot
    from .core.engine import Engine
    from .core.nlu import load_nlu
    from .core.train import ARTIFACTS

    nlu = load_nlu(ARTIFACTS / "nlu.pt")
    if dataset == "rossmann":
        from .core.rossmann_model import ROSS_DIR
        from .rossmann_service import kpi_names
        return Copilot(Engine.load(ROSS_DIR), nlu, kpi_names=kpi_names)
    return Copilot(Engine.load(), nlu, kpi_names=kpi_names_for)
