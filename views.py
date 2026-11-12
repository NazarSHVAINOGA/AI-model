"""API Copilot: ``POST /api/copilot/ask/`` та ``GET /api/copilot/status/``."""
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.throttling import UserRateThrottle
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .core.assistant import MAX_QUESTION_LEN
from .core.train import ARTIFACTS
from .service import DbContext, example_employee, get_copilot


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def copilot_status(request):
    need = ["forecaster.pt", "conformal.json", "nlu.pt"]
    missing = [n for n in need if not (ARTIFACTS / n).exists()]
    ready = not missing
    calibrated = "db" if (ARTIFACTS / "conformal_db.json").exists() else "simulation"
    from . import rossmann_service as rs
    ros_ok = rs.available()
    return Response({
        "ready": ready, "missing": missing, "calibrated_on": calibrated,
        "example_employee": example_employee(request.user),
        "datasets": {"bank": ready, "rossmann": ros_ok},
        "example_store": rs.example_store() if ros_ok and not request.user.is_system_admin else None,
    })


class CopilotThrottle(UserRateThrottle):
    """Не більше 20 запитів на хвилину від одного користувача (захист від зловживань і перевантаження GPU/CPU)."""
    scope = "copilot"
    rate = "20/min"


@api_view(["POST"])
@permission_classes([IsAuthenticated])
@throttle_classes([CopilotThrottle])
def copilot_ask(request):
    if request.user.is_system_admin:
        return Response({"detail": "Доступ заборонено."}, status=status.HTTP_403_FORBIDDEN)
    question = str(request.data.get("question", "")).strip()
    if not question or len(question) > MAX_QUESTION_LEN:
        return Response({"detail": f"Питання має містити від 1 до {MAX_QUESTION_LEN} символів."}, status=400)
    dataset = request.data.get("dataset", "bank")
    if dataset not in ("bank", "rossmann"):
        return Response({"detail": "Невідомий набір даних."}, status=400)
    try:
        copilot = get_copilot(dataset)
        if dataset == "rossmann":
            from .rossmann_service import RossmannContext, available
            if not available():
                raise FileNotFoundError
    except FileNotFoundError:
        return Response(
            {"detail": "Модель Copilot ще не навчена: виконайте python manage.py copilot_train."},
            status=503,
        )
    year, month = request.data.get("year"), request.data.get("month")
    try:
        year = int(year) if year else None
        month = int(month) if month else None
    except (TypeError, ValueError):
        return Response({"detail": "Некоректний період."}, status=400)
    if dataset == "rossmann":
        from .rossmann_service import RossmannContext
        return Response(copilot.answer(question, RossmannContext(request.user), None, None))
    return Response(copilot.answer(question, DbContext(request.user), year, month))
