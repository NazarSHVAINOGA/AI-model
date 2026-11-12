from django.urls import path

from .views import copilot_ask, copilot_status

urlpatterns = [
    path("copilot/ask/", copilot_ask, name="copilot-ask"),
    path("copilot/status/", copilot_status, name="copilot-status"),
]
