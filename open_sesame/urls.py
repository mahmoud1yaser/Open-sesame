from django.urls import path

from . import views

app_name = "open_sesame"

urlpatterns = [
    path("/send", views.index, name="index"),
]
