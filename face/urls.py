from django.urls import path
from . import views

urlpatterns = [
    path('face', views.face, name='face'),
]