from django.urls import path
from . import views

urlpatterns = [
    path('face', views.face_analysis, name='face_analysis'),
]
