from django.urls import path
from SIMON1 import views

urlpatterns = [
    path('upload/', upload_image, name='upload_image'),
]