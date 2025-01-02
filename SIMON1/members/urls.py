from django.urls import path

urlpatterns = [
    path('upload/', upload_image, name='upload_image'),
]