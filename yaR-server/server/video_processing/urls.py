from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_video, name="upload_video"),
    path("ocr/", views.ocr_view, name="ocr_view"),
    path("hello/", views.hello_world, name="hello_world"),
]
