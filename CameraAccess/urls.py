"""
URL configuration for CameraAccess project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from camera.views import *
from django.conf import settings
from django.conf.urls.static import static

# from camera.views import VideoConsumer, home

urlpatterns = (
    [
        path("admin/", admin.site.urls),
        path("", home, name="home"),
        # path("", WLSTL, name="WLSTL"),
        path("get-sentence/", get_sentence, name="get-sentence"),
        path("clear-sentence/", clear_sentence, name="clear-sentence"),
        path("set-model/<str:model>/", SetModel, name="set-model"),
        # path('ws/video/', VideoConsumer.as_asgi()),
    ]
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    + static(settings.STATIC_URL, document_root=settings.STATIC_URL)
)
