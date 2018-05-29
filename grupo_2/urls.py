from django.conf.urls import url, include
from django.contrib import admin
from django.views.generic import RedirectView

urlpatterns = [
    url(r'^', include('menu.urls')),
    url(r'^taller_3', include('taller_3.urls')),
    url(r'^menu', RedirectView.as_view(url='/', permanent = True)),
]
