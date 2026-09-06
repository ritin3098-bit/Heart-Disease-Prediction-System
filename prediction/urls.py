from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('result/', views.result, name='result'),
    path('train/', views.train_models, name='train_models'),
    path('data-analysis/', views.data_analysis, name='data_analysis'),
    path('history/', views.history, name='history'),
    path('logout/', views.logout_view, name='logout'),
]
