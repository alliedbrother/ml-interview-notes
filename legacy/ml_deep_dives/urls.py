from django.urls import path
from . import views

app_name = 'ml_deep_dives'
 
urlpatterns = [
    path('', views.category_list, name='category_list'),
    path('<slug:slug>/', views.category_detail, name='category_detail'),
    path('<slug:category_slug>/<slug:topic_slug>/', views.topic_detail, name='topic_detail'),
] 