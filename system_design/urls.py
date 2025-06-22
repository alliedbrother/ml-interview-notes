from django.urls import path
from . import views

app_name = 'system_design'
 
urlpatterns = [
    path('', views.category_list, name='category_list'),
    path('<slug:slug>/', views.category_detail, name='category_detail'),
    path('<slug:category_slug>/<slug:design_slug>/', views.design_detail, name='design_detail'),
] 