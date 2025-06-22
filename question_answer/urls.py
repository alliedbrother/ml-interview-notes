from django.urls import path
from . import views

app_name = 'question_answer'
 
urlpatterns = [
    path('', views.category_list, name='category_list'),
    path('<slug:slug>/', views.category_detail, name='category_detail'),
    path('<slug:category_slug>/<slug:question_slug>/', views.question_detail, name='question_detail'),
] 