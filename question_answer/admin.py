from django.contrib import admin
from .models import QuestionCategory, Question


@admin.register(QuestionCategory)
class QuestionCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'order', 'created_at']
    list_editable = ['order']
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ['name', 'description']
    ordering = ['order', 'name']


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['question_text', 'category', 'difficulty', 'question_type', 'order', 'is_published', 'created_at']
    list_filter = ['category', 'difficulty', 'question_type', 'is_published', 'created_at']
    list_editable = ['order', 'is_published']
    prepopulated_fields = {'slug': ('question_text',)}
    search_fields = ['question_text', 'answer', 'explanation']
    ordering = ['category', 'order', 'question_text']
    date_hierarchy = 'created_at'
