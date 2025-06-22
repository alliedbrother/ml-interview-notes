from django.contrib import admin
from .models import SystemDesignCategory, SystemDesign


@admin.register(SystemDesignCategory)
class SystemDesignCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'order', 'created_at']
    list_editable = ['order']
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ['name', 'description']
    ordering = ['order', 'name']


@admin.register(SystemDesign)
class SystemDesignAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'difficulty', 'order', 'is_published', 'created_at']
    list_filter = ['category', 'difficulty', 'is_published', 'created_at']
    list_editable = ['order', 'is_published']
    prepopulated_fields = {'slug': ('title',)}
    search_fields = ['title', 'problem_statement', 'solution_overview']
    ordering = ['category', 'order', 'title']
    date_hierarchy = 'created_at'
