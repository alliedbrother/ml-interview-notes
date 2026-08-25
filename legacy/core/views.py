from django.shortcuts import render
from ml_deep_dives.models import Category as MLDeepDivesCategory
from system_design.models import SystemDesignCategory
from question_answer.models import QuestionCategory


def home(request):
    """Home page view"""
    context = {
        'ml_deep_dives_categories': MLDeepDivesCategory.objects.all(),
        'system_design_categories': SystemDesignCategory.objects.all(),
        'question_categories': QuestionCategory.objects.all(),
    }
    return render(request, 'core/home.html', context)
