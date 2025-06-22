from django.shortcuts import render, get_object_or_404
from .models import QuestionCategory, Question


def category_list(request):
    """List all Question categories"""
    categories = QuestionCategory.objects.all()
    return render(request, 'question_answer/category_list.html', {'categories': categories})


def category_detail(request, slug):
    """Show questions within a specific category"""
    category = get_object_or_404(QuestionCategory, slug=slug)
    questions = category.questions.filter(is_published=True)
    return render(request, 'question_answer/category_detail.html', {
        'category': category,
        'questions': questions
    })


def question_detail(request, category_slug, question_slug):
    """Show detailed view of a specific question"""
    question = get_object_or_404(Question, category__slug=category_slug, slug=question_slug, is_published=True)
    return render(request, 'question_answer/question_detail.html', {'question': question})
