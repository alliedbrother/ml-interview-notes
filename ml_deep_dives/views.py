from django.shortcuts import render, get_object_or_404
from .models import Category, Topic


def category_list(request):
    """List all ML Deep Dives categories"""
    categories = Category.objects.all()
    return render(request, 'ml_deep_dives/category_list.html', {'categories': categories})


def category_detail(request, slug):
    """Show topics within a specific category"""
    category = get_object_or_404(Category, slug=slug)
    topics = category.topics.filter(is_published=True)
    all_categories = Category.objects.all()
    return render(request, 'ml_deep_dives/category_detail.html', {
        'category': category,
        'topics': topics,
        'categories': all_categories,
    })


def topic_detail(request, category_slug, topic_slug):
    """Show detailed view of a specific topic"""
    topic = get_object_or_404(Topic, category__slug=category_slug, slug=topic_slug, is_published=True)
    return render(request, 'ml_deep_dives/topic_detail.html', {'topic': topic})
