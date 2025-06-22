from django.shortcuts import render, get_object_or_404
from .models import SystemDesignCategory, SystemDesign


def category_list(request):
    """List all System Design categories"""
    categories = SystemDesignCategory.objects.all()
    return render(request, 'system_design/category_list.html', {'categories': categories})


def category_detail(request, slug):
    """Show designs within a specific category"""
    category = get_object_or_404(SystemDesignCategory, slug=slug)
    designs = category.designs.filter(is_published=True)
    return render(request, 'system_design/category_detail.html', {
        'category': category,
        'designs': designs
    })


def design_detail(request, category_slug, design_slug):
    """Show detailed view of a specific system design"""
    design = get_object_or_404(SystemDesign, category__slug=category_slug, slug=design_slug, is_published=True)
    return render(request, 'system_design/design_detail.html', {'design': design})
