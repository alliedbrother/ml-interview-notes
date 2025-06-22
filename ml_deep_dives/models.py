from django.db import models
from django.urls import reverse


class Category(models.Model):
    """Categories for ML Deep Dives: Math, Libraries, Machine Learning, NLP"""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, blank=True, help_text="CSS class for icon")
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['order', 'name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('ml_deep_dives:category_detail', kwargs={'slug': self.slug})


class Topic(models.Model):
    """Individual topics within each category"""
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='topics')
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    content = models.TextField()
    summary = models.TextField(blank=True, help_text="Brief summary of the topic")
    difficulty = models.CharField(
        max_length=20,
        choices=[
            ('beginner', 'Beginner'),
            ('intermediate', 'Intermediate'),
            ('advanced', 'Advanced'),
        ],
        default='intermediate'
    )
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated tags")
    order = models.IntegerField(default=0)
    is_published = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['category', 'order', 'title']
        unique_together = ['category', 'slug']

    def __str__(self):
        return f"{self.category.name} - {self.title}"

    def get_absolute_url(self):
        return reverse('ml_deep_dives:topic_detail', kwargs={'category_slug': self.category.slug, 'topic_slug': self.slug})

    def get_next_by_order(self):
        """Returns the next topic in the same category, ordered by the 'order' field."""
        return Topic.objects.filter(category=self.category, order__gt=self.order).order_by('order').first()

    def get_previous_by_order(self):
        """Returns the previous topic in the same category, ordered by the 'order' field."""
        return Topic.objects.filter(category=self.category, order__lt=self.order).order_by('-order').first()

    def get_tags_list(self):
        """Return tags as a list"""
        if self.tags:
            return [tag.strip() for tag in self.tags.split(',')]
        return []
