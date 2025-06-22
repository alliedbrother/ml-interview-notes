from django.db import models
from django.urls import reverse


class SystemDesignCategory(models.Model):
    """Categories for System Design: ML Systems, Data Pipelines, etc."""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, blank=True, help_text="CSS class for icon")
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "System Design Categories"
        ordering = ['order', 'name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('system_design:category_detail', kwargs={'slug': self.slug})


class SystemDesign(models.Model):
    """ML System Design patterns and architectures"""
    category = models.ForeignKey(SystemDesignCategory, on_delete=models.CASCADE, related_name='designs')
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    problem_statement = models.TextField()
    solution_overview = models.TextField()
    architecture_diagram = models.ImageField(upload_to='system_designs/', blank=True, null=True)
    key_components = models.TextField(help_text="Key components of the system")
    scalability_considerations = models.TextField(blank=True)
    trade_offs = models.TextField(blank=True)
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
        return reverse('system_design:design_detail', kwargs={'category_slug': self.category.slug, 'design_slug': self.slug})

    def get_tags_list(self):
        """Return tags as a list"""
        if self.tags:
            return [tag.strip() for tag in self.tags.split(',')]
        return []
