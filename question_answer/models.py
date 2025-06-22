from django.db import models
from django.urls import reverse


class QuestionCategory(models.Model):
    """Categories for interview questions"""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, blank=True, help_text="CSS class for icon")
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Question Categories"
        ordering = ['order', 'name']

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('question_answer:category_detail', kwargs={'slug': self.slug})


class Question(models.Model):
    """Interview questions with answers"""
    category = models.ForeignKey(QuestionCategory, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    slug = models.SlugField(max_length=200, unique=True)
    answer = models.TextField()
    explanation = models.TextField(blank=True, help_text="Detailed explanation of the answer")
    code_example = models.TextField(blank=True, help_text="Code example if applicable")
    difficulty = models.CharField(
        max_length=20,
        choices=[
            ('easy', 'Easy'),
            ('medium', 'Medium'),
            ('hard', 'Hard'),
        ],
        default='medium'
    )
    question_type = models.CharField(
        max_length=20,
        choices=[
            ('conceptual', 'Conceptual'),
            ('coding', 'Coding'),
            ('system_design', 'System Design'),
            ('math', 'Mathematics'),
            ('case_study', 'Case Study'),
        ],
        default='conceptual'
    )
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated tags")
    order = models.IntegerField(default=0)
    is_published = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['category', 'order', 'question_text']
        unique_together = ['category', 'slug']

    def __str__(self):
        return f"{self.category.name} - {self.question_text[:50]}..."

    def get_absolute_url(self):
        return reverse('question_answer:question_detail', kwargs={'category_slug': self.category.slug, 'question_slug': self.slug})

    def get_tags_list(self):
        """Return tags as a list"""
        if self.tags:
            return [tag.strip() for tag in self.tags.split(',')]
        return []
