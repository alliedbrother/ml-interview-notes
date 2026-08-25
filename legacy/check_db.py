#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_interview_notes.settings')
django.setup()

from ml_deep_dives.models import Category, Topic
from question_answer.models import QuestionCategory, Question
from system_design.models import SystemDesignCategory, SystemDesign

def check_database():
    print("=== DATABASE CONTENTS ===\n")
    
    # Check ML Deep Dives
    print("1. ML DEEP DIVES:")
    print("-" * 50)
    
    categories = Category.objects.all()
    print(f"Categories: {categories.count()}")
    for cat in categories:
        print(f"  - {cat.name} (slug: {cat.slug})")
        topics = cat.topics.all()
        print(f"    Topics: {topics.count()}")
        for topic in topics:
            print(f"      * {topic.title} (difficulty: {topic.difficulty})")
    print()
    
    # Check Question & Answer
    print("2. QUESTION & ANSWER:")
    print("-" * 50)
    
    q_categories = QuestionCategory.objects.all()
    print(f"Question Categories: {q_categories.count()}")
    for cat in q_categories:
        print(f"  - {cat.name} (slug: {cat.slug})")
        questions = cat.questions.all()
        print(f"    Questions: {questions.count()}")
        for q in questions:
            print(f"      * {q.question_text[:50]}... (difficulty: {q.difficulty})")
    print()
    
    # Check System Design
    print("3. SYSTEM DESIGN:")
    print("-" * 50)
    
    sd_categories = SystemDesignCategory.objects.all()
    print(f"System Design Categories: {sd_categories.count()}")
    for cat in sd_categories:
        print(f"  - {cat.name} (slug: {cat.slug})")
        designs = cat.designs.all()
        print(f"    Designs: {designs.count()}")
        for design in designs:
            print(f"      * {design.title} (difficulty: {design.difficulty})")
    print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"Total Categories: {categories.count() + q_categories.count() + sd_categories.count()}")
    print(f"Total Topics: {Topic.objects.count()}")
    print(f"Total Questions: {Question.objects.count()}")
    print(f"Total System Designs: {SystemDesign.objects.count()}")

if __name__ == "__main__":
    check_database() 