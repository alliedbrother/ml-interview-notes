from django.core.management.base import BaseCommand
from ml_deep_dives.models import Category, Topic


class Command(BaseCommand):
    help = 'Create sample ML Deep Dives categories and topics'

    def handle(self, *args, **options):
        self.stdout.write('Deleting existing ML Deep Dives data...')
        Topic.objects.all().delete()
        Category.objects.all().delete()
        self.stdout.write('Existing data deleted.')

        self.stdout.write('Creating sample ML Deep Dives data...')

        # Create categories
        categories_data = [
            {
                'name': 'Math for ML',
                'slug': 'math',
                'description': 'Mathematical foundations essential for Machine Learning including linear algebra, calculus, probability, and optimization.',
                'icon': 'fas fa-calculator',
                'order': 1
            },
            {
                'name': 'Libraries for ML',
                'slug': 'libraries',
                'description': 'Essential Python libraries and frameworks for Machine Learning development and deployment.',
                'icon': 'fas fa-code',
                'order': 2
            },
            {
                'name': 'Machine Learning',
                'slug': 'ml',
                'description': 'Core Machine Learning concepts, algorithms, model evaluation, and best practices.',
                'icon': 'fas fa-robot',
                'order': 3
            },
            {
                'name': 'Deep Learning',
                'slug': 'deep-learning',
                'description': 'Neural networks, deep architectures, optimization techniques, and advanced deep learning concepts.',
                'icon': 'fas fa-network-wired',
                'order': 4
            },
            {
                'name': 'NLP',
                'slug': 'nlp',
                'description': 'Natural Language Processing techniques, models, and applications for text and speech processing.',
                'icon': 'fas fa-language',
                'order': 5
            }
        ]

        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                slug=cat_data['slug'],
                defaults=cat_data
            )
            if created:
                self.stdout.write(f'Created category: {category.name}')
            else:
                self.stdout.write(f'Category already exists: {category.name}')

        # Create topics for Math for ML
        math_category = Category.objects.get(slug='math')
        math_topics = [
            {
                'title': 'Linear Algebra',
                'slug': 'math-linear-algebra',
                'difficulty': 'intermediate',
                'tags': 'vectors, matrices, eigenvalues, SVD',
                'order': 1
            },
            {
                'title': 'Calculus',
                'slug': 'math-calculus',
                'difficulty': 'intermediate',
                'tags': 'derivatives, gradients, optimization',
                'order': 2
            },
            {
                'title': 'Probability & Statistics',
                'slug': 'math-probability-statistics',
                'difficulty': 'intermediate',
                'tags': 'probability, distributions, MLE, hypothesis testing',
                'order': 3
            },
            {
                'title': 'Optimization Techniques',
                'slug': 'math-optimization',
                'difficulty': 'advanced',
                'tags': 'convex optimization, gradient descent, SGD',
                'order': 4
            },
            {
                'title': 'Discrete Mathematics',
                'slug': 'math-discrete-math',
                'difficulty': 'intermediate',
                'tags': 'combinatorics, graph theory, networks',
                'order': 5
            }
        ]

        for topic_data in math_topics:
            topic, created = Topic.objects.get_or_create(
                category=math_category,
                slug=topic_data['slug'],
                defaults={
                    **topic_data,
                    'content': 'Content coming soon...',
                    'summary': f'Comprehensive guide to {topic_data["title"].lower()} for machine learning applications.'
                }
            )
            if created:
                self.stdout.write(f'Created topic: {topic.title}')

        # Create topics for Libraries
        libraries_category = Category.objects.get(slug='libraries')
        libraries_topics = [
            {
                'title': 'TensorFlow',
                'slug': 'lib-tensorflow',
                'difficulty': 'intermediate',
                'tags': 'deep learning, neural networks, keras',
                'order': 1
            },
            {
                'title': 'PyTorch',
                'slug': 'lib-pytorch',
                'difficulty': 'intermediate',
                'tags': 'deep learning, dynamic graphs, autograd',
                'order': 2
            },
            {
                'title': 'Scikit-learn',
                'slug': 'lib-scikit-learn',
                'difficulty': 'beginner',
                'tags': 'machine learning, algorithms, preprocessing',
                'order': 3
            },
            {
                'title': 'Pandas & NumPy',
                'slug': 'lib-pandas-numpy',
                'difficulty': 'beginner',
                'tags': 'data manipulation, arrays, dataframes',
                'order': 4
            },
            {
                'title': 'Visualization',
                'slug': 'lib-visualization',
                'difficulty': 'beginner',
                'tags': 'matplotlib, seaborn, plotting',
                'order': 5
            },
            {
                'title': 'Boosting Libraries',
                'slug': 'lib-boosting',
                'difficulty': 'intermediate',
                'tags': 'xgboost, lightgbm, catboost',
                'order': 6
            }
        ]

        for topic_data in libraries_topics:
            topic, created = Topic.objects.get_or_create(
                category=libraries_category,
                slug=topic_data['slug'],
                defaults={
                    **topic_data,
                    'content': 'Content coming soon...',
                    'summary': f'Complete guide to {topic_data["title"].lower()} for machine learning development.'
                }
            )
            if created:
                self.stdout.write(f'Created topic: {topic.title}')

        # Create topics for Machine Learning
        ml_category = Category.objects.get(slug='ml')
        ml_topics = [
            {
                'title': 'Types of ML',
                'slug': 'ml-types',
                'difficulty': 'beginner',
                'tags': 'supervised, unsupervised, reinforcement learning',
                'order': 1
            },
            {
                'title': 'ML Algorithms',
                'slug': 'ml-algorithms',
                'difficulty': 'intermediate',
                'tags': 'regression, classification, clustering',
                'order': 2
            },
            {
                'title': 'Model Evaluation',
                'slug': 'ml-model-evaluation',
                'difficulty': 'intermediate',
                'tags': 'cross-validation, metrics, confusion matrix',
                'order': 3
            },
            {
                'title': 'Feature Engineering',
                'slug': 'ml-feature-engineering',
                'difficulty': 'intermediate',
                'tags': 'feature selection, scaling, normalization',
                'order': 4
            },
            {
                'title': 'Hyperparameter Tuning',
                'slug': 'ml-hyperparameter-tuning',
                'difficulty': 'advanced',
                'tags': 'grid search, random search, bayesian optimization',
                'order': 5
            },
            {
                'title': 'Ethics in ML',
                'slug': 'ml-ethics',
                'difficulty': 'intermediate',
                'tags': 'bias, fairness, explainable AI',
                'order': 6
            }
        ]

        for topic_data in ml_topics:
            topic, created = Topic.objects.get_or_create(
                category=ml_category,
                slug=topic_data['slug'],
                defaults={
                    **topic_data,
                    'content': 'Content coming soon...',
                    'summary': f'Essential concepts in {topic_data["title"].lower()} for machine learning practitioners.'
                }
            )
            if created:
                self.stdout.write(f'Created topic: {topic.title}')

        # Create topics for Deep Learning
        dl_category = Category.objects.get(slug='deep-learning')
        dl_topics = [
            {
                'title': 'Neural Networks',
                'slug': 'dl-neural-networks',
                'difficulty': 'intermediate',
                'tags': 'feedforward, backpropagation, activation functions',
                'order': 1
            },
            {
                'title': 'Deep Architectures',
                'slug': 'dl-deep-architectures',
                'difficulty': 'advanced',
                'tags': 'CNNs, RNNs, LSTM, GRU',
                'order': 2
            },
            {
                'title': 'Optimization Techniques',
                'slug': 'dl-optimization',
                'difficulty': 'advanced',
                'tags': 'SGD, Adam, learning rate scheduling',
                'order': 3
            },
            {
                'title': 'Transfer Learning',
                'slug': 'dl-transfer-learning',
                'difficulty': 'intermediate',
                'tags': 'pretrained models, fine-tuning',
                'order': 4
            },
            {
                'title': 'Generative Models',
                'slug': 'dl-generative-models',
                'difficulty': 'advanced',
                'tags': 'GANs, VAEs, generative AI',
                'order': 5
            },
            {
                'title': 'Deep Reinforcement Learning',
                'slug': 'dl-deep-rl',
                'difficulty': 'advanced',
                'tags': 'Q-learning, policy gradients, DQN',
                'order': 6
            }
        ]

        for topic_data in dl_topics:
            topic, created = Topic.objects.get_or_create(
                category=dl_category,
                slug=topic_data['slug'],
                defaults={
                    **topic_data,
                    'content': 'Content coming soon...',
                    'summary': f'Advanced concepts in {topic_data["title"].lower()} for deep learning applications.'
                }
            )
            if created:
                self.stdout.write(f'Created topic: {topic.title}')

        # Create topics for NLP
        nlp_category = Category.objects.get(slug='nlp')
        nlp_topics = [
            {
                'title': 'Text Preprocessing',
                'slug': 'nlp-text-preprocessing',
                'difficulty': 'beginner',
                'tags': 'tokenization, stemming, lemmatization',
                'order': 1
            },
            {
                'title': 'Language Models',
                'slug': 'nlp-language-models',
                'difficulty': 'advanced',
                'tags': 'n-grams, transformers, BERT, GPT',
                'order': 2
            },
            {
                'title': 'Text Classification',
                'slug': 'nlp-text-classification',
                'difficulty': 'intermediate',
                'tags': 'sentiment analysis, NER, spam detection',
                'order': 3
            },
            {
                'title': 'Machine Translation',
                'slug': 'nlp-machine-translation',
                'difficulty': 'advanced',
                'tags': 'sequence-to-sequence, attention mechanism',
                'order': 4
            },
            {
                'title': 'Speech Recognition',
                'slug': 'nlp-speech-recognition',
                'difficulty': 'advanced',
                'tags': 'phoneme recognition, speech-to-text',
                'order': 5
            },
            {
                'title': 'Text Generation',
                'slug': 'nlp-text-generation',
                'difficulty': 'advanced',
                'tags': 'language generation, text summarization',
                'order': 6
            }
        ]

        for topic_data in nlp_topics:
            topic, created = Topic.objects.get_or_create(
                category=nlp_category,
                slug=topic_data['slug'],
                defaults={
                    **topic_data,
                    'content': 'Content coming soon...',
                    'summary': f'Natural language processing techniques for {topic_data["title"].lower()}.'
                }
            )
            if created:
                self.stdout.write(f'Created topic: {topic.title}')

        self.stdout.write(
            self.style.SUCCESS('Successfully created sample ML Deep Dives data!')
        ) 