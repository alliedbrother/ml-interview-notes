# ML Interview Notes

A comprehensive Django web application for organizing and studying machine learning interview materials.

## Features

- **ML Deep Dives**: In-depth coverage of machine learning topics including mathematics, algorithms, and libraries
- **Question & Answer**: Structured Q&A format for interview preparation
- **System Design**: System design concepts and patterns for ML systems
- **Interactive Learning**: Web-based interface for easy navigation and study

## Technology Stack

- **Backend**: Django 5.2.3
- **Database**: PostgreSQL (with psycopg2)
- **Frontend**: HTML, CSS, JavaScript
- **Python**: 3.11+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alliedbrother/ml-interview-notes.git
cd ml-interview-notes
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py migrate
```

5. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

6. Run the development server:
```bash
python manage.py runserver
```

7. Visit http://127.0.0.1:8000/ in your browser

## Project Structure

```
ml-interview-notes/
├── core/                 # Core app with home page
├── ml_deep_dives/        # ML topics and deep dives
├── question_answer/      # Q&A format content
├── system_design/        # System design concepts
├── templates/           # HTML templates
├── static/              # Static files (CSS, JS, images)
└── manage.py           # Django management script
```

## Usage

- Navigate through different sections using the main menu
- Browse ML topics by category
- Study system design concepts
- Practice with Q&A format questions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE). 