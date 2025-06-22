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
├── scripts/             # Deployment and migration scripts
└── manage.py           # Django management script
```

## Usage

- Navigate through different sections using the main menu
- Browse ML topics by category
- Study system design concepts
- Practice with Q&A format questions

## AWS Deployment

### Prerequisites
- AWS Account
- EC2 instance (Ubuntu 20.04+ recommended)
- RDS PostgreSQL instance

### Quick Deployment

1. **Set up RDS Database**:
   ```bash
   # Follow the detailed guide in scripts/aws_rds_setup.md
   # Or use the automated setup script:
   python3 scripts/setup_rds.py
   ```

2. **Backup Local Database**:
   ```bash
   ./scripts/backup_database.sh
   ```

3. **Restore to RDS**:
   ```bash
   # Update scripts/restore_database.sh with your RDS details
   ./scripts/restore_database.sh
   ```

4. **Deploy to EC2**:
   ```bash
   # Copy deploy script to your EC2 instance
   sudo ./scripts/deploy_to_ec2.sh
   ```

### Manual Deployment Steps

1. **Create RDS PostgreSQL Instance**:
   - Engine: PostgreSQL 14.x
   - Instance: db.t3.micro (free tier) or larger
   - Storage: 20 GB minimum
   - Enable public access for initial setup

2. **Configure Security Groups**:
   - Allow PostgreSQL (5432) from your IP and EC2
   - Allow HTTP (80) and HTTPS (443) for web traffic

3. **Migrate Data**:
   ```bash
   # Backup local database
   pg_dump -U mlinterviews -h localhost ml_interview_notes > backup.sql
   
   # Restore to RDS
   psql -h <rds-endpoint> -U <username> -d ml_interview_notes -f backup.sql
   ```

4. **Deploy Application**:
   - Clone repository on EC2
   - Install dependencies (Python, Nginx, Gunicorn)
   - Configure environment variables
   - Set up systemd service
   - Configure Nginx

### Environment Variables

Create a `.env` file for production:
```bash
DB_NAME=ml_interview_notes
DB_USER=your_rds_username
DB_PASSWORD=your_rds_password
DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
DB_PORT=5432
DEBUG=False
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=your-domain.com,your-ec2-ip
```

## Database Management

### Check Database Contents
```bash
python check_db.py
```

### Create Sample Data
```bash
python manage.py create_sample_data
```

### Backup Database
```bash
./scripts/backup_database.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).
