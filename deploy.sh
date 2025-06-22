#!/bin/bash

# Exit on any error
set -e

echo "Starting deployment..."

# Navigate to project directory
cd /home/ubuntu/ml-interview-notes

# Pull latest changes
echo "Pulling latest changes from git..."
git pull origin main

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Run database migrations
echo "Running database migrations..."
python manage.py migrate

# Restart Gunicorn service
echo "Restarting Gunicorn service..."
sudo systemctl restart gunicorn

# Restart Nginx
echo "Restarting Nginx..."
sudo systemctl restart nginx

echo "Deployment completed successfully!" 