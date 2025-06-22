#!/bin/bash

# EC2 Setup Script for ML Interview Notes
# Run this script on your EC2 instance as root or with sudo

set -e

echo "Starting EC2 setup for ML Interview Notes..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt install -y python3 python3-pip python3-venv nginx postgresql postgresql-contrib git curl

# Create application user
echo "Creating application user..."
sudo useradd -m -s /bin/bash ubuntu || echo "User ubuntu already exists"

# Clone the repository
echo "Cloning the repository..."
cd /home/ubuntu
sudo -u ubuntu git clone https://github.com/alliedbrother/ml-interview-notes.git

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
cd /home/ubuntu/ml-interview-notes
sudo -u ubuntu python3 -m venv venv
sudo -u ubuntu /home/ubuntu/ml-interview-notes/venv/bin/pip install --upgrade pip
sudo -u ubuntu /home/ubuntu/ml-interview-notes/venv/bin/pip install -r requirements.txt

# Set up PostgreSQL
echo "Setting up PostgreSQL..."
sudo -u postgres createdb ml_interview_notes
sudo -u postgres psql -c "CREATE USER mluser WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ml_interview_notes TO mluser;"

# Create environment file
echo "Creating environment file..."
sudo -u ubuntu tee /home/ubuntu/ml-interview-notes/.env > /dev/null <<EOF
DEBUG=False
SECRET_KEY=your-super-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,your-ec2-ip
DB_NAME=ml_interview_notes
DB_USER=mluser
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
EOF

# Set up Gunicorn
echo "Setting up Gunicorn..."
sudo cp /home/ubuntu/ml-interview-notes/gunicorn.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gunicorn

# Set up Nginx
echo "Setting up Nginx..."
sudo cp /home/ubuntu/ml-interview-notes/nginx.conf /etc/nginx/sites-available/ml-interview-notes
sudo ln -sf /etc/nginx/sites-available/ml-interview-notes /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Collect static files
echo "Collecting static files..."
cd /home/ubuntu/ml-interview-notes
sudo -u ubuntu /home/ubuntu/ml-interview-notes/venv/bin/python manage.py collectstatic --noinput

# Run migrations
echo "Running database migrations..."
sudo -u ubuntu /home/ubuntu/ml-interview-notes/venv/bin/python manage.py migrate

# Set proper permissions
echo "Setting proper permissions..."
sudo chown -R ubuntu:www-data /home/ubuntu/ml-interview-notes
sudo chmod -R 755 /home/ubuntu/ml-interview-notes

# Start services
echo "Starting services..."
sudo systemctl start gunicorn
sudo systemctl restart nginx

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
sudo ufw --force enable

echo "EC2 setup completed successfully!"
echo "Your application should now be accessible at http://your-ec2-ip"
echo "Don't forget to:"
echo "1. Update the domain name in nginx.conf"
echo "2. Update the SECRET_KEY in .env"
echo "3. Update the database password in .env"
echo "4. Configure your domain DNS to point to this EC2 instance" 