#!/bin/bash

# EC2 Deployment Script for ML Interview Notes
# This script helps set up your Django app on an EC2 instance

set -e  # Exit on any error

echo "🚀 ML Interview Notes - EC2 Deployment Script"
echo "=============================================="

# Configuration
APP_NAME="ml-interview-notes"
APP_DIR="/opt/$APP_NAME"
SERVICE_NAME="ml-interview-notes"
DOMAIN="your-domain.com"  # Update this

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_error "Please run this script as root (use sudo)"
    exit 1
fi

print_status "Starting EC2 deployment setup..."

# Update system packages
print_status "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install required packages
print_status "Installing required packages..."
apt-get install -y python3 python3-pip python3-venv nginx postgresql-client git

# Create application directory
print_status "Creating application directory..."
mkdir -p $APP_DIR
cd $APP_DIR

# Clone your repository (update with your actual repo URL)
print_status "Cloning application repository..."
git clone https://github.com/alliedbrother/ml-interview-notes.git .

# Create virtual environment
print_status "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional production dependencies
pip install gunicorn psycopg2-binary

# Create application user
print_status "Creating application user..."
useradd -r -s /bin/false $SERVICE_NAME || true

# Set proper permissions
chown -R $SERVICE_NAME:$SERVICE_NAME $APP_DIR
chmod +x $APP_DIR/scripts/*.sh

# Create environment file
print_status "Setting up environment configuration..."
cat > $APP_DIR/.env << EOF
# Production Environment Variables
DB_NAME=ml_interview_notes
DB_USER=your_rds_username
DB_PASSWORD=your_rds_password
DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
DB_PORT=5432
DEBUG=False
SECRET_KEY=your-production-secret-key-here
ALLOWED_HOSTS=$DOMAIN,localhost,127.0.0.1
EOF

# Run Django migrations
print_status "Running Django migrations..."
cd $APP_DIR
source venv/bin/activate
python manage.py migrate --noinput

# Collect static files
print_status "Collecting static files..."
python manage.py collectstatic --noinput

# Create Gunicorn service file
print_status "Setting up Gunicorn service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=ML Interview Notes Django Application
After=network.target

[Service]
Type=notify
User=$SERVICE_NAME
Group=$SERVICE_NAME
RuntimeDirectory=$SERVICE_NAME
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/gunicorn --workers 3 --bind unix:$APP_DIR/$SERVICE_NAME.sock ml_interview_notes.wsgi:application
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration
print_status "Setting up Nginx configuration..."
cat > /etc/nginx/sites-available/$SERVICE_NAME << EOF
server {
    listen 80;
    server_name $DOMAIN www.$DOMAIN;

    location = /favicon.ico { access_log off; log_not_found off; }
    
    location /static/ {
        root $APP_DIR;
    }

    location /media/ {
        root $APP_DIR;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:$APP_DIR/$SERVICE_NAME.sock;
    }
}
EOF

# Enable Nginx site
ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
nginx -t

# Start and enable services
print_status "Starting services..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME
systemctl enable nginx
systemctl restart nginx

# Set up firewall
print_status "Configuring firewall..."
ufw allow 'Nginx Full'
ufw allow OpenSSH
ufw --force enable

print_status "Deployment completed successfully!"
echo ""
print_warning "IMPORTANT: Please update the following files with your actual values:"
echo "1. $APP_DIR/.env - Update database credentials and secret key"
echo "2. /etc/nginx/sites-available/$SERVICE_NAME - Update domain name"
echo ""
print_status "Next steps:"
echo "1. Update .env file with your RDS credentials"
echo "2. Update Nginx configuration with your domain"
echo "3. Restart services: systemctl restart $SERVICE_NAME nginx"
echo "4. Test your application at http://$DOMAIN"
echo ""
print_status "Useful commands:"
echo "- Check service status: systemctl status $SERVICE_NAME"
echo "- View logs: journalctl -u $SERVICE_NAME -f"
echo "- Restart application: systemctl restart $SERVICE_NAME" 