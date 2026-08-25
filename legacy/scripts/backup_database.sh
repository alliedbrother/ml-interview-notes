#!/bin/bash

# Database backup script for ML Interview Notes
# This script creates a backup of the current PostgreSQL database

# Configuration
DB_NAME="ml_interview_notes"
DB_USER="mlinterviews"
DB_HOST="localhost"
DB_PORT="5432"
BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/ml_interview_notes_${TIMESTAMP}.sql"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

echo "Starting database backup..."
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Backup file: $BACKUP_FILE"

# Create the backup
pg_dump -U $DB_USER -h $DB_HOST -p $DB_PORT $DB_NAME > $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "✅ Database backup completed successfully!"
    echo "Backup file: $BACKUP_FILE"
    echo "File size: $(du -h $BACKUP_FILE | cut -f1)"
else
    echo "❌ Database backup failed!"
    exit 1
fi 