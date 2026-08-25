#!/bin/bash

# Database restore script for ML Interview Notes
# This script restores a backup to an RDS PostgreSQL instance

# Configuration - Update these values for your RDS instance
RDS_ENDPOINT="ml-interview-notes.cfy2muw6o6vp.us-east-2.rds.amazonaws.com"
RDS_DB_NAME="ml_interview_notes"
RDS_USERNAME="mlinterviews"
RDS_PORT="5432"
BACKUP_FILE="backups/ml_interview_notes_20250622_013520.sql"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    echo "Please run backup_database.sh first or specify the correct backup file path."
    exit 1
fi

echo "Starting database restore to RDS..."
echo "RDS Endpoint: $RDS_ENDPOINT"
echo "Database: $RDS_DB_NAME"
echo "Username: $RDS_USERNAME"
echo "Backup file: $BACKUP_FILE"

# Restore the database
psql -h $RDS_ENDPOINT -U $RDS_USERNAME -d $RDS_DB_NAME -p $RDS_PORT -f $BACKUP_FILE

if [ $? -eq 0 ]; then
    echo "✅ Database restore completed successfully!"
    echo "Your data has been migrated to RDS."
else
    echo "❌ Database restore failed!"
    echo "Please check your RDS connection details and try again."
    exit 1
fi 