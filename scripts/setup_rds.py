#!/usr/bin/env python3
"""
RDS Setup and Connection Test Script for ML Interview Notes
This script helps you set up and test your RDS PostgreSQL connection.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
import getpass

def test_local_connection():
    """Test connection to local PostgreSQL database"""
    print("🔍 Testing local database connection...")
    
    try:
        conn = psycopg2.connect(
            dbname="ml_interview_notes",
            user="mlinterviews",
            password="mlinterviews@123",
            host="localhost",
            port="5432"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print(f"✅ Local connection successful!")
        print(f"   PostgreSQL version: {version[0]}")
        
        # Check table count
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()[0]
        print(f"   Tables found: {table_count}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Local connection failed: {e}")
        return False

def test_rds_connection(rds_endpoint, rds_username, rds_password, rds_dbname, rds_port="5432"):
    """Test connection to RDS PostgreSQL database"""
    print(f"\n🔍 Testing RDS connection to {rds_endpoint}...")
    
    try:
        conn = psycopg2.connect(
            dbname=rds_dbname,
            user=rds_username,
            password=rds_password,
            host=rds_endpoint,
            port=rds_port
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        
        print(f"✅ RDS connection successful!")
        print(f"   PostgreSQL version: {version[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ RDS connection failed: {e}")
        return False

def create_rds_database(rds_endpoint, rds_username, rds_password, rds_dbname, rds_port="5432"):
    """Create the database on RDS if it doesn't exist"""
    print(f"\n🔧 Creating database '{rds_dbname}' on RDS...")
    
    try:
        # Connect to default postgres database first
        conn = psycopg2.connect(
            dbname="postgres",
            user=rds_username,
            password=rds_password,
            host=rds_endpoint,
            port=rds_port
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (rds_dbname,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(rds_dbname)))
            print(f"✅ Database '{rds_dbname}' created successfully!")
        else:
            print(f"ℹ️  Database '{rds_dbname}' already exists.")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False

def main():
    print("🚀 ML Interview Notes - RDS Setup Script")
    print("=" * 50)
    
    # Test local connection first
    if not test_local_connection():
        print("\n❌ Cannot proceed without local database connection.")
        sys.exit(1)
    
    # Get RDS details
    print("\n📝 Enter your RDS details:")
    rds_endpoint = input("RDS Endpoint (e.g., your-db.region.rds.amazonaws.com): ").strip()
    rds_username = input("RDS Username: ").strip()
    rds_password = getpass.getpass("RDS Password: ").strip()
    rds_dbname = input("RDS Database Name (default: ml_interview_notes): ").strip() or "ml_interview_notes"
    rds_port = input("RDS Port (default: 5432): ").strip() or "5432"
    
    if not all([rds_endpoint, rds_username, rds_password]):
        print("❌ All fields are required!")
        sys.exit(1)
    
    # Create database if needed
    if not create_rds_database(rds_endpoint, rds_username, rds_password, rds_dbname, rds_port):
        sys.exit(1)
    
    # Test RDS connection
    if not test_rds_connection(rds_endpoint, rds_username, rds_password, rds_dbname, rds_port):
        sys.exit(1)
    
    # Generate environment file
    print(f"\n📄 Generating .env file for production...")
    env_content = f"""# Production Environment Variables for ML Interview Notes
DB_NAME={rds_dbname}
DB_USER={rds_username}
DB_PASSWORD={rds_password}
DB_HOST={rds_endpoint}
DB_PORT={rds_port}
DEBUG=False
SECRET_KEY=your-production-secret-key-here
ALLOWED_HOSTS=your-ec2-public-ip,your-domain.com
"""
    
    with open('.env.production', 'w') as f:
        f.write(env_content)
    
    print("✅ .env.production file created!")
    print("\n📋 Next steps:")
    print("1. Update the RDS endpoint in scripts/restore_database.sh")
    print("2. Run: chmod +x scripts/backup_database.sh")
    print("3. Run: ./scripts/backup_database.sh")
    print("4. Run: ./scripts/restore_database.sh")
    print("5. Copy .env.production to your EC2 instance")
    print("6. Update SECRET_KEY and ALLOWED_HOSTS in .env.production")

if __name__ == "__main__":
    main() 