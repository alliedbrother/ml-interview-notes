# AWS RDS Setup Guide for ML Interview Notes

This guide will help you set up Amazon RDS PostgreSQL for your ML Interview Notes application.

## Prerequisites

- AWS Account
- AWS CLI configured (optional but recommended)
- Basic knowledge of AWS Console

## Step 1: Create RDS PostgreSQL Instance

### 1.1 Access AWS RDS Console
1. Log into AWS Console
2. Navigate to **RDS** service
3. Click **Create database**

### 1.2 Choose Database Settings
- **Choose a database creation method**: Standard create
- **Engine type**: PostgreSQL
- **Version**: 14.x (compatible with your local setup)
- **Templates**: Free tier (for testing) or Production

### 1.3 Configure Database Settings
- **DB instance identifier**: `ml-interview-notes-db`
- **Master username**: `mlinterviews` (or your preferred username)
- **Master password**: Create a strong password (save this!)
- **Confirm password**: Re-enter the password

### 1.4 Configure Instance Settings
- **DB instance class**: 
  - Free tier: `db.t3.micro`
  - Production: `db.t3.small` or larger
- **Storage type**: General Purpose SSD (gp2)
- **Allocated storage**: 20 GB (minimum)
- **Enable storage autoscaling**: Yes (recommended)

### 1.5 Configure Connectivity
- **Virtual Private Cloud (VPC)**: Default VPC
- **Subnet group**: Default
- **Publicly accessible**: **Yes** (for initial setup, change to No for production)
- **VPC security group**: Create new
  - Security group name: `ml-interview-notes-rds-sg`
  - Description: Security group for ML Interview Notes RDS
- **Availability Zone**: No preference
- **Database port**: 5432

### 1.6 Configure Security Group Rules
After creating the security group, add these inbound rules:

| Type | Protocol | Port | Source | Description |
|------|----------|------|--------|-------------|
| PostgreSQL | TCP | 5432 | Your IP address | Local development access |
| PostgreSQL | TCP | 5432 | EC2 security group | EC2 instance access |

### 1.7 Configure Database Authentication
- **Database authentication options**: Password authentication

### 1.8 Configure Additional Settings
- **Initial database name**: `ml_interview_notes`
- **Backup retention period**: 7 days (free tier) or 30 days (production)
- **Enable automated backups**: Yes
- **Backup window**: No preference
- **Enable encryption**: Yes (recommended)
- **Enable deletion protection**: Yes (recommended for production)

### 1.9 Review and Create
- Review all settings
- Click **Create database**

## Step 2: Wait for Database Creation
- Database creation takes 5-10 minutes
- Status will change from "Creating" to "Available"

## Step 3: Get Connection Details
Once the database is available:

1. Click on your database instance
2. Note the **Endpoint** (e.g., `ml-interview-notes-db.abc123.us-east-1.rds.amazonaws.com`)
3. Note the **Port** (5432)
4. Note the **Database name** (`ml_interview_notes`)
5. Note the **Username** and **Password**

## Step 4: Test Connection

### 4.1 Using the Setup Script
```bash
# Run the RDS setup script
python3 scripts/setup_rds.py
```

### 4.2 Manual Connection Test
```bash
# Test connection using psql
psql -h <your-rds-endpoint> -U <your-username> -d ml_interview_notes -p 5432
```

## Step 5: Migrate Your Data

### 5.1 Backup Local Database
```bash
# Make scripts executable
chmod +x scripts/backup_database.sh

# Create backup
./scripts/backup_database.sh
```

### 5.2 Update Restore Script
Edit `scripts/restore_database.sh` and update:
- `RDS_ENDPOINT` with your RDS endpoint
- `RDS_USERNAME` with your RDS username
- `RDS_DB_NAME` with your database name

### 5.3 Restore to RDS
```bash
# Make restore script executable
chmod +x scripts/restore_database.sh

# Restore data
./scripts/restore_database.sh
```

## Step 6: Update Django Settings

### 6.1 Create Production Environment File
The setup script will create `.env.production`. Update it with your RDS details:

```bash
# Production Environment Variables
DB_NAME=ml_interview_notes
DB_USER=your_rds_username
DB_PASSWORD=your_rds_password
DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
DB_PORT=5432
DEBUG=False
SECRET_KEY=your-production-secret-key-here
ALLOWED_HOSTS=your-ec2-public-ip,your-domain.com
```

### 6.2 Test Django Connection
```bash
# Activate virtual environment
source venv/bin/activate

# Test Django database connection
python manage.py check --database default
```

## Step 7: Security Best Practices

### 7.1 Update Security Group
For production, restrict access:
- Remove public access (set "Publicly accessible" to No)
- Only allow connections from your EC2 instance
- Use VPC peering if needed

### 7.2 Enable SSL
- RDS PostgreSQL supports SSL by default
- Update your Django settings to use SSL:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT'),
        'OPTIONS': {
            'sslmode': 'require',
        },
    }
}
```

### 7.3 Regular Backups
- Enable automated backups
- Set up cross-region backup copies
- Test backup restoration regularly

## Step 8: Monitoring and Maintenance

### 8.1 Enable CloudWatch Monitoring
- Enable detailed monitoring
- Set up CloudWatch alarms for:
  - CPU utilization
  - Free storage space
  - Database connections
  - Read/Write latency

### 8.2 Performance Optimization
- Monitor slow queries
- Optimize indexes
- Consider read replicas for read-heavy workloads

## Troubleshooting

### Connection Issues
1. Check security group rules
2. Verify endpoint and credentials
3. Ensure database is publicly accessible (if testing locally)
4. Check VPC and subnet settings

### Performance Issues
1. Monitor CloudWatch metrics
2. Check for slow queries
3. Consider upgrading instance class
4. Optimize database indexes

### Backup Issues
1. Verify backup retention settings
2. Check storage space
3. Ensure automated backups are enabled

## Cost Optimization

### Free Tier
- 750 hours per month of db.t3.micro
- 20 GB of storage
- 20 GB of backup storage

### Production Recommendations
- Use Reserved Instances for predictable workloads
- Enable storage autoscaling
- Monitor unused resources
- Use appropriate instance sizes

## Next Steps

After setting up RDS:
1. Deploy your application to EC2
2. Set up a domain name and SSL certificate
3. Configure monitoring and alerting
4. Set up automated backups and disaster recovery
5. Implement CI/CD pipeline

For EC2 deployment, use the provided `scripts/deploy_to_ec2.sh` script. 