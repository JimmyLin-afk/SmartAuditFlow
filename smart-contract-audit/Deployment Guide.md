# Deployment Guide

This guide covers deploying the Smart Contract Audit Tool to various environments.

## Local Development

### Quick Start
```bash
cd smart-contract-audit
python3 -m venv venv
`source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
cp .env.template .env
pip install -r requirements.txt
# Edit .env with your API keys
python src/main.py
```

Access at: http://localhost:5000

## Production Deployment

### Option 1: Traditional Server

#### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.11+
- Nginx (recommended)
- MySQL or PostgreSQL (optional)

#### Steps

1. **Server Setup**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv nginx mysql-server
```

2. **Application Setup**:
```bash
# Clone/upload your application
cd /opt/smart-contract-audit
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Environment Configuration**:
```bash
cp .env.template .env
# Edit .env with production settings
nano .env
```

Production `.env` example:
```
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key
DB_USERNAME=audit_user
DB_PASSWORD=secure_password
DB_HOST=localhost
DB_NAME=smart_audit_db
```

4. **Database Setup** (if using MySQL):
```sql
CREATE DATABASE smart_audit_db;
CREATE USER 'audit_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON smart_audit_db.* TO 'audit_user'@'localhost';
FLUSH PRIVILEGES;
```

5. **Systemd Service**:
Create `/etc/systemd/system/smart-audit.service`:
```ini
[Unit]
Description=Smart Contract Audit Tool
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/smart-contract-audit
Environment=PATH=/opt/smart-contract-audit/venv/bin
ExecStart=/opt/smart-contract-audit/venv/bin/python src/main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

6. **Nginx Configuration**:
Create `/etc/nginx/sites-available/smart-audit`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

7. **Enable and Start**:
```bash
sudo systemctl enable smart-audit
sudo systemctl start smart-audit
sudo ln -s /etc/nginx/sites-available/smart-audit /etc/nginx/sites-enabled/
sudo systemctl reload nginx
```

### Option 2: Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY .env .

EXPOSE 5000

CMD ["python", "src/main.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    env_file:
      - .env
    depends_on:
      - db

  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: smart_audit_db
      MYSQL_USER: audit_user
      MYSQL_PASSWORD: secure_password
    volumes:
      - mysql_data:/var/lib/mysql

volumes:
  mysql_data:
```

#### Deploy with Docker
```bash
docker-compose up -d
```

### Option 3: Cloud Deployment

#### AWS EC2
1. Launch EC2 instance (t3.medium recommended)
2. Follow traditional server setup
3. Configure security groups for port 80/443
4. Use RDS for database (optional)
5. Use Application Load Balancer for scaling

#### Google Cloud Platform
1. Create Compute Engine instance
2. Follow traditional server setup
3. Use Cloud SQL for database
4. Configure firewall rules

#### DigitalOcean
1. Create droplet
2. Follow traditional server setup
3. Use managed database (optional)

## Environment Variables

### Required
- `GEMINI_API_KEY` or `OPENAI_API_KEY`: At least one AI model API key

### Optional
- `CLAUDE_API_KEY`: Additional AI model
- `SECRET_KEY`: Flask secret (auto-generated if not set)
- `DB_*`: Database configuration (uses SQLite if not set)

## Security Considerations

### Production Checklist
- [ ] Use HTTPS (SSL certificate)
- [ ] Set strong SECRET_KEY
- [ ] Use environment variables for sensitive data
- [ ] Configure firewall rules
- [ ] Regular security updates
- [ ] Monitor API usage and costs
- [ ] Implement rate limiting
- [ ] Use secure database credentials
- [ ] Regular backups

### API Key Security
- Store API keys in environment variables
- Rotate keys regularly
- Monitor usage and set quotas
- Use separate keys for different environments

## Monitoring and Maintenance

### Health Checks
- Monitor `/health` endpoint
- Check application logs
- Monitor database connections
- Track API usage

### Logs
```bash
# View application logs
sudo journalctl -u smart-audit -f

# View nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Backup
```bash
# Database backup (MySQL)
mysqldump -u audit_user -p smart_audit_db > backup.sql

# SQLite backup
cp audit.db backup_$(date +%Y%m%d).db
```

## Scaling

### Horizontal Scaling
- Use load balancer (nginx, HAProxy, AWS ALB)
- Deploy multiple application instances
- Use shared database
- Consider Redis for session storage

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Use caching (Redis, Memcached)
- Monitor resource usage

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change port in main.py
2. **Database connection**: Check credentials and network
3. **API limits**: Monitor usage and implement caching
4. **Memory issues**: Increase server resources or optimize code

### Performance Optimization
- Use production WSGI server (Gunicorn, uWSGI)
- Enable gzip compression
- Use CDN for static files
- Implement caching strategies
- Database query optimization

## Cost Optimization

### AI API Costs
- Monitor token usage
- Implement request caching
- Use cheaper models for simple tasks
- Set usage quotas and alerts

### Infrastructure Costs
- Right-size instances
- Use spot instances (AWS)
- Implement auto-scaling
- Regular cost reviews

