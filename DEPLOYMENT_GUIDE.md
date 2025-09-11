# 🚀 Deployment Guide - Solar Panel Soiling Detection System

This comprehensive deployment guide covers all aspects of deploying the solar panel soiling detection system across different environments, from local development to production Raspberry Pi installations.

## 📋 Pre-Deployment Checklist

### ✅ Essential Requirements
- [ ] Python 3.8+ installed
- [ ] YOLOv8 model file (`.pt` format)
- [ ] WeatherAPI.com account and API key
- [ ] Git repository cloned locally
- [ ] Basic terminal/command line knowledge

### ✅ Hardware Requirements
| Environment | CPU | RAM | Storage | Network |
|-------------|-----|-----|---------|---------|
| **Development** | Modern laptop | 4GB+ | 2GB free | Internet |
| **Production** | Raspberry Pi 4 | 2GB+ | 8GB+ SD card | WiFi/Ethernet |
| **Cloud** | 2+ vCPUs | 4GB+ | 10GB+ SSD | High bandwidth |

## 🛠️ Environment Setup

### Option 1: Automated Setup (Recommended)
```bash
# Clone repository
git clone https://github.com/Akshayc002/Solar_Cleaning.git
cd solar-soiling-detection

# Run automated setup (Linux/macOS)
chmod +x setup.sh
./setup.sh

# Follow prompts for configuration
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv soiling_env

# 2. Activate environment
# Linux/macOS:
source soiling_env/bin/activate
# Windows:
soiling_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create directories
mkdir -p static/uploads templates test_images logs

# 6. Set permissions (Linux/macOS)
chmod +x *.py *.sh
```

## ⚙️ Configuration

### 1. Basic Configuration
Copy and edit the configuration file:
```bash
cp config.json my_config.json
nano my_config.json  # or your preferred editor
```

### 2. Essential Settings
```json
{
    "MODEL_PATH": "./models/your_yolov8_model.pt",
    "WEATHER_API_KEY": "your_actual_api_key_here",
    "LATITUDE": 40.7128,
    "LONGITUDE": -74.0060,
    "SOILING_AREA_THRESHOLD": 5.0
}
```

### 3. Environment-Specific Configurations

#### Development Configuration
```json
{
    "FLASK_DEBUG": true,
    "LOG_LEVEL": "DEBUG",
    "FLASK_HOST": "127.0.0.1",
    "FLASK_PORT": 5000,
    "CAMERA_WIDTH": 640,
    "CAMERA_HEIGHT": 640
}
```

#### Production Configuration
```json
{
    "FLASK_DEBUG": false,
    "LOG_LEVEL": "INFO",
    "FLASK_HOST": "0.0.0.0",
    "FLASK_PORT": 5000,
    "CAMERA_WIDTH": 512,
    "CAMERA_HEIGHT": 512
}
```

#### Raspberry Pi Configuration
```json
{
    "FLASK_DEBUG": false,
    "LOG_LEVEL": "WARNING",
    "FLASK_HOST": "0.0.0.0",
    "FLASK_PORT": 5000,
    "CAMERA_WIDTH": 416,
    "CAMERA_HEIGHT": 416,
    "YOLO_CONFIDENCE": 0.3
}
```

### 4. API Key Setup
Get your free WeatherAPI key:
1. Visit [WeatherAPI.com](https://weatherapi.com)
2. Sign up for free account
3. Copy API key from dashboard
4. Update `WEATHER_API_KEY` in config.json

## 🧪 Testing & Validation

### 1. System Health Check
```bash
# Run comprehensive test suite
python test_system.py

# Quick validation
python run.py --test-mode
```

### 2. Individual Component Testing
```bash
# Test configuration loading
python -c "import enhanced_soiling_detector; print('Config OK')"

# Test model loading (if you have model file)
python -c "from ultralytics import YOLO; YOLO('your_model.pt'); print('Model OK')"

# Test weather API
python -c "import requests; r=requests.get('http://api.weatherapi.com/v1/current.json?key=YOUR_KEY&q=London'); print('Weather API OK' if r.status_code==200 else 'API Error')"
```

### 3. Mock Testing (Without Model)
```bash
# Enable mock mode in test_system.py
python test_system.py --mock-mode

# This will test all components with simulated data
```

## 🖥️ Local Development Deployment

### 1. Standard Development Server
```bash
# Activate environment
source soiling_env/bin/activate

# Start development server
python enhanced_soiling_detector.py

# Access at: http://localhost:5000
```

### 2. Development with Auto-Reload
```bash
# Set debug mode in config.json
"FLASK_DEBUG": true

# Start server
python enhanced_soiling_detector.py

# Server will auto-reload on code changes
```

### 3. Custom Development Setup
```bash
# Use custom config file
CONFIG_FILE=dev_config.json python enhanced_soiling_detector.py

# Use custom port
FLASK_PORT=8080 python enhanced_soiling_detector.py
```

## 🏭 Production Server Deployment

### 1. Basic Production Setup
```bash
# Install production server
pip install gunicorn

# Create WSGI app wrapper
cat > wsgi.py << 'EOF'
from enhanced_soiling_detector import app

if __name__ == "__main__":
    app.run()
EOF

# Start production server
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

### 2. Advanced Production Configuration
```bash
# Create gunicorn config
cat > gunicorn.conf.py << 'EOF'
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
preload_app = True
max_requests = 1000
max_requests_jitter = 100
EOF

# Start with config
gunicorn -c gunicorn.conf.py wsgi:app
```

### 3. Process Management with Supervisor
```bash
# Install supervisor
sudo apt install supervisor

# Create supervisor config
sudo tee /etc/supervisor/conf.d/soiling-detector.conf << 'EOF'
[program:soiling-detector]
command=/path/to/soiling_env/bin/gunicorn -c gunicorn.conf.py wsgi:app
directory=/path/to/solar-soiling-detection
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/soiling-detector.log
EOF

# Start service
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start soiling-detector
```

## 🥧 Raspberry Pi Deployment

### 1. Raspberry Pi OS Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git libopencv-dev

# Optional: Install camera support
sudo apt install -y python3-picamera

# Enable camera (if using Pi camera)
sudo raspi-config  # Enable camera in interface options
```

### 2. Optimize for Raspberry Pi
```bash
# Increase GPU memory split
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Optimize for performance
echo "arm_freq=1500" | sudo tee -a /boot/config.txt
echo "over_voltage=2" | sudo tee -a /boot/config.txt

# Reboot to apply changes
sudo reboot
```

### 3. Install Application
```bash
# Clone repository
cd /home/pi
git clone https://github.com/yourusername/solar-soiling-detection.git
cd solar-soiling-detection

# Setup environment
python3 -m venv soiling_env
source soiling_env/bin/activate
pip install -r requirements.txt

# Configure for Pi
cp config.json pi_config.json
# Edit pi_config.json with Pi-optimized settings
```

### 4. Create System Service
```bash
# Create service file
sudo tee /etc/systemd/system/soiling-detector.service << 'EOF'
[Unit]
Description=Solar Panel Soiling Detector
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/solar-soiling-detection
Environment=PATH=/home/pi/solar-soiling-detection/soiling_env/bin
ExecStart=/home/pi/solar-soiling-detection/soiling_env/bin/python enhanced_soiling_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable soiling-detector
sudo systemctl start soiling-detector

# Check status
sudo systemctl status soiling-detector
```

### 5. Auto-Start Configuration
```bash
# Check service logs
sudo journalctl -u soiling-detector -f

# Restart service
sudo systemctl restart soiling-detector

# Stop service
sudo systemctl stop soiling-detector
```

## 🐳 Docker Deployment

### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads templates logs

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=enhanced_soiling_detector.py
ENV FLASK_ENV=production

# Run application
CMD ["python", "enhanced_soiling_detector.py"]
```

### 2. Build and Run Docker Container
```bash
# Build image
docker build -t solar-soiling-detector .

# Run container
docker run -d \
  --name soiling-detector \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  solar-soiling-detector

# Check logs
docker logs soiling-detector -f
```

### 3. Docker Compose Setup
```yaml
version: '3.8'

services:
  soiling-detector:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config.json:/app/config.json
    restart: unless-stopped
    environment:
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/data"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ☁️ Cloud Deployment

### 1. AWS EC2 Deployment
```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# Connect via SSH

# Install dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv git nginx

# Clone and setup application
git clone https://github.com/yourusername/solar-soiling-detection.git
cd solar-soiling-detection
python3 -m venv soiling_env
source soiling_env/bin/activate
pip install -r requirements.txt gunicorn

# Configure Nginx reverse proxy
sudo tee /etc/nginx/sites-available/soiling-detector << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/soiling-detector /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

### 2. Heroku Deployment
```bash
# Create Procfile
echo "web: gunicorn enhanced_soiling_detector:app" > Procfile

# Create runtime.txt
echo "python-3.9.18" > runtime.txt

# Deploy to Heroku
heroku create your-app-name
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 3. Digital Ocean App Platform
```yaml
# app.yaml
name: solar-soiling-detector
services:
- name: web
  source_dir: /
  github:
    repo: yourusername/solar-soiling-detection
    branch: main
  run_command: gunicorn enhanced_soiling_detector:app
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 5000
  envs:
  - key: FLASK_ENV
    value: production
```

## 🔒 Security & Production Hardening

### 1. Basic Security
```bash
# Create non-root user for service
sudo useradd -m -s /bin/bash soiling
sudo usermod -a -G sudo soiling

# Set file permissions
sudo chown -R soiling:soiling /path/to/solar-soiling-detection
chmod 750 /path/to/solar-soiling-detection
chmod 640 config.json  # Protect config file
```

### 2. Firewall Configuration
```bash
# Ubuntu UFW
sudo ufw allow ssh
sudo ufw allow 5000/tcp
sudo ufw enable

# CentOS/RHEL Firewalld
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

### 3. SSL/HTTPS Setup (with Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring & Maintenance

### 1. Health Checks
```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/data)
if [ $response -eq 200 ]; then
    echo "OK: Service is healthy"
    exit 0
else
    echo "ERROR: Service returned $response"
    exit 1
fi
EOF

chmod +x health_check.sh
```

### 2. Log Management
```bash
# Setup log rotation
sudo tee /etc/logrotate.d/soiling-detector << 'EOF'
/var/log/soiling-detector.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

### 3. Backup Configuration
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/soiling-detector"
mkdir -p $BACKUP_DIR

# Backup database and config
cp soiling_data.db $BACKUP_DIR/soiling_data_$DATE.db
cp config.json $BACKUP_DIR/config_$DATE.json

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
EOF

chmod +x backup.sh

# Add to crontab for daily backup
crontab -e
# Add: 0 2 * * * /path/to/backup.sh
```

## 🐛 Troubleshooting Common Issues

### Model Loading Issues
```bash
# Check model file exists and permissions
ls -la models/your_model.pt
file models/your_model.pt

# Test model loading manually
python -c "from ultralytics import YOLO; m = YOLO('models/your_model.pt'); print('Model loaded successfully')"

# Common fix: Download PyTorch CPU version for compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Port Binding Issues
```bash
# Find process using port 5000
sudo lsof -i :5000
sudo netstat -tulpn | grep :5000

# Kill process if needed
sudo kill -9 <PID>

# Change port in config.json if needed
```

### Memory Issues (Raspberry Pi)
```bash
# Check memory usage
free -m
htop

# Optimize Pi config
# Add to /boot/config.txt:
gpu_mem=64
arm_freq=1400

# Reduce model size or image resolution in config.json
```

### Database Permissions
```bash
# Fix database permissions
chmod 664 soiling_data.db
chown user:group soiling_data.db

# Check database integrity
sqlite3 soiling_data.db "PRAGMA integrity_check;"
```

## 📈 Performance Optimization

### 1. System Optimization
```bash
# For production systems
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 2. Application Optimization
- Use smaller YOLOv8 models (yolov8n vs yolov8x)
- Reduce image processing resolution
- Implement image caching
- Use async processing for weather API calls

### 3. Database Optimization
```sql
-- Create indices for faster queries
CREATE INDEX idx_timestamp ON soiling_data(timestamp);
CREATE INDEX idx_soiling_area ON soiling_data(soiling_area_percent);
```

## 📞 Support & Updates

### Getting Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart soiling-detector
```

### Rollback Procedure
```bash
# View commit history
git log --oneline

# Rollback to previous version
git checkout <previous-commit-hash>

# Restart service
sudo systemctl restart soiling-detector
```

### Community Support
- **Issues**: [GitHub Issues](https://github.com/yourusername/solar-soiling-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/solar-soiling-detection/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/solar-soiling-detection/wiki)

---

## ✅ Deployment Verification

After deployment, verify your system is working correctly:

1. **✅ Web Interface**: Access dashboard at your configured URL
2. **✅ API Endpoints**: Test `/api/data` and `/api/history` endpoints
3. **✅ Image Upload**: Upload test image and verify detection
4. **✅ Weather Integration**: Check weather data is displaying
5. **✅ Database**: Verify data is being stored correctly
6. **✅ Logs**: Check application logs for errors

### Final Checklist
- [ ] Application starts without errors
- [ ] Web dashboard loads correctly
- [ ] Image upload and processing works
- [ ] Weather data displays properly
- [ ] Database operations function correctly
- [ ] System service starts automatically
- [ ] Firewall configured properly
- [ ] SSL certificate installed (production)
- [ ] Monitoring and backups configured
- [ ] Documentation updated with deployment details

**🎉 Congratulations! Your Solar Panel Soiling Detection System is now deployed and ready for operation.**