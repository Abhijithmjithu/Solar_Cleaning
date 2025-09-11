# 📷 Camera Module Integration Guide

## Overview

This document outlines the necessary changes to replace the manual image upload functionality with automatic camera capture using Raspberry Pi Camera Module or USB cameras. The system will automatically capture images at configured intervals and process them for soiling detection.

## 🎯 Key Changes Required

### 1. Hardware Setup
- **Raspberry Pi Camera Module**: CSI camera connection
- **USB Camera**: USB 2.0/3.0 webcam
- **Power Management**: Ensure adequate power supply
- **Mounting**: Physical camera mounting system

### 2. Software Dependencies
- **Additional Libraries**: Camera-specific Python packages
- **System Drivers**: Camera drivers and kernel modules
- **Storage Management**: Automated image cleanup

---

## 🔧 Required Code Changes

### A. Configuration Updates (`config.json`)

Add new camera-specific configuration options:

```json
{
    "CAMERA_TYPE": "pi_camera",
    "CAMERA_ENABLED": true,
    "CAMERA_CAPTURE_INTERVAL": 1800,
    "CAMERA_WARMUP_TIME": 2,
    "CAMERA_RESOLUTION": [1920, 1080],
    "CAMERA_AUTO_EXPOSURE": true,
    "CAMERA_ISO": 0,
    "CAMERA_SHUTTER_SPEED": 0,
    "IMAGE_QUALITY": 85,
    "AUTO_DELETE_IMAGES": true,
    "KEEP_IMAGES_DAYS": 7,
    "CAPTURE_SCHEDULE": {
        "enabled": true,
        "start_hour": 6,
        "end_hour": 18,
        "interval_minutes": 30
    }
}
```

### B. Dependencies Update (`requirements.txt`)

Add camera-specific packages:

```txt
# Existing dependencies
flask
opencv-python
numpy
ultralytics
scikit-image
requests

# Camera dependencies
picamera2>=0.3.12        # For Pi Camera Module v2/v3
opencv-contrib-python    # Enhanced OpenCV with camera support
pillow>=9.0.0           # Image processing
schedule>=1.2.0         # Task scheduling

# System monitoring
psutil>=5.9.0           # System resource monitoring
```

### C. Camera Interface Class

Create `camera_manager.py`:

```python
#!/usr/bin/env python3
"""
Camera Management Module for Solar Panel Monitoring
Supports Raspberry Pi Camera Module and USB cameras
"""

import cv2
import time
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Optional, Tuple
import schedule

# Try to import Pi-specific camera libraries
try:
    from picamera2 import Picamera2, Preview
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    logging.warning("Pi Camera library not available. Using OpenCV for camera access.")

class CameraManager:
    """Manages camera operations for automatic image capture"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.capture_thread = None
        self.is_capturing = False
        self.last_capture_time = None
        
        # Create capture directory
        self.capture_dir = Path("static/captures")
        self.capture_dir.mkdir(exist_ok=True, parents=True)
        
        self.initialize_camera()
        self.setup_scheduler()
    
    def initialize_camera(self) -> bool:
        """Initialize camera based on configuration"""
        camera_type = self.config.get('CAMERA_TYPE', 'usb_camera')
        
        try:
            if camera_type == 'pi_camera' and PI_CAMERA_AVAILABLE:
                return self._init_pi_camera()
            else:
                return self._init_usb_camera()
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            return False
    
    def _init_pi_camera(self) -> bool:
        """Initialize Raspberry Pi camera module"""
        try:
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_still_configuration(
                main={
                    "size": tuple(self.config.get('CAMERA_RESOLUTION', [1920, 1080])),
                    "format": "RGB888"
                }
            )
            
            self.camera.configure(config)
            self.camera.start()
            
            # Warmup time
            time.sleep(self.config.get('CAMERA_WARMUP_TIME', 2))
            
            logging.info("Pi Camera initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Pi camera: {e}")
            return False
    
    def _init_usb_camera(self) -> bool:
        """Initialize USB camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            
            # Set camera properties
            width, height = self.config.get('CAMERA_RESOLUTION', [1920, 1080])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Failed to capture test frame")
            
            logging.info("USB Camera initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize USB camera: {e}")
            return False
    
    def capture_image(self) -> Optional[str]:
        """Capture single image and return file path"""
        if not self.camera:
            logging.error("Camera not initialized")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = self.capture_dir / filename
            
            if hasattr(self.camera, 'capture_file'):  # Pi Camera
                self.camera.capture_file(str(filepath))
            else:  # USB Camera
                ret, frame = self.camera.read()
                if ret:
                    cv2.imwrite(str(filepath), frame)
                else:
                    logging.error("Failed to read frame from USB camera")
                    return None
            
            self.last_capture_time = datetime.now()
            logging.info(f"Image captured: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logging.error(f"Failed to capture image: {e}")
            return None
    
    def setup_scheduler(self):
        """Setup automatic capture scheduling"""
        if not self.config.get('CAMERA_ENABLED', False):
            return
        
        schedule_config = self.config.get('CAPTURE_SCHEDULE', {})
        
        if schedule_config.get('enabled', False):
            # Schedule captures during daylight hours
            start_hour = schedule_config.get('start_hour', 6)
            end_hour = schedule_config.get('end_hour', 18)
            interval = schedule_config.get('interval_minutes', 30)
            
            # Schedule captures every interval during daylight hours
            for hour in range(start_hour, end_hour):
                for minute in range(0, 60, interval):
                    schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self.scheduled_capture)
        else:
            # Simple interval-based capture
            interval = self.config.get('CAMERA_CAPTURE_INTERVAL', 1800)  # 30 minutes
            schedule.every(interval).seconds.do(self.scheduled_capture)
    
    def scheduled_capture(self):
        """Perform scheduled capture and processing"""
        try:
            # Check if it's daylight hours (optional sun angle check)
            current_hour = datetime.now().hour
            schedule_config = self.config.get('CAPTURE_SCHEDULE', {})
            
            if schedule_config.get('enabled', False):
                start_hour = schedule_config.get('start_hour', 6)
                end_hour = schedule_config.get('end_hour', 18)
                
                if not (start_hour <= current_hour < end_hour):
                    logging.info("Outside capture hours, skipping")
                    return
            
            # Capture image
            image_path = self.capture_image()
            if image_path:
                # Process image (this will be called by the main detector)
                from enhanced_soiling_detector import detector
                if detector:
                    metrics, overlay_path = detector.process_image_file(image_path)
                    logging.info(f"Processed capture: {metrics['soiling_area_percent']:.1f}% soiling")
            
        except Exception as e:
            logging.error(f"Scheduled capture failed: {e}")
    
    def start_capture_service(self):
        """Start the capture service in a separate thread"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_service_loop, daemon=True)
        self.capture_thread.start()
        logging.info("Camera capture service started")
    
    def _capture_service_loop(self):
        """Main service loop for scheduled captures"""
        while self.is_capturing:
            try:
                schedule.run_pending()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logging.error(f"Capture service error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop_capture_service(self):
        """Stop the capture service"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
        logging.info("Camera capture service stopped")
    
    def cleanup_old_images(self):
        """Remove old images based on retention policy"""
        if not self.config.get('AUTO_DELETE_IMAGES', False):
            return
        
        try:
            keep_days = self.config.get('KEEP_IMAGES_DAYS', 7)
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            deleted_count = 0
            for image_file in self.capture_dir.glob("capture_*.jpg"):
                if image_file.stat().st_mtime < cutoff_date.timestamp():
                    image_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} old images")
                
        except Exception as e:
            logging.error(f"Image cleanup failed: {e}")
    
    def get_camera_status(self) -> dict:
        """Get current camera status"""
        return {
            'initialized': self.camera is not None,
            'capturing': self.is_capturing,
            'last_capture': self.last_capture_time.isoformat() if self.last_capture_time else None,
            'camera_type': self.config.get('CAMERA_TYPE', 'unknown'),
            'total_captures': len(list(self.capture_dir.glob("capture_*.jpg")))
        }
    
    def __del__(self):
        """Cleanup camera resources"""
        self.stop_capture_service()
        if self.camera:
            if hasattr(self.camera, 'stop'):
                self.camera.stop()
            elif hasattr(self.camera, 'release'):
                self.camera.release()
```

### D. Main Application Updates (`enhanced_soiling_detector.py`)

Add camera integration to the main detector:

```python
# Add imports at the top
from camera_manager import CameraManager
import atexit

# Add to Config class
class Config:
    # ... existing config ...
    CAMERA_ENABLED = config_data.get("CAMERA_ENABLED", False)
    CAMERA_TYPE = config_data.get("CAMERA_TYPE", "usb_camera")
    CAMERA_CAPTURE_INTERVAL = config_data.get("CAMERA_CAPTURE_INTERVAL", 1800)

# Initialize camera manager after detector
camera_manager = None
if Config.CAMERA_ENABLED:
    try:
        camera_manager = CameraManager(config_data)
        camera_manager.start_capture_service()
        logging.info("Camera service started successfully")
        
        # Cleanup function for graceful shutdown
        def cleanup_camera():
            if camera_manager:
                camera_manager.stop_capture_service()
        
        atexit.register(cleanup_camera)
        
    except Exception as e:
        logging.error(f"Failed to initialize camera manager: {e}")
        camera_manager = None

# Add new routes for camera management
@app.route('/api/camera/status')
def camera_status():
    """Get camera status"""
    if camera_manager:
        return jsonify(camera_manager.get_camera_status())
    else:
        return jsonify({'error': 'Camera not available'}), 404

@app.route('/api/camera/capture', methods=['POST'])
def manual_capture():
    """Trigger manual capture"""
    if not camera_manager:
        return jsonify({'error': 'Camera not available'}), 404
    
    try:
        image_path = camera_manager.capture_image()
        if image_path:
            # Process the captured image
            metrics, overlay_path = detector.process_image_file(image_path)
            return jsonify({
                'success': True,
                'image_path': image_path,
                'metrics': metrics,
                'overlay_path': overlay_path
            })
        else:
            return jsonify({'error': 'Failed to capture image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/cleanup', methods=['POST'])
def cleanup_images():
    """Clean up old images"""
    if not camera_manager:
        return jsonify({'error': 'Camera not available'}), 404
    
    try:
        camera_manager.cleanup_old_images()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### E. Database Schema Updates

Add camera-specific fields to track automated captures:

```sql
-- Add new columns to soiling_data table
ALTER TABLE soiling_data ADD COLUMN capture_type TEXT DEFAULT 'upload';
ALTER TABLE soiling_data ADD COLUMN camera_settings TEXT;
ALTER TABLE soiling_data ADD COLUMN capture_scheduled BOOLEAN DEFAULT 0;

-- Create camera status table
CREATE TABLE IF NOT EXISTS camera_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_capture_time TEXT,
    total_captures INTEGER DEFAULT 0,
    failed_captures INTEGER DEFAULT 0,
    camera_errors TEXT,
    last_maintenance TEXT
);

INSERT OR IGNORE INTO camera_status (id, total_captures, failed_captures) VALUES (1, 0, 0);
```

### F. Web Interface Updates (`dashboard.html`)

Replace upload section with camera controls:

```html
<!-- Replace the upload section with camera section -->
<div class="camera-section">
    <h3>📷 Camera Control</h3>
    
    <div class="camera-status" id="cameraStatus">
        <div class="status-item">
            <span class="label">Camera Status:</span>
            <span class="value" id="cameraStatusText">Loading...</span>
        </div>
        <div class="status-item">
            <span class="label">Last Capture:</span>
            <span class="value" id="lastCaptureTime">--</span>
        </div>
        <div class="status-item">
            <span class="label">Total Captures:</span>
            <span class="value" id="totalCaptures">--</span>
        </div>
    </div>
    
    <div class="camera-controls">
        <button id="manualCaptureBtn" class="btn btn-primary">
            📸 Capture Now
        </button>
        <button id="cleanupImagesBtn" class="btn btn-secondary">
            🗑️ Cleanup Old Images
        </button>
    </div>
    
    <div class="capture-schedule">
        <h4>Automatic Capture Schedule</h4>
        <p id="scheduleInfo">Next capture in: <span id="nextCaptureTime">--</span></p>
    </div>
</div>

<!-- Add JavaScript for camera controls -->
<script>
// Camera management functions
async function updateCameraStatus() {
    try {
        const response = await fetch('/api/camera/status');
        if (response.ok) {
            const status = await response.json();
            document.getElementById('cameraStatusText').textContent = 
                status.initialized ? 'Active' : 'Offline';
            document.getElementById('lastCaptureTime').textContent = 
                status.last_capture ? new Date(status.last_capture).toLocaleString() : 'Never';
            document.getElementById('totalCaptures').textContent = status.total_captures || 0;
        } else {
            document.getElementById('cameraStatusText').textContent = 'Not Available';
        }
    } catch (error) {
        console.error('Failed to fetch camera status:', error);
        document.getElementById('cameraStatusText').textContent = 'Error';
    }
}

async function manualCapture() {
    const btn = document.getElementById('manualCaptureBtn');
    btn.disabled = true;
    btn.textContent = '📸 Capturing...';
    
    try {
        const response = await fetch('/api/camera/capture', {method: 'POST'});
        const result = await response.json();
        
        if (result.success) {
            alert('Image captured successfully!');
            updateData(); // Refresh dashboard data
            updateCameraStatus();
        } else {
            alert('Capture failed: ' + result.error);
        }
    } catch (error) {
        alert('Capture error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '📸 Capture Now';
    }
}

async function cleanupImages() {
    if (!confirm('Remove old images according to retention policy?')) return;
    
    try {
        const response = await fetch('/api/camera/cleanup', {method: 'POST'});
        const result = await response.json();
        
        if (result.success) {
            alert('Image cleanup completed!');
            updateCameraStatus();
        } else {
            alert('Cleanup failed: ' + result.error);
        }
    } catch (error) {
        alert('Cleanup error: ' + error.message);
    }
}

// Event listeners
document.getElementById('manualCaptureBtn').addEventListener('click', manualCapture);
document.getElementById('cleanupImagesBtn').addEventListener('click', cleanupImages);

// Update camera status periodically
setInterval(updateCameraStatus, 30000); // Every 30 seconds
updateCameraStatus(); // Initial call
</script>
```

---

## 🔧 Installation & Setup

### A. Raspberry Pi Camera Module Setup

```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Install camera dependencies
sudo apt update
sudo apt install -y python3-pip python3-picamera2

# Test camera
libcamera-hello --display 0 --width 1920 --height 1080 --timeout 5000
```

### B. USB Camera Setup

```bash
# Install USB camera dependencies
sudo apt install -y v4l-utils

# List available cameras
v4l2-ctl --list-devices

# Test USB camera
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### C. System Service Configuration

Update systemd service file:

```ini
# /etc/systemd/system/soiling-detector.service
[Unit]
Description=Solar Panel Soiling Detector with Camera
After=network.target
Requires=network.target

[Service]
Type=simple
User=pi
Group=video
WorkingDirectory=/home/pi/solar-soiling-detection
Environment=PATH=/home/pi/solar-soiling-detection/soiling_env/bin
ExecStart=/home/pi/solar-soiling-detection/soiling_env/bin/python enhanced_soiling_detector.py
Restart=always
RestartSec=10

# Camera-specific environment
Environment=CAMERA_ENABLED=true
Environment=CAMERA_TYPE=pi_camera

[Install]
WantedBy=multi-user.target
```

---

## 📊 Testing Camera Integration

### A. Camera Test Script

Create `test_camera.py`:

```python
#!/usr/bin/env python3
"""Test camera functionality"""

import json
from camera_manager import CameraManager

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Enable camera for testing
config['CAMERA_ENABLED'] = True

# Test camera manager
camera_mgr = CameraManager(config)

print("Camera Status:", camera_mgr.get_camera_status())

# Test capture
print("Testing image capture...")
image_path = camera_mgr.capture_image()

if image_path:
    print(f"✅ Image captured successfully: {image_path}")
else:
    print("❌ Image capture failed")

# Cleanup
del camera_mgr
```

### B. Test Commands

```bash
# Test camera functionality
python test_camera.py

# Test with different camera types
CAMERA_TYPE=pi_camera python test_camera.py
CAMERA_TYPE=usb_camera python test_camera.py

# Monitor camera service logs
sudo journalctl -u soiling-detector -f | grep -i camera
```

---

## 🔍 Troubleshooting Camera Issues

### A. Common Problems

| Problem | Cause | Solution |
|---------|-------|----------|
| Camera not detected | Driver/hardware issue | Check connections, enable camera interface |
| Permission denied | User not in video group | `sudo usermod -a -G video pi` |
| Poor image quality | Wrong settings | Adjust resolution, ISO, exposure |
| Memory errors | Insufficient RAM | Reduce image resolution, increase swap |
| Capture failures | Resource conflicts | Ensure only one app uses camera |

### B. Debug Commands

```bash
# Check camera hardware
vcgencmd get_camera

# List video devices
ls -la /dev/video*

# Check camera permissions
groups $USER

# Monitor system resources
htop
free -m
df -h

# Test camera directly
# Pi Camera:
libcamera-still -o test.jpg --width 1920 --height 1080

# USB Camera:
fswebcam -r 1920x1080 test.jpg
```

### C. Performance Optimization

```json
{
    "CAMERA_RESOLUTION": [1280, 720],
    "IMAGE_QUALITY": 75,
    "CAMERA_WARMUP_TIME": 1,
    "CAPTURE_SCHEDULE": {
        "interval_minutes": 60
    }
}
```

---

## 🚀 Deployment Checklist

### Pre-deployment
- [ ] Camera hardware connected and tested
- [ ] Camera interface enabled in raspi-config
- [ ] Required Python packages installed
- [ ] Camera permissions configured
- [ ] Configuration updated with camera settings

### Code Changes
- [ ] `camera_manager.py` created and tested
- [ ] Main application updated with camera integration
- [ ] Database schema updated
- [ ] Web interface updated with camera controls
- [ ] Requirements.txt updated

### Testing
- [ ] Manual capture test successful
- [ ] Scheduled capture working
- [ ] Image processing pipeline functional
- [ ] Web interface camera controls working
- [ ] Auto-cleanup functioning

### Production
- [ ] Systemd service updated and restarted
- [ ] Log monitoring configured
- [ ] Storage space monitoring setup
- [ ] Backup procedures updated
- [ ] Documentation updated

---

## 📈 Benefits of Camera Integration

### Automation Benefits
- **Continuous Monitoring**: 24/7 automated surveillance
- **Consistent Timing**: Regular captures at optimal intervals
- **Reduced Manual Work**: No need for manual uploads
- **Better Data**: Consistent lighting and positioning

### Technical Benefits
- **Real-time Processing**: Immediate analysis after capture
- **Historical Trends**: Automated data collection
- **Weather Integration**: Smart scheduling based on conditions
- **Resource Efficiency**: Optimized capture and storage

### Operational Benefits
- **Early Detection**: Faster identification of soiling events
- **Maintenance Planning**: Predictive maintenance scheduling
- **Cost Reduction**: Reduced manual monitoring costs
- **Scalability**: Easy to extend to multiple locations

---

## 🔧 Future Enhancements

### Advanced Camera Features
- **Multiple Camera Support**: Monitor multiple panel arrays
- **PTZ Camera Control**: Pan/tilt/zoom for better coverage
- **Night Vision**: Infrared cameras for 24/7 monitoring
- **Weather Protection**: Weatherproof camera enclosures

### Smart Scheduling
- **Sun Position Calculation**: Optimal capture timing
- **Weather-based Scheduling**: Skip captures during rain
- **Adaptive Intervals**: Dynamic timing based on conditions
- **Remote Control**: Manual capture triggering via API

### Analytics Integration
- **Real-time Alerts**: Immediate notifications for issues
- **Trend Analysis**: Long-term pattern recognition
- **Performance Correlation**: Link soiling to power output
- **Maintenance Optimization**: Predictive cleaning schedules

This camera integration transforms the system from a manual testing tool to a fully automated monitoring solution suitable for production solar installations.