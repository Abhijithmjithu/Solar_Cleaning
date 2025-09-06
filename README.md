# Solar Panel Soiling Detection & Monitoring System

An AI-powered system for detecting and monitoring soiling on solar panels using YOLOv8 computer vision, weather integration, and a web-based dashboard.

## Features

- 🎯 **YOLOv8 Soiling Detection**: Uses advanced computer vision to identify soiling areas
- 🌤️ **Weather Integration**: Checks weather conditions to optimize cleaning alerts
- 📊 **Real-time Dashboard**: Web interface showing soiling metrics and trends
- 📱 **Image Upload**: Test system with your own solar panel images
- 🗄️ **Data Storage**: SQLite database for historical tracking
- ⚙️ **Configurable**: All parameters adjustable via config file
- 🔍 **Visual Overlays**: Highlights detected soiling areas on images

## System Requirements

- Python 3.8 or higher
- OpenCV-compatible system (Windows, macOS, Linux)
- At least 2GB RAM (4GB recommended for YOLOv8)
- Storage: ~500MB for dependencies, additional for model and data

## Quick Start

### 1. Setup Environment

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
python -m venv soiling_env
soiling_env\Scripts\activate
pip install -r requirements.txt
mkdir static\uploads templates test_images
```

### 2. Configure System

Edit `config.json` with your specific values:

```json
{
    "MODEL_PATH": "path/to/your/yolov8_model.pt",
    "WEATHER_API_KEY": "your_weatherapi_key_here",
    "LATITUDE": 40.7128,
    "LONGITUDE": -74.0060,
    "SOILING_AREA_THRESHOLD": 5.0
}
```

**Required Configuration:**
- `MODEL_PATH`: Path to your trained YOLOv8 segmentation model
- `WEATHER_API_KEY`: Get free API key from [WeatherAPI.com](https://weatherapi.com)
- `LATITUDE/LONGITUDE`: Your solar installation coordinates

### 3. Test System

Run comprehensive tests:
```bash
python test_system.py
```

This will:
- Create mock data and images for testing
- Validate all system components  
- Generate test reports
- Verify database operations

### 4. Run Application

Start the web server:
```bash
python enhanced_soiling_detector.py
```

Access dashboard at: http://localhost:5000

## Configuration Options

All system parameters are configurable in `config.json`:

### Core Settings
- `MODEL_PATH`: YOLOv8 model file path
- `WEATHER_API_KEY`: WeatherAPI.com API key  
- `DB_PATH`: SQLite database file location
- `LATITUDE/LONGITUDE`: Geographic coordinates

### Detection Parameters
- `SOILING_AREA_THRESHOLD`: Minimum soiling % to trigger alerts (default: 5.0)
- `YOLO_CONFIDENCE`: Detection confidence threshold (default: 0.5)
- `WEATHER_CHECK_HOURS`: Hours ahead to check rain forecast (default: 1)
- `RAIN_PROBABILITY_THRESHOLD`: Rain % threshold for alert suppression (default: 50)

### Image Processing
- `CAMERA_WIDTH/HEIGHT`: Image processing resolution (default: 640x640)
- `CLAHE_CLIP_LIMIT`: Contrast enhancement strength (default: 2.0)
- `OVERLAY_ALPHA`: Mask overlay transparency (default: 0.4)

### Web Server
- `FLASK_HOST`: Server host address (default: "0.0.0.0")
- `FLASK_PORT`: Server port number (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: false)

## Usage Guide

### Web Dashboard

1. **Upload Images**: Use the upload form to test with your solar panel images
2. **View Results**: Dashboard displays:
   - Current soiling area percentage
   - Soiling intensity score
   - Structural similarity (SSIM) to reference
   - Weather conditions and forecast
   - Historical trends
3. **Download Reports**: Export data for analysis

### API Endpoints

- `GET /api/data`: Latest soiling detection results
- `GET /api/history`: Historical data (last 50 records)
- `POST /upload`: Upload image for processing

Example API usage:
```python
import requests

# Get latest data
response = requests.get('http://localhost:5000/api/data')
data = response.json()
print(f"Current soiling: {data['soiling_area_percent']:.1f}%")
```

### Command Line Usage

Process single image:
```python
from enhanced_soiling_detector import SoilingDetector

detector = SoilingDetector()
metrics, overlay_path = detector.process_image_file('panel_image.jpg')
print(f"Soiling detected: {metrics['soiling_area_percent']:.1f}%")
```

## Data Flow

```
Image Input → Preprocessing → YOLOv8 Detection → Metric Calculation
     ↓
Weather Check → Alert Decision → Database Storage → Web Dashboard
     ↓
Overlay Generation → File Storage → API Response
```

## Testing Strategy

The system includes comprehensive testing:

### Automated Tests
- **Unit Tests**: Individual component validation
- **Integration Tests**: Database and file operations  
- **Performance Tests**: Processing speed benchmarks
- **Mock Testing**: Simulated model and weather data

### Manual Testing
1. Upload test images through web interface
2. Verify detection accuracy on known samples
3. Test weather API integration
4. Validate alert thresholds

## Deployment

### Raspberry Pi Deployment

1. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv libopencv-dev
   ```

2. **Transfer Files:** Copy entire project to Pi

3. **Modify Config:** Update for Pi-specific settings:
   ```json
   {
       "FLASK_HOST": "0.0.0.0",
       "CAMERA_WIDTH": 416,
       "CAMERA_HEIGHT": 416
   }
   ```

4. **Auto-start Service:** Create systemd service for automatic startup

### Production Considerations

- Use HTTPS for secure access
- Implement user authentication
- Set up log rotation
- Configure backup for database
- Monitor disk space for images
- Use reverse proxy (nginx) for better performance

## Troubleshooting

### Common Issues

**Model Loading Failed:**
- Verify model file exists and path is correct
- Check model compatibility with ultralytics version
- Ensure sufficient RAM available

**Weather API Errors:**
- Confirm API key is valid and active
- Check network connectivity
- Verify coordinate format (decimal degrees)

**Detection Not Working:**
- Test with sample images first
- Adjust YOLO confidence threshold
- Verify image format compatibility

**Web Interface Issues:**
- Check Flask port availability (5000)
- Verify templates folder exists
- Review browser console for JavaScript errors

### Debug Mode

Enable detailed logging:
```json
{
    "LOG_LEVEL": "DEBUG",
    "FLASK_DEBUG": true
}
```

View logs:
```bash
python enhanced_soiling_detector.py 2>&1 | tee system.log
```

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check troubleshooting section
- Review test output for diagnostics
- Create GitHub issue with logs and configuration

## Version History

- **v1.0**: Initial release with YOLOv8 detection
- **v1.1**: Added weather integration and configurable parameters
- **v1.2**: Enhanced testing suite and Raspberry Pi support