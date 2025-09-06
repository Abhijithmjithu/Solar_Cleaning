#!/usr/bin/env python3
"""
Quick Start Script for Solar Panel Soiling Detection System
Performs basic checks and starts the application
"""

import os
import sys
import json
import subprocess

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} - Compatible")

def check_config_file():
    """Check if config file exists and is valid"""
    if not os.path.exists('config.json'):
        print("❌ config.json not found")
        print("Creating template config file...")
        
        template_config = {
            "MODEL_PATH": "models/yolov8_soiling.pt",
            "WEATHER_API_KEY": "your_weatherapi_key_here",
            "DB_PATH": "soiling_data.db",
            "LATITUDE": 40.7128,
            "LONGITUDE": -74.0060,
            "SOILING_AREA_THRESHOLD": 5.0,
            "CAMERA_WIDTH": 640,
            "CAMERA_HEIGHT": 640,
            "CAPTURE_INTERVAL": 7200,
            "YOLO_CONFIDENCE": 0.5,
            "CLAHE_CLIP_LIMIT": 2.0,
            "CLAHE_TILE_GRID_SIZE": [8, 8],
            "OVERLAY_ALPHA": 0.4,
            "RAIN_PROBABILITY_THRESHOLD": 50,
            "WEATHER_CHECK_HOURS": 1,
            "UPLOAD_FOLDER": "static/uploads",
            "ALLOWED_EXTENSIONS": ["png", "jpg", "jpeg"],
            "FLASK_HOST": "0.0.0.0",
            "FLASK_PORT": 5000,
            "FLASK_DEBUG": False,
            "LOG_LEVEL": "INFO"
        }
        
        with open('config.json', 'w') as f:
            json.dump(template_config, f, indent=4)
        
        print("✅ Template config.json created")
        print("📝 Please edit config.json with your actual values:")
        print("   - MODEL_PATH: Path to your YOLOv8 model")
        print("   - WEATHER_API_KEY: Your WeatherAPI.com key")
        print("   - LATITUDE/LONGITUDE: Your location coordinates")
        return False
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Check critical configuration
        if config.get('MODEL_PATH') == 'models/yolov8_soiling.pt':
            print("⚠️  Default MODEL_PATH detected - please update with actual model path")
        
        if config.get('WEATHER_API_KEY') == 'your_weatherapi_key_here':
            print("⚠️  Default WEATHER_API_KEY detected - weather features will be limited")
        
        print("✅ config.json loaded successfully")
        return True
        
    except json.JSONDecodeError:
        print("❌ config.json contains invalid JSON")
        return False

def check_directories():
    """Create necessary directories"""
    dirs = ['static/uploads', 'templates']
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'opencv-python', 'numpy', 'ultralytics', 
        'scikit-image', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def run_tests():
    """Ask user if they want to run tests"""
    response = input("\n🧪 Run system tests? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\nRunning system tests...")
        try:
            subprocess.run([sys.executable, 'test_system.py'], check=True)
            print("✅ All tests completed")
        except subprocess.CalledProcessError:
            print("⚠️  Some tests failed - check output above")
        except FileNotFoundError:
            print("❌ test_system.py not found")

def start_application():
    """Start the Flask application"""
    print("\n" + "="*50)
    print("🚀 Starting Solar Panel Soiling Detection System")
    print("="*50)
    
    try:
        # Import and start the application
        subprocess.run([sys.executable, 'enhanced_soiling_detector.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except FileNotFoundError:
        print("❌ enhanced_soiling_detector.py not found")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main startup sequence"""
    print("🌞 Solar Panel Soiling Detection System - Startup")
    print("=" * 50)
    
    # Run startup checks
    check_python_version()
    
    if not check_config_file():
        print("\n⏸️  Please update config.json and run this script again")
        sys.exit(1)
    
    check_directories()
    
    if not check_dependencies():
        print("\n⏸️  Please install dependencies and run this script again")
        sys.exit(1)
    
    run_tests()
    start_application()

if __name__ == "__main__":
    main()