#!/bin/bash

# Solar Panel Soiling Detection System Setup Script
# Run this script to set up the environment for testing

echo "======================================================"
echo "Solar Panel Soiling Detection System Setup"
echo "======================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv soiling_env

# Activate virtual environment
echo "Activating virtual environment..."
source soiling_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p static/uploads
mkdir -p templates
mkdir -p test_images

# Copy config template
if [ ! -f "config.json" ]; then
    echo "Creating config.json from template..."
    cp config.json config_template.json
    echo "Please edit config.json with your actual values:"
    echo "  - MODEL_PATH: Path to your YOLOv8 model file"
    echo "  - WEATHER_API_KEY: Your WeatherAPI.com API key"
    echo "  - LATITUDE/LONGITUDE: Your location coordinates"
else
    echo "✓ config.json already exists"
fi

# Set permissions
chmod +x enhanced_soiling_detector.py
chmod +x test_system.py

echo ""
echo "======================================================"
echo "Setup completed successfully!"
echo "======================================================"
echo ""
echo "Next steps:"
echo "1. Edit config.json with your actual values"
echo "2. Place your YOLOv8 model file in the project directory"
echo "3. Run tests: python3 test_system.py"
echo "4. Start the application: python3 enhanced_soiling_detector.py"
echo ""
echo "To activate the virtual environment later:"
echo "  source soiling_env/bin/activate"
echo ""
echo "Web interface will be available at: http://localhost:5000"