#!/usr/bin/env python3
"""
Solar Panel Soiling Detection & Monitoring System
- Uses YOLOv8 segmentation to calculate soiling area
- Loads config from external JSON
- Uses WeatherAPI.com for rain forecast check
- Shows alerts on web dashboard (no MQTT/Telegram)
- Supports image file upload for testing
"""

import os
import cv2
import numpy as np
import torch.serialization
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
import time
import sqlite3
from datetime import datetime
import json
import logging
import requests
from flask import Flask, render_template, jsonify, request, redirect, url_for
from skimage.metrics import structural_similarity as ssim

# Add safe globals for PyTorch loading
torch.serialization.add_safe_globals([SegmentationModel])

# Load config from JSON file
try:
    with open('config.json') as cfile:
        config_data = json.load(cfile)
except FileNotFoundError:
    print("ERROR: config.json file not found. Please create one based on config_template.json")
    exit(1)
except json.JSONDecodeError:
    print("ERROR: Invalid JSON in config.json file")
    exit(1)

class Config:
    MODEL_PATH = config_data.get("MODEL_PATH")
    WEATHER_API_KEY = config_data.get("WEATHER_API_KEY")
    DB_PATH = config_data.get("DB_PATH", "soiling_data.db")
    LATITUDE = config_data.get("LATITUDE", 40.7128)
    LONGITUDE = config_data.get("LONGITUDE", -74.0060)
    SOILING_AREA_THRESHOLD = config_data.get("SOILING_AREA_THRESHOLD", 5.0)
    IMAGE_WIDTH = config_data.get("IMAGE_WIDTH", 512)
    IMAGE_HEIGHT = config_data.get("IMAGE_HEIGHT", 512)
    CAPTURE_INTERVAL = config_data.get("CAPTURE_INTERVAL", 7200)
    YOLO_CONFIDENCE = config_data.get("YOLO_CONFIDENCE", 0.5)
    OVERLAY_ALPHA = config_data.get("OVERLAY_ALPHA", 0.4)
    RAIN_PROBABILITY_THRESHOLD = config_data.get("RAIN_PROBABILITY_THRESHOLD", 50)
    WEATHER_CHECK_HOURS = config_data.get("WEATHER_CHECK_HOURS", 1)
    UPLOAD_FOLDER = config_data.get("UPLOAD_FOLDER", "static/uploads")
    ALLOWED_EXTENSIONS = set(config_data.get("ALLOWED_EXTENSIONS", ["png", "jpg", "jpeg"]))
    FLASK_HOST = config_data.get("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = config_data.get("FLASK_PORT", 5000)
    FLASK_DEBUG = config_data.get("FLASK_DEBUG", False)
    LOG_LEVEL = config_data.get("LOG_LEVEL", "INFO")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

class SoilingDetector:
    def __init__(self):
        try:
            self.model = YOLO(Config.MODEL_PATH)
            logging.info(f"Model loaded successfully from {Config.MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model from {Config.MODEL_PATH}: {e}")
            raise
            
        self.reference_image = None
        self.setup_database()

    def setup_database(self):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS soiling_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    soiling_area_percent REAL,
                    soiling_intensity REAL,
                    total_pixels INTEGER,
                    soiled_pixels INTEGER,
                    ssim_score REAL,
                    image_path TEXT,
                    weather_data TEXT,
                    alert_sent BOOLEAN
                )
            ''')
            conn.commit()
            conn.close()
            logging.info("Database setup completed")
        except Exception as e:
            logging.error(f"Database setup failed: {e}")
            raise

    def preprocess_image(self, image):
        """Simple preprocessing: only resize to configured dimensions"""
        try:
            image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
            return image
        except Exception as e:
            logging.error(f"Image preprocessing failed: {e}")
            raise

    def detect_soiling(self, image):
        """Run YOLOv8 detection with configurable confidence threshold"""
        try:
            results = self.model(image, conf=Config.YOLO_CONFIDENCE)
            masks = []
            if results[0].masks is not None:
                for mask in results[0].masks.data:
                    masks.append(mask.cpu().numpy())
            logging.info(f"detect_soiling: {len(masks)} masks detected")
            return masks
        except Exception as e:
            logging.error(f"Soiling detection failed: {e}")
            return []

    def save_mask_overlay(self, original_image, mask, save_path):
        """Save overlay image with configurable transparency"""
        try:
            # Resize original image to match processing dimensions for better overlay alignment
            overlay = cv2.resize(original_image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
            red = np.zeros_like(overlay, dtype=np.uint8)
            red[:, :, 2] = 255  # Red channel
            
            mask_uint8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Apply red overlay with configurable alpha
            overlay = cv2.addWeighted(overlay, 1, red, Config.OVERLAY_ALPHA, 0)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
            
            cv2.imwrite(save_path, overlay)
            logging.info(f"Overlay image saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save mask overlay: {e}")

    def calculate_soiling_metrics(self, image, masks):
        """Calculate comprehensive soiling metrics"""
        try:
            height, width = image.shape[:2]
            total_pixels = height * width
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            for mask in masks:
                mask_resized = cv2.resize(mask, (width, height))
                combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
                
            soiled_pixels = np.sum(combined_mask)
            soiling_area_percent = (soiled_pixels / total_pixels) * 100
            
            logging.info(f"calculate_soiling_metrics: soiling_area_percent={soiling_area_percent:.2f}%")
            
            # Calculate intensity score if reference available
            if soiled_pixels > 0 and self.reference_image is not None:
                # Resize reference image to match processing dimensions
                reference_resized = cv2.resize(self.reference_image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_reference = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
                soiled_regions = gray_image[combined_mask]
                reference_regions = gray_reference[combined_mask]
                
                if len(soiled_regions) > 0 and len(reference_regions) > 0:
                    mean_soiled_intensity = np.mean(soiled_regions)
                    mean_reference_intensity = np.mean(reference_regions)
                    intensity_score = 1 - (mean_soiled_intensity / max(mean_reference_intensity, 1))
                else:
                    intensity_score = 0.0
            else:
                intensity_score = 0.0
            
            # Calculate SSIM score
            ssim_score = 0.0
            if self.reference_image is not None:
                reference_resized = cv2.resize(self.reference_image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
                gray_current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_ref = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
                ssim_score = ssim(gray_current, gray_ref)
            
            return {
                'soiling_area_percent': soiling_area_percent,
                'soiling_intensity': intensity_score,
                'total_pixels': total_pixels,
                'soiled_pixels': int(soiled_pixels),
                'ssim_score': ssim_score,
                'soiling_mask': combined_mask
            }
        except Exception as e:
            logging.error(f"Metric calculation failed: {e}")
            return {
                'soiling_area_percent': 0.0,
                'soiling_intensity': 0.0,
                'total_pixels': 0,
                'soiled_pixels': 0,
                'ssim_score': 0.0,
                'soiling_mask': np.zeros((height, width), dtype=np.uint8) if 'height' in locals() and 'width' in locals() else np.zeros((512, 512), dtype=np.uint8)
            }

    def check_weather(self):
        """Check weather with configurable time window"""
        try:
            if not Config.WEATHER_API_KEY or Config.WEATHER_API_KEY == "your_weatherapi_key_here":
                logging.warning("No valid weather API key configured")
                return False, {}
                
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                'key': Config.WEATHER_API_KEY,
                'q': f"{Config.LATITUDE},{Config.LONGITUDE}",
                'days': 1,
                'aqi': 'no',
                'alerts': 'no'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            rain_probability = 0
            hours = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])
            
            # Check only the configured number of hours (now 1 hour instead of 6)
            for hour_data in hours[:Config.WEATHER_CHECK_HOURS]:
                rain_prob_hour = int(hour_data.get('chance_of_rain', 0))
                snow_prob_hour = int(hour_data.get('chance_of_snow', 0))
                rain_probability = max(rain_probability, rain_prob_hour, snow_prob_hour)
            
            rain_expected = rain_probability > Config.RAIN_PROBABILITY_THRESHOLD
            logging.info(f"Weather check: rain probability={rain_probability}%, expected={rain_expected}")
            
            return rain_expected, data
        except Exception as e:
            logging.error(f"WeatherAPI.com error: {e}")
            return False, {}

    def save_data(self, metrics, image_path, weather_data, alert_sent):
        """Save processing results to database"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO soiling_data 
                (timestamp, soiling_area_percent, soiling_intensity, total_pixels, 
                 soiled_pixels, ssim_score, image_path, weather_data, alert_sent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metrics['soiling_area_percent'],
                metrics['soiling_intensity'],
                metrics['total_pixels'],
                metrics['soiled_pixels'],
                metrics['ssim_score'],
                image_path,
                json.dumps(weather_data),
                alert_sent
            ))
            conn.commit()
            conn.close()
            logging.info("Data saved to database successfully")
        except Exception as e:
            logging.error(f"Database save failed: {e}")

    def process_image_file(self, filepath):
        """Main image processing pipeline"""
        try:
            image = cv2.imread(filepath)
            if image is None:
                logging.error(f"Failed to read image from {filepath}")
                return None, None
                
            processed_image = self.preprocess_image(image)
            masks = self.detect_soiling(processed_image)
            metrics = self.calculate_soiling_metrics(processed_image, masks)
            
            rain_expected, weather_data = self.check_weather()
            alert_needed = (
                metrics['soiling_area_percent'] > Config.SOILING_AREA_THRESHOLD 
                and not rain_expected
            )
            
            # Generate combined mask for overlay
            height, width = processed_image.shape[:2]
            combined_mask = np.zeros((height, width), dtype=bool)
            for mask in masks:
                mask_resized = cv2.resize(mask, (width, height))
                combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
            
            # Save overlay image
            base_filename = os.path.basename(filepath)
            masked_filename = base_filename.rsplit('.', 1)[0] + "_masked.jpg"
            masked_filepath = os.path.join(app.config['UPLOAD_FOLDER'], masked_filename)
            
            self.save_mask_overlay(image, combined_mask, masked_filepath)
            
            # Save data with relative path
            relative_image_path = os.path.relpath(masked_filepath, start='static')
            self.save_data(metrics, relative_image_path, weather_data, alert_needed)
            
            logging.info(
                f"Processing complete - Soiling: {metrics['soiling_area_percent']:.1f}%, "
                f"Alert: {alert_needed}"
            )
            
            return metrics, relative_image_path
            
        except Exception as e:
            logging.error(f"Image processing failed: {e}")
            return None, None

# Initialize detector
detector = SoilingDetector()

# Try to load reference image
try:
    if os.path.exists('reference_clean_panel.jpg'):
        detector.reference_image = cv2.imread('reference_clean_panel.jpg')
        logging.info("Reference image loaded successfully")
    else:
        logging.warning("No reference image found (reference_clean_panel.jpg)")
except Exception as e:
    logging.warning(f"Failed to load reference image: {e}")

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        try:
            if 'image' not in request.files:
                return "No file part", 400
                
            file = request.files['image']
            if file.filename == '':
                return "No selected file", 400
                
            if file and allowed_file(file.filename):
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                    
                filename = datetime.now().strftime("upload_%Y%m%d_%H%M%S.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                metrics, masked_image_path = detector.process_image_file(filepath)
                
                if metrics is not None:
                    return redirect(url_for('dashboard'))
                else:
                    return "Processing failed", 500
            else:
                return "Invalid file type", 400
        except Exception as e:
            logging.error(f"Upload failed: {e}")
            return f"Upload failed: {str(e)}", 500
    
    return '''
    <!doctype html>
    <html>
    <head><title>Upload Image</title></head>
    <body>
    <h2>Upload Solar Panel Image</h2>
    <form method=post enctype=multipart/form-data>
      <input type=file name=image accept="image/*">
      <input type=submit value=Upload>
    </form>
    <a href="/">Back to Dashboard</a>
    </body>
    </html>
    '''

@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM soiling_data 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            weather_json = json.loads(row[8]) if row[8] else {}
            return jsonify({
                'timestamp': row[1],
                'soiling_area_percent': row[2],
                'soiling_intensity': row[3],
                'ssim_score': row[6],
                'image_path': row[7],
                'alert_sent': row[9],
                'weather_data': weather_json
            })
    except Exception as e:
        logging.error(f"API data fetch failed: {e}")
        
    return jsonify({'error': 'No data available'})

@app.route('/api/history')
def get_history():
    """API endpoint for historical data"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, soiling_area_percent, soiling_intensity, ssim_score
            FROM soiling_data 
            ORDER BY timestamp DESC LIMIT 50
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history = [
            {
                'timestamp': row[0],
                'soiling_area_percent': row[1],
                'soiling_intensity': row[2], 
                'ssim_score': row[3]
            }
            for row in rows
        ]
        
        return jsonify(history)
    except Exception as e:
        logging.error(f"API history fetch failed: {e}")
        return jsonify([])

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        logging.warning("flask-cors not installed. CORS not enabled.")
    
    logging.info(f"Starting Flask app on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        host=Config.FLASK_HOST, 
        port=Config.FLASK_PORT, 
        debug=Config.FLASK_DEBUG
    )