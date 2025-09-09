#!/usr/bin/env python3
"""
Solar Panel Soiling Detection & Monitoring System - Quick Fix Version
- Uses YOLOv8 segmentation to calculate soiling area
- Loads config from external JSON
- Uses WeatherAPI.com for rain forecast check
- Shows alerts on web dashboard (no MQTT/Telegram)
- Supports image file upload for testing
"""

import os
import cv2
import numpy as np

# Fix for PyTorch loading issue
try:
    import torch.serialization
    from ultralytics.nn.tasks import SegmentationModel
    torch.serialization.add_safe_globals([SegmentationModel])
except ImportError:
    # If ultralytics.nn.tasks doesn't exist, continue without the fix
    pass

from ultralytics import YOLO
import time
import sqlite3
from datetime import datetime
import json
import logging
import requests
from flask import Flask, render_template, jsonify, request, redirect, url_for
from skimage.metrics import structural_similarity as ssim

# Load config from JSON file
try:
    with open('config.json') as cfile:
        config_data = json.load(cfile)
except FileNotFoundError:
    print("ERROR: config.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("ERROR: Invalid JSON in config.json file")
    exit(1)

class Config:
    MODEL_PATH = config_data.get("MODEL_PATH")
    WEATHER_API_KEY = config_data.get("WEATHER_API_KEY")
    DB_PATH = config_data.get("DB_PATH", "soiling_data.db")
    LATITUDE = config_data.get("LATITUDE", 10.5241)
    LONGITUDE = config_data.get("LONGITUDE", 76.2121)
    SOILING_AREA_THRESHOLD = config_data.get("SOILING_AREA_THRESHOLD")
    CAMERA_WIDTH = config_data.get("CAMERA_WIDTH", 512)
    CAMERA_HEIGHT = config_data.get("CAMERA_HEIGHT", 512)
    WEATHER_CHECK_HOURS = config_data.get("WEATHER_CHECK_HOURS", 1)
    CAPTURE_INTERVAL = 7200  # seconds (2 hours)
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    RAIN_PROBABILITY_THRESHOLD = config_data.get("RAIN_PROBABILITY_THRESHOLD", 75)

print(f"[DEBUG] SOILING_AREA_THRESHOLD loaded: {Config.SOILING_AREA_THRESHOLD}")
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
        # Table to track consecutive rain suppressions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_suppression (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                consecutive_count INTEGER
            )
        ''')
        # Ensure single row exists
        cursor.execute('INSERT OR IGNORE INTO alert_suppression (id, consecutive_count) VALUES (1, 0)')
        conn.commit()
        conn.close()
    def get_suppression_count(self):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT consecutive_count FROM alert_suppression WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 0

    def set_suppression_count(self, count):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE alert_suppression SET consecutive_count = ? WHERE id = 1', (count,))
        conn.commit()
        conn.close()

    def preprocess_image(self, image):
        # Simple preprocessing: only resize to configured dimensions
        image = cv2.resize(image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
        return image

    def detect_soiling(self, image):
        results = self.model(image, conf=0.1)
        masks = []
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                masks.append(mask.cpu().numpy())
        logging.info(f"detect_soiling: {len(masks)} masks detected")
        return masks

    def save_mask_overlay(self, original_image, mask, save_path):
        """Save overlay image with reddish color only inside mask"""
        overlay = cv2.resize(original_image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
        mask_bool = mask.astype(bool)
        # Create a copy to modify
        result = overlay.copy()
        # Create a red array matching the shape of masked pixels
        red_pixels = np.zeros_like(result[mask_bool], dtype=np.uint8)
        red_pixels[:, 2] = 255
        # Blend only masked pixels with red if any masked pixels exist
        if result[mask_bool].size > 0 and result[mask_bool].shape == red_pixels.shape:
            result[mask_bool] = cv2.addWeighted(result[mask_bool], 0.6, red_pixels, 0.4, 0)
        # Optionally, draw contours for mask boundary
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(save_path, result)

    def calculate_soiling_metrics(self, image, masks):
        height, width = image.shape[:2]
        total_pixels = height * width
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for mask in masks:
            mask_resized = cv2.resize(mask, (width, height))
            combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
            
        soiled_pixels = np.sum(combined_mask)
        soiling_area_percent = (soiled_pixels / total_pixels) * 100
        
        logging.info(f"calculate_soiling_metrics: soiling_area_percent={soiling_area_percent}")
        
        if soiled_pixels > 0 and self.reference_image is not None:
            # Resize reference image to match processing dimensions
            reference_resized = cv2.resize(self.reference_image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_reference = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
            soiled_regions = gray_image[combined_mask]
            reference_regions = gray_reference[combined_mask]
            mean_soiled_intensity = np.mean(soiled_regions)
            mean_reference_intensity = np.mean(reference_regions)
            intensity_score = 1 - (mean_soiled_intensity / max(mean_reference_intensity, 1))
        else:
            intensity_score = 0.0
            
        ssim_score = 0.0
        if self.reference_image is not None:
            reference_resized = cv2.resize(self.reference_image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
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

    def check_weather(self):
        try:
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                'key': Config.WEATHER_API_KEY,
                'q': f"{Config.LATITUDE},{Config.LONGITUDE}",
                'days': 1,
                'aqi': 'no',
                'alerts': 'no'
            }
            response = requests.get(url, params=params)
            data = response.json()
            rain_probability = 0
            hours = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])
            
            # Check only next 1 hour as requested
            for hour_data in hours[:1]:
                rain_prob_hour = int(hour_data.get('chance_of_rain', 0))
                snow_prob_hour = int(hour_data.get('chance_of_snow', 0))
                rain_probability = max(rain_probability, rain_prob_hour, snow_prob_hour)

            rain_expected = rain_probability > Config.RAIN_PROBABILITY_THRESHOLD
            return rain_expected, data
        except Exception as e:
            logging.error(f"WeatherAPI.com error: {e}")
            return False, {}

    def save_data(self, metrics, image_path, weather_data, alert_sent):
        print(f"[DEBUG] Saving to DB: alert_sent={alert_sent}")
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO soiling_data (timestamp, soiling_area_percent, soiling_intensity, total_pixels,
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
            int(alert_sent)  # Store as integer 0/1
        ))
        conn.commit()
        conn.close()

    def process_image_file(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            logging.error(f"Failed to read image from {filepath}")
            return None, None
        processed_image = self.preprocess_image(image)
        masks = self.detect_soiling(processed_image)
        metrics = self.calculate_soiling_metrics(processed_image, masks)
        rain_expected, weather_data = self.check_weather()

        threshold = float(Config.SOILING_AREA_THRESHOLD)
        suppression_count = self.get_suppression_count()
        alert_needed = False
        debug_msg = f"[DEBUG] soiling_area_percent={metrics['soiling_area_percent']} threshold={threshold} rain_expected={rain_expected} suppression_count={suppression_count} "

        if metrics['soiling_area_percent'] >= threshold:
            if rain_expected:
                suppression_count += 1
                # Trigger alert if suppression count reaches 2 or more
                if suppression_count >= 2:
                    alert_needed = True
                    self.set_suppression_count(0)
                    debug_msg += "Alert triggered after multiple suppressions despite rain."
                else:
                    self.set_suppression_count(suppression_count)
                    debug_msg += f"Suppressed due to rain. New suppression_count={suppression_count}"
            else:
                if suppression_count >= 2:
                    alert_needed = True
                    debug_msg += "Alert triggered after multiple suppressions."
                else:
                    alert_needed = True
                    debug_msg += "Alert triggered (no rain, below suppression threshold)."
                self.set_suppression_count(0)
        else:
            self.set_suppression_count(0)
            debug_msg += "Below threshold. Suppression count reset."

        print(debug_msg)

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
        
        logging.info(f"Processed and saved masked image: {relative_image_path} - Soiling: {metrics['soiling_area_percent']:.1f}%, Alert sent: {alert_needed}")
        
        return metrics, relative_image_path
    
def get_system_status(soiling_area_percent, alert_sent):
    if soiling_area_percent < 3:
        return "Clean"
    elif soiling_area_percent < float(Config.SOILING_AREA_THRESHOLD):
        return "Moderate"
    else:
        if alert_sent:
            return "Needs cleaning - Alert triggered"
        else:
            return "Needs cleaning"


# Initialize detector
detector = SoilingDetector()

# Try to load reference image
try:
    if os.path.exists('./static/reference_panel.jpg'):
        detector.reference_image = cv2.imread('./static/reference_panel.jpg')
        logging.info("Reference image loaded successfully")
    else:
        logging.warning("No reference image found (/static/reference_panel.jpg)")
except Exception as e:
    logging.warning(f"Failed to load reference image: {e}")

@app.route('/')
def dashboard():
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT soiling_area_percent, alert_sent FROM soiling_data ORDER BY timestamp DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    if row:
        soiling_area_percent, alert_sent = row
        status = get_system_status(soiling_area_percent, alert_sent)
    else:
        status = "No data available"
    return render_template('dashboard.html', system_status=status)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
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
            return redirect(url_for('dashboard'))
    
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
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, soiling_area_percent, soiling_intensity, ssim_score,
                   image_path, alert_sent, weather_data
            FROM soiling_data
            ORDER BY timestamp DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()

        if row:
            # If weather_data is bytes, decode to string first then load JSON
            weather_raw = row[6]
            if isinstance(weather_raw, bytes):
                weather_str = weather_raw.decode('utf-8')
            else:
                weather_str = weather_raw

            weather_json = json.loads(weather_str) if weather_str else {}

            return jsonify({
                'timestamp': row[0],
                'soiling_area_percent': row[1],
                'soiling_intensity': row[2],
                'ssim_score': row[3],
                'image_path': row[4],
                'alert_sent': bool(int(row[5])),
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
    
@app.route('/api/latest')
def api_latest_redirect():
    return redirect(url_for('get_data'))

# Utility route to clear old alerts below threshold
@app.route('/api/clear_old_alerts', methods=['POST'])
def clear_old_alerts():
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM soiling_data WHERE soiling_area_percent < ?', (Config.SOILING_AREA_THRESHOLD,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'deleted_records': deleted})
    except Exception as e:
        logging.error(f"Failed to clear old alerts: {e}")
        return jsonify({'status': 'error', 'message': str(e)})
            
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        logging.warning("flask-cors not installed. CORS not enabled.")
    
    logging.info("Starting Flask app on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)