#!/usr/bin/env python3
"""
Solar Panel Soiling Detection & Monitoring System
Features: Self-Healing Serial, Physical Disconnect Detection, YOLOv8 AI
"""

import os
import csv
import io
import cv2
import numpy as np
import time
import sqlite3
import json
import logging
import requests
import serial
import serial.tools.list_ports # Required for port scanning
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for, make_response
from skimage.metrics import structural_similarity as ssim

# Fix for PyTorch loading issue
try:
    import torch.serialization
    from ultralytics.nn.tasks import SegmentationModel
    torch.serialization.add_safe_globals([SegmentationModel])
except ImportError:
    pass

from ultralytics import YOLO

# --- CONFIGURATION ---
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
    CAPTURE_INTERVAL = 7200
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    RAIN_PROBABILITY_THRESHOLD = config_data.get("RAIN_PROBABILITY_THRESHOLD", 75)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# --- ROBUST ARDUINO CONNECTION ---
ARDUINO_PORT = 'COM4' # <--- VERIFY THIS PORT
BAUD_RATE = 9600
arduino = None

def get_arduino_connection():
    global arduino
    if arduino is not None and arduino.is_open:
        return arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) 
        logging.info(f"SUCCESS: Reconnected to Robot on {ARDUINO_PORT}")
        return arduino
    except Exception as e:
        return None

def send_arduino_command(command_char):
    device = get_arduino_connection()
    if device:
        try:
            cmd = command_char.upper()
            device.write(cmd.encode())
            logging.info(f"ROBOT SIGNAL SENT: {cmd}")
            return True
        except Exception as e:
            logging.error(f"Lost connection during send: {e}")
            global arduino
            arduino = None
    else:
        logging.warning(f"Cannot send '{command_char}' - Robot Offline")
    return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

class SoilingDetector:
    def __init__(self):
        try:
            self.model = YOLO(Config.MODEL_PATH)
            logging.info(f"Model loaded successfully from {Config.MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
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
                alert_type TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                suppression_count INTEGER DEFAULT 0,
                cleaning_stage INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('INSERT OR IGNORE INTO system_state (id, suppression_count, cleaning_stage) VALUES (1, 0, 0)')
        conn.commit()
        conn.close()

    def get_state(self):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT suppression_count, cleaning_stage FROM system_state WHERE id = 1')
        row = cursor.fetchone()
        conn.close()
        return row if row else (0, 0)

    def update_state(self, suppression_count, cleaning_stage):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('UPDATE system_state SET suppression_count = ?, cleaning_stage = ? WHERE id = 1', 
                      (suppression_count, cleaning_stage))
        conn.commit()
        conn.close()

    def preprocess_image(self, image):
        image = cv2.resize(image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
        return image

    def detect_soiling(self, image):
        results = self.model(image, conf=0.1)
        masks = []
        if results[0].masks is not None:
            for mask in results[0].masks.data:
                masks.append(mask.cpu().numpy())
        return masks

    def save_mask_overlay(self, original_image, mask, save_path):
        overlay = cv2.resize(original_image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
        mask_bool = mask.astype(bool)
        result = overlay.copy()
        red_pixels = np.zeros_like(result[mask_bool], dtype=np.uint8)
        red_pixels[:, 2] = 255
        if result[mask_bool].size > 0 and result[mask_bool].shape == red_pixels.shape:
            result[mask_bool] = cv2.addWeighted(result[mask_bool], 0.6, red_pixels, 0.4, 0)
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
        
        intensity_score = 0.0
        if soiled_pixels > 0 and self.reference_image is not None:
            reference_resized = cv2.resize(self.reference_image, (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_reference = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
            soiled_regions = gray_image[combined_mask]
            reference_regions = gray_reference[combined_mask]
            if len(reference_regions) > 0:
                mean_soiled = np.mean(soiled_regions)
                mean_ref = np.mean(reference_regions)
                intensity_score = 1 - (mean_soiled / max(mean_ref, 1))

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
            for hour_data in hours[:1]:
                rain_prob = int(hour_data.get('chance_of_rain', 0))
                rain_probability = max(rain_probability, rain_prob)

            rain_expected = rain_probability > Config.RAIN_PROBABILITY_THRESHOLD
            return rain_expected, data
        except Exception as e:
            logging.error(f"WeatherAPI error: {e}")
            return False, {}

    def save_data_new(self, metrics, image_path, weather_data, alert_type):
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO soiling_data (timestamp, soiling_area_percent, soiling_intensity, total_pixels,
                                      soiled_pixels, ssim_score, image_path, weather_data, alert_type)
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
            alert_type
        ))
        conn.commit()
        conn.close()

    def process_image_file(self, filepath):
        image = cv2.imread(filepath)
        if image is None:
            logging.error(f"Failed to read image from {filepath}")
            return None, None, "None"
        
        processed_image = self.preprocess_image(image)
        masks = self.detect_soiling(processed_image)
        metrics = self.calculate_soiling_metrics(processed_image, masks)
        rain_expected, weather_data = self.check_weather()

        threshold = float(Config.SOILING_AREA_THRESHOLD)
        suppression_count, cleaning_stage = self.get_state()
        
        alert_type = "None"
        new_stage = cleaning_stage
        
        debug_msg = f"[LOGIC] Soiling: {metrics['soiling_area_percent']:.1f}% | Stage: {cleaning_stage} | Rain: {rain_expected} | "

        is_soiled = metrics['soiling_area_percent'] >= threshold

        if cleaning_stage == 0:
            if is_soiled:
                if rain_expected:
                    suppression_count += 1
                    if suppression_count >= 2:
                        alert_type = "Dry"
                        send_arduino_command('D') 
                        new_stage = 1  
                        suppression_count = 0 
                        debug_msg += "Rain Override -> Dry Clean Initiated"
                    else:
                        alert_type = "None"
                        new_stage = 0
                        debug_msg += f"Rain Detected -> Suppressing (Count {suppression_count})"
                else:
                    alert_type = "Dry"
                    send_arduino_command('D') 
                    new_stage = 1 
                    suppression_count = 0
                    debug_msg += "Dirty -> Dry Clean Initiated"
            else:
                suppression_count = 0
                new_stage = 0
                debug_msg += "System Clean"

        elif cleaning_stage == 1:
            if is_soiled:
                alert_type = "Wet"
                send_arduino_command('W') 
                new_stage = 2 
                suppression_count = 0
                debug_msg += "Dry Failed -> Escalating to Wet Clean"
            else:
                alert_type = "None"
                new_stage = 0
                suppression_count = 0
                debug_msg += "Dry Clean Successful -> System Clean"

        elif cleaning_stage == 2:
            if is_soiled:
                alert_type = "Manual"
                new_stage = 2 
                debug_msg += "Wet Failed -> MANUAL INSPECTION ALERT"
            else:
                alert_type = "None"
                new_stage = 0 
                debug_msg += "Manual/Wet Clean Successful -> System Clean"

        self.update_state(suppression_count, new_stage)
        
        height, width = processed_image.shape[:2]
        combined_mask = np.zeros((height, width), dtype=bool)
        for mask in masks:
            mask_resized = cv2.resize(mask, (width, height))
            combined_mask = np.logical_or(combined_mask, mask_resized > 0.5)
            
        base_filename = os.path.basename(filepath)
        masked_filename = base_filename.rsplit('.', 1)[0] + "_masked.jpg"
        masked_filepath = os.path.join(app.config['UPLOAD_FOLDER'], masked_filename)
        self.save_mask_overlay(image, combined_mask, masked_filepath)
        
        relative_image_path = os.path.relpath(masked_filepath, start='static')
        self.save_data_new(metrics, relative_image_path, weather_data, alert_type)
        
        logging.info(f"{debug_msg}")
        return metrics, relative_image_path, alert_type
    
detector = SoilingDetector()

try:
    if os.path.exists('./static/reference_panel.jpg'):
        detector.reference_image = cv2.imread('./static/reference_panel.jpg')
        logging.info("Reference image loaded successfully")
    else:
        logging.warning("No reference image found")
except Exception as e:
    logging.warning(f"Failed to load reference image: {e}")

@app.route('/')
def dashboard():
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT soiling_area_percent, alert_type FROM soiling_data ORDER BY timestamp DESC LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    
    soiling_area_percent = row[0] if row else 0
    db_alert_type = row[1] if row else "None"
    
    status = "Clean"
    if soiling_area_percent >= Config.SOILING_AREA_THRESHOLD:
        status = "Needs Attention"
    if db_alert_type != "None":
        status = f"Action Required: {db_alert_type}"
        
    capture_interval_ms = Config.CAPTURE_INTERVAL * 1000
    trigger_action = request.args.get('trigger', 'None')
    
    return render_template('dashboard.html', 
                           system_status=status, 
                           capture_interval_ms=capture_interval_ms,
                           threshold=Config.SOILING_AREA_THRESHOLD,
                           trigger_action=trigger_action)

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
            
            metrics, masked_image_path, alert_type = detector.process_image_file(filepath)
            return redirect(url_for('dashboard', trigger=alert_type))
            
    return "Upload Failed", 400

@app.route('/api/data')
def get_data():
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, soiling_area_percent, soiling_intensity, ssim_score,
                   image_path, alert_type, weather_data
            FROM soiling_data
            ORDER BY timestamp DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()

        if row:
            weather_str = row[6]
            if isinstance(weather_str, bytes):
                weather_str = weather_str.decode('utf-8')
            weather_json = json.loads(weather_str) if weather_str else {}

            return jsonify({
                'timestamp': row[0],
                'soiling_area_percent': row[1],
                'soiling_intensity': row[2],
                'ssim_score': row[3],
                'image_path': row[4],
                'alert_type': row[5],
                'alert_sent': row[5] != "None",
                'weather_data': weather_json
            })
    except Exception as e:
        logging.error(f"API data fetch failed: {e}")
    return jsonify({'error': 'No data available'})

@app.route('/api/history')
def get_history():
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
        history = [{'timestamp': r[0], 'soiling_area_percent': r[1], 'soiling_intensity': r[2], 'ssim_score': r[3]} for r in rows]
        return jsonify(history)
    except Exception as e:
        logging.error(f"API history fetch failed: {e}")
        return jsonify([])

@app.route('/api/export_history')
def export_history():
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, soiling_area_percent, soiling_intensity, ssim_score, image_path, alert_type FROM soiling_data ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Timestamp', 'Soiling Area (%)', 'Intensity', 'SSIM', 'Image Path', 'Alert Type'])
    cw.writerows(rows)
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=solar_cleaning_log.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/api/manual_dry', methods=['POST'])
def manual_dry():
    send_arduino_command('D')
    return jsonify({"status": "sent", "type": "dry"})

@app.route('/api/manual_wet', methods=['POST'])
def manual_wet():
    send_arduino_command('W')
    return jsonify({"status": "sent", "type": "wet"})

@app.route('/api/manual_return', methods=['POST'])
def manual_return():
    send_arduino_command('R')
    return jsonify({"status": "sent", "type": "return"})

@app.route('/api/manual_stop', methods=['POST'])
def manual_stop():
    send_arduino_command('S')
    return jsonify({"status": "sent", "type": "stop"})

# --- UPDATED: PHYSICAL DISCONNECT DETECTION ---
@app.route('/api/hardware_status')
def hardware_status():
    global arduino
    
    # 1. Check if the Port still exists in the OS
    ports = [p.device for p in serial.tools.list_ports.comports()]
    
    if ARDUINO_PORT not in ports:
        if arduino:
            try:
                arduino.close()
            except: pass
        arduino = None
        return jsonify({'connected': False, 'port': ARDUINO_PORT})

    # 2. If Port exists, check/init connection
    device = get_arduino_connection()
    is_connected = (device is not None and device.is_open)
    
    return jsonify({'connected': is_connected, 'port': ARDUINO_PORT})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError: pass
    app.run(host='0.0.0.0', port=5000, debug=True)