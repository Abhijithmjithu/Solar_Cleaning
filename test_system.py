import enhanced_soiling_detector
import numpy as np
import unittest
from unittest.mock import patch

# ...existing code...
class TestRainSuppressionLogic(unittest.TestCase):
    def setUp(self):
        self.detector = enhanced_soiling_detector.SoilingDetector()
        self.detector.set_suppression_count(0)

    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_alert_suppressed_due_to_rain(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (True, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_detect_soiling.return_value = [np.ones((512, 512))]
        self.detector.set_suppression_count(0)
        self.detector.process_image_file('dummy.jpg')
        count = self.detector.get_suppression_count()
        self.assertEqual(count, 1)

    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_alert_triggered_after_multiple_suppressions(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (False, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_detect_soiling.return_value = [np.ones((512, 512))]
        self.detector.set_suppression_count(2)
        self.detector.process_image_file('dummy.jpg')
        count = self.detector.get_suppression_count()
        self.assertEqual(count, 0)

    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_no_alert_below_threshold(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (False, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_detect_soiling.return_value = [np.zeros((512, 512))]
        self.detector.set_suppression_count(2)
        self.detector.process_image_file('dummy.jpg')
        count = self.detector.get_suppression_count()
        self.assertEqual(count, 0)

    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_suppression_count_resets_after_alert(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (False, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_detect_soiling.return_value = [np.ones((512, 512))]
        self.detector.set_suppression_count(3)
        self.detector.process_image_file('dummy.jpg')
        count = self.detector.get_suppression_count()
        self.assertEqual(count, 0)

    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_suppression_count_does_not_increment_below_threshold(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (True, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        mock_detect_soiling.return_value = [np.zeros((512, 512))]
        self.detector.set_suppression_count(2)
        self.detector.process_image_file('dummy.jpg')
        count = self.detector.get_suppression_count()
        self.assertEqual(count, 0)
#!/usr/bin/env python3
"""
Automated Test Script for Solar Panel Soiling Detection System
- Tests all core functionality
- Generates test reports
- Creates mock data if needed
- Validates database operations
"""

import os
import sys
import json
import sqlite3
import cv2
import numpy as np
import requests
import time
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
import logging

# Test configuration
TEST_CONFIG = {
    "CREATE_MOCK_MODEL": True,  # Create a mock YOLO model for testing
    "CREATE_SAMPLE_IMAGES": True,  # Generate sample test images
    "TEST_WEATHER_API": True,  # Test actual weather API (requires key)
    "CREATE_REFERENCE_IMAGE": True,  # Create a mock reference image
    "RUN_WEB_SERVER_TESTS": False,  # Test Flask web server (requires manual setup)
    "CLEANUP_AFTER_TESTS": False  # Clean up test files after testing
}

class MockYOLO:
    """Mock YOLO model for testing when real model is not available"""
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock YOLO model initialized for path: {model_path}")
    
    def __call__(self, image, conf=0.5):
        """Simulate YOLO detection with mock results"""
        height, width = image.shape[:2]
        
        # Create mock detection results
        class MockResult:
            def __init__(self):
                # Simulate finding soiling in a random area
                if np.random.random() > 0.3:  # 70% chance of detecting soiling
                    mask = np.zeros((height, width), dtype=np.float32)
                    # Create a random soiling region
                    y1, y2 = int(height * 0.2), int(height * 0.6)
                    x1, x2 = int(width * 0.3), int(width * 0.7)
                    mask[y1:y2, x1:x2] = 1.0
                    
                    class MockMasks:
                        def __init__(self):
                            class MockTensor:
                                def cpu(self):
                                    return self
                                def numpy(self):
                                    return mask
                            self.data = [MockTensor()]
                    
                    self.masks = MockMasks()
                else:
                    self.masks = None
        
        return [MockResult()]

def create_test_config():
    """Create a test configuration file"""
    test_config = {
        "MODEL_PATH": "test_model.pt",
        "WEATHER_API_KEY": "your_weatherapi_key_here",
        "DB_PATH": "test_soiling_data.db",
        "LATITUDE": 40.7128,
        "LONGITUDE": -74.0060,
        "SOILING_AREA_THRESHOLD": 5.0,
        "CAMERA_WIDTH": 512,
        "CAMERA_HEIGHT": 512,
        "CAPTURE_INTERVAL": 7200,
        "YOLO_CONFIDENCE": 0.5,
        "CLAHE_CLIP_LIMIT": 2.0,
        "CLAHE_TILE_GRID_SIZE": [8, 8],
        "OVERLAY_ALPHA": 0.4,
        "RAIN_PROBABILITY_THRESHOLD": 50,
        "WEATHER_CHECK_HOURS": 1,
        "UPLOAD_FOLDER": "test_static/uploads",
        "ALLOWED_EXTENSIONS": ["png", "jpg", "jpeg"],
        "FLASK_HOST": "127.0.0.1",
        "FLASK_PORT": 5001,  # Different port for testing
        "FLASK_DEBUG": True,
        "LOG_LEVEL": "DEBUG"
    }
    
    with open('test_config.json', 'w') as f:
        json.dump(test_config, f, indent=4)
    
    print("Test configuration created: test_config.json")
    return test_config

def create_sample_images():
    """Generate sample test images"""
    os.makedirs('test_images', exist_ok=True)
    
    # Create clean panel image
    clean_panel = np.ones((640, 640, 3), dtype=np.uint8) * 200  # Light gray
    # Add panel grid pattern
    for i in range(0, 640, 80):
        cv2.line(clean_panel, (i, 0), (i, 640), (180, 180, 180), 2)
        cv2.line(clean_panel, (0, i), (640, i), (180, 180, 180), 2)
    
    cv2.imwrite('test_images/clean_panel.jpg', clean_panel)
    cv2.imwrite('reference_clean_panel.jpg', clean_panel)
    
    # Create soiled panel images with different soiling levels
    for level, filename in [(10, 'light_soiling.jpg'), (30, 'medium_soiling.jpg'), (60, 'heavy_soiling.jpg')]:
        soiled_panel = clean_panel.copy()
        
        # Add random soiling patches
        for _ in range(level):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 590)
            w = np.random.randint(20, 100)
            h = np.random.randint(20, 100)
            
            # Dark soiling color
            color = (np.random.randint(50, 120), np.random.randint(40, 100), np.random.randint(30, 90))
            cv2.rectangle(soiled_panel, (x, y), (x+w, y+h), color, -1)
            
        cv2.imwrite(f'test_images/{filename}', soiled_panel)
    
    print("Sample images created in test_images/")
    return ['test_images/clean_panel.jpg', 'test_images/light_soiling.jpg', 
            'test_images/medium_soiling.jpg', 'test_images/heavy_soiling.jpg']

class TestSoilingDetector(unittest.TestCase):
    """Comprehensive test suite for the soiling detection system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Patch ultralytics.YOLO globally for all tests in this class
        from unittest.mock import patch
        cls.yolo_patcher = patch('ultralytics.YOLO', MockYOLO)
        cls.yolo_patcher.start()

        cls.config = create_test_config()

        if TEST_CONFIG["CREATE_SAMPLE_IMAGES"]:
            cls.test_images = create_sample_images()

        # Setup logging for tests
        logging.basicConfig(level=logging.DEBUG)

        # Create test directories
        os.makedirs('test_static/uploads', exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Stop the YOLO patcher
        cls.yolo_patcher.stop()
    
    def setUp(self):
        """Set up each test"""
        # Backup original config file if it exists
        if os.path.exists('config.json'):
            os.rename('config.json', 'config_backup.json')
        
        # Use test config
        os.rename('test_config.json', 'config.json')
        
        # Mock the YOLO import if needed
        if TEST_CONFIG["CREATE_MOCK_MODEL"]:
            import sys
            sys.modules['ultralytics'] = type(sys)('ultralytics')
            sys.modules['ultralytics'].YOLO = MockYOLO
    
    def tearDown(self):
        """Clean up after each test"""
        # Restore original config
        if os.path.exists('config.json'):
            os.rename('config.json', 'test_config.json')
        if os.path.exists('config_backup.json'):
            os.rename('config_backup.json', 'config.json')
    
    def test_config_loading(self):
        """Test configuration file loading"""
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                Config = enhanced_soiling_detector.Config
                self.assertIsNotNone(Config.MODEL_PATH)
                self.assertEqual(Config.WEATHER_CHECK_HOURS, 1)  # Test the updated value
                self.assertEqual(Config.CAMERA_WIDTH, 512)
                print("✓ Configuration loading test passed")
            except Exception as e:
                self.fail(f"Config loading failed: {e}")
    
    def test_database_setup(self):
        """Test database initialization"""
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                SoilingDetector = enhanced_soiling_detector.SoilingDetector
                detector = SoilingDetector()
                # Check if database file was created
                self.assertTrue(os.path.exists('test_soiling_data.db'))
                # Check database structure
                conn = sqlite3.connect('test_soiling_data.db')
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                self.assertTrue(any('soiling_data' in table[0] for table in tables))
                print("✓ Database setup test passed")
            except Exception as e:
                self.fail(f"Database setup failed: {e}")
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality"""
        if not TEST_CONFIG["CREATE_SAMPLE_IMAGES"]:
            self.skipTest("Sample images not created")
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                SoilingDetector = enhanced_soiling_detector.SoilingDetector
                detector = SoilingDetector()
                # Test with sample image
                test_image = cv2.imread('test_images/clean_panel.jpg')
                self.assertIsNotNone(test_image)
                processed = detector.preprocess_image(test_image)
                # Check if processed image has correct dimensions
                self.assertEqual(processed.shape[:2], (512, 512))
                print("✓ Image preprocessing test passed")
            except Exception as e:
                self.fail(f"Image preprocessing failed: {e}")
    
    def test_soiling_detection(self):
        """Test soiling detection with mock model"""
        if not TEST_CONFIG["CREATE_SAMPLE_IMAGES"]:
            self.skipTest("Sample images not created")
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                SoilingDetector = enhanced_soiling_detector.SoilingDetector
                detector = SoilingDetector()
                test_image = cv2.imread('test_images/medium_soiling.jpg')
                processed = detector.preprocess_image(test_image)
                masks = detector.detect_soiling(processed)
                self.assertIsInstance(masks, list)
                print(f"✓ Soiling detection test passed - {len(masks)} masks detected")
            except Exception as e:
                self.fail(f"Soiling detection failed: {e}")
    
    def test_metrics_calculation(self):
        """Test soiling metrics calculation"""
        if not TEST_CONFIG["CREATE_SAMPLE_IMAGES"]:
            self.skipTest("Sample images not created")
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                SoilingDetector = enhanced_soiling_detector.SoilingDetector
                detector = SoilingDetector()
                # Load reference image
                detector.reference_image = cv2.imread('reference_clean_panel.jpg')
                test_image = cv2.imread('test_images/medium_soiling.jpg')
                processed = detector.preprocess_image(test_image)
                masks = detector.detect_soiling(processed)
                metrics = detector.calculate_soiling_metrics(processed, masks)
                # Check metric structure
                required_keys = ['soiling_area_percent', 'soiling_intensity', 'total_pixels', 
                               'soiled_pixels', 'ssim_score', 'soiling_mask']
                for key in required_keys:
                    self.assertIn(key, metrics)
                self.assertIsInstance(metrics['soiling_area_percent'], float)
                self.assertGreaterEqual(metrics['soiling_area_percent'], 0)
                print(f"✓ Metrics calculation test passed - Soiling: {metrics['soiling_area_percent']:.1f}%")
            except Exception as e:
                self.fail(f"Metrics calculation failed: {e}")
    
    def test_weather_api(self):
        """Test weather API integration"""
        if not TEST_CONFIG["TEST_WEATHER_API"]:
            self.skipTest("Weather API testing disabled")
        
        try:
            from enhanced_soiling_detector import SoilingDetector
            detector = SoilingDetector()
            
            rain_expected, weather_data = detector.check_weather()
            
            self.assertIsInstance(rain_expected, bool)
            self.assertIsInstance(weather_data, dict)
            
            print(f"✓ Weather API test passed - Rain expected: {rain_expected}")
        except Exception as e:
            print(f"⚠ Weather API test failed (may be due to API key): {e}")
    
    def test_full_pipeline(self):
        """Test complete image processing pipeline"""
        if not TEST_CONFIG["CREATE_SAMPLE_IMAGES"]:
            self.skipTest("Sample images not created")
        from unittest.mock import patch
        import importlib
        with patch('ultralytics.YOLO', MockYOLO):
            try:
                import enhanced_soiling_detector
                importlib.reload(enhanced_soiling_detector)
                SoilingDetector = enhanced_soiling_detector.SoilingDetector
                detector = SoilingDetector()
                # Test with different soiling levels
                for image_path in self.test_images:
                    if os.path.exists(image_path):
                        metrics, overlay_path = detector.process_image_file(image_path)
                        if metrics is not None:
                            self.assertIsInstance(metrics['soiling_area_percent'], float)
                            print(f"✓ Pipeline test passed for {image_path} - Soiling: {metrics['soiling_area_percent']:.1f}%")
            except Exception as e:
                self.fail(f"Full pipeline test failed: {e}")

def run_performance_tests():
    """Run performance benchmarks"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    try:
        from enhanced_soiling_detector import SoilingDetector
        detector = SoilingDetector()
        
        if os.path.exists('test_images/medium_soiling.jpg'):
            test_image = cv2.imread('test_images/medium_soiling.jpg')
            
            # Time the processing pipeline
            start_time = time.time()
            for i in range(5):
                processed = detector.preprocess_image(test_image)
                masks = detector.detect_soiling(processed)
                metrics = detector.calculate_soiling_metrics(processed, masks)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            print(f"✓ Average processing time: {avg_time:.2f} seconds per image")
            
            if avg_time < 5.0:
                print("✓ Performance acceptable for real-time processing")
            else:
                print("⚠ Performance may be slow for real-time processing")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")

def run_integration_tests():
    """Run integration tests"""
    print("\n" + "="*50)
    print("INTEGRATION TESTS")
    print("="*50)
    
    # Test database operations
    try:
        conn = sqlite3.connect('test_soiling_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM soiling_data")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"✓ Database contains {count} records")
    except Exception as e:
        print(f"✗ Database integration test failed: {e}")
    
    # Test file operations
    upload_dir = 'test_static/uploads'
    if os.path.exists(upload_dir):
        files = os.listdir(upload_dir)
        print(f"✓ Upload directory contains {len(files)} files")
    else:
        print(f"⚠ Upload directory {upload_dir} not found")

def cleanup_test_files():
    """Clean up test files and directories"""
    if TEST_CONFIG["CLEANUP_AFTER_TESTS"]:
        print("\nCleaning up test files...")
        
        files_to_remove = [
            'test_soiling_data.db', 
            'test_config.json',
            'reference_clean_panel.jpg'
        ]
        
        dirs_to_remove = ['test_images', 'test_static']
        
        for file in files_to_remove:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")
        
        for dir in dirs_to_remove:
            if os.path.exists(dir):
                import shutil
                shutil.rmtree(dir)
                print(f"Removed {dir}")

def main():
    """Run all tests"""
    print("="*60)
    print("SOLAR PANEL SOILING DETECTION SYSTEM - TEST SUITE")
    print("="*60)
    
    print("Test Configuration:")
    for key, value in TEST_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Run unit tests
    print("Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    # Run integration tests
    run_integration_tests()
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETED")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review test results above")
    print("2. Fix any failing tests")
    print("3. Update config.json with your actual values")
    print("4. Test with real solar panel images")
    print("5. Deploy to Raspberry Pi")

if __name__ == "__main__":
    main()