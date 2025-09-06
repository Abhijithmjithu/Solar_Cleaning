import unittest
import numpy as np
from unittest.mock import patch
import enhanced_soiling_detector

class TestRainSuppressionLogic(unittest.TestCase):
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'check_weather')
    @patch.object(enhanced_soiling_detector.cv2, 'imread')
    @patch.object(enhanced_soiling_detector.SoilingDetector, 'detect_soiling')
    def test_no_alert_below_threshold(self, mock_detect_soiling, mock_imread, mock_weather):
        mock_weather.return_value = (False, {'simulated': True})
        mock_imread.return_value = np.ones((512, 512, 3), dtype=np.uint8)
        # Mask with all zeros to simulate no soiling
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
    def setUp(self):
        self.detector = enhanced_soiling_detector.SoilingDetector()
        # Reset suppression count before each test
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

if __name__ == '__main__':
    unittest.main()
