"""
Test script for Face Detection Attendance System
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import torch
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_CONFIG, DATABASE_CONFIG, FACE_NAME_MAPPING
from utils.logger import logger
from utils.attendance_manager import AttendanceManager
from utils.data_processor import ImageAugmenter, DatasetProcessor

class TestFaceDetectionAttendanceSystem(unittest.TestCase):
    """Test cases for the face detection attendance system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_attendance.db"
        self.attendance_manager = AttendanceManager(self.test_db_path)
        
        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        logger.log_system_event("Test setup completed")
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        logger.log_system_event("Test cleanup completed")
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(MODEL_CONFIG)
        self.assertIsNotNone(DATABASE_CONFIG)
        self.assertIsNotNone(FACE_NAME_MAPPING)
        
        # Check required config keys
        required_model_keys = ["model_path", "embedding_size", "threshold"]
        for key in required_model_keys:
            self.assertIn(key, MODEL_CONFIG)
        
        logger.log_system_event("Configuration test passed")
    
    def test_attendance_manager(self):
        """Test attendance manager functionality"""
        # Test adding person
        self.attendance_manager.add_person("Test Person", 999)
        
        # Test recording attendance
        result = self.attendance_manager.record_attendance(999, "check_in", 0.95)
        self.assertTrue(result)
        
        # Test getting today's attendance
        df = self.attendance_manager.get_today_attendance()
        self.assertIsInstance(df, pd.DataFrame)
        
        logger.log_system_event("Attendance manager test passed")
    
    def test_image_augmentation(self):
        """Test image augmentation utilities"""
        augmenter = ImageAugmenter()
        
        # Test rotation
        rotated = augmenter.apply_rotation(self.test_image, 45)
        self.assertIsInstance(rotated, Image.Image)
        
        # Test contrast adjustment
        contrasted = augmenter.adjust_contrast(self.test_image, 1.5)
        self.assertIsInstance(contrasted, Image.Image)
        
        # Test noise addition
        noisy = augmenter.add_noise(self.test_image, 100)
        self.assertIsInstance(noisy, Image.Image)
        
        # Test sepia
        sepia = augmenter.apply_sepia(self.test_image)
        self.assertIsInstance(sepia, Image.Image)
        
        # Test black and white conversion
        bw = augmenter.convert_to_bw(self.test_image)
        self.assertIsInstance(bw, Image.Image)
        
        logger.log_system_event("Image augmentation test passed")
    
    def test_dataset_processor(self):
        """Test dataset processor functionality"""
        # Create test directories
        test_input = "test_input"
        test_output = "test_output"
        
        os.makedirs(test_input, exist_ok=True)
        os.makedirs(test_output, exist_ok=True)
        
        # Save test image
        self.test_image.save(os.path.join(test_input, "test.jpg"))
        
        try:
            processor = DatasetProcessor(test_input, test_output)
            
            # Test image processing
            processor.process_images()
            
            # Test path label generation
            processor.generate_path_labels("test_path_labels.csv")
            
            # Check if files were created
            self.assertTrue(os.path.exists("test_path_labels.csv"))
            
            # Clean up
            os.remove("test_path_labels.csv")
            
        finally:
            # Clean up test directories
            import shutil
            if os.path.exists(test_input):
                shutil.rmtree(test_input)
            if os.path.exists(test_output):
                shutil.rmtree(test_output)
        
        logger.log_system_event("Dataset processor test passed")
    
    def test_face_name_mapping(self):
        """Test face name mapping configuration"""
        # Check if mapping contains expected names
        expected_names = ["Chandu", "Harshith", "Anji", "Deekshith", "Giri"]
        
        for name in expected_names:
            self.assertIn(name, FACE_NAME_MAPPING.values())
        
        # Check if all face IDs are integers
        for face_id in FACE_NAME_MAPPING.keys():
            self.assertIsInstance(face_id, int)
        
        logger.log_system_event("Face name mapping test passed")
    
    @patch('torch.cuda.is_available')
    def test_device_detection(self, mock_cuda):
        """Test device detection for GPU/CPU"""
        # Test CPU detection
        mock_cuda.return_value = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(device.type, "cpu")
        
        # Test GPU detection
        mock_cuda.return_value = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(device.type, "cuda")
        
        logger.log_system_event("Device detection test passed")
    
    def test_logger_functionality(self):
        """Test logger functionality"""
        # Test different log levels
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")
        
        # Test custom logging functions
        logger.log_attendance("Test Person", "CHECK_IN", 0.95)
        logger.log_face_detection(2, [0.8, 0.9])
        logger.log_model_operation("Test Operation", "Test Details")
        logger.log_system_event("Test Event", "Test Details")
        
        # Check if log file exists
        from config import LOGGING_CONFIG
        self.assertTrue(os.path.exists(LOGGING_CONFIG["log_file"]))
        
        logger.log_system_event("Logger test passed")
    
    def test_data_structures(self):
        """Test data structure integrity"""
        # Test model config structure
        self.assertIsInstance(MODEL_CONFIG["embedding_size"], int)
        self.assertIsInstance(MODEL_CONFIG["threshold"], float)
        self.assertIsInstance(MODEL_CONFIG["batch_size"], int)
        
        # Test database config structure
        self.assertIsInstance(DATABASE_CONFIG["db_path"], str)
        self.assertIsInstance(DATABASE_CONFIG["csv_path"], str)
        
        # Test face detection config structure
        from config import FACE_DETECTION_CONFIG
        self.assertIsInstance(FACE_DETECTION_CONFIG["min_face_size"], int)
        self.assertIsInstance(FACE_DETECTION_CONFIG["scale_factor"], float)
        
        logger.log_system_event("Data structure test passed")
    
    def test_error_handling(self):
        """Test error handling capabilities"""
        # Test attendance manager with invalid data
        result = self.attendance_manager.record_attendance(None, "invalid_action")
        self.assertFalse(result)
        
        # Test with invalid confidence
        result = self.attendance_manager.record_attendance(999, "check_in", "invalid_confidence")
        self.assertFalse(result)
        
        logger.log_system_event("Error handling test passed")

def run_system_check():
    """Run a comprehensive system check"""
    print("Face Detection Attendance System - System Check")
    print("=" * 50)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'mtcnn',
        'pandas', 'PIL', 'matplotlib', 'tqdm'
    ]
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Check file structure
    print("\nChecking file structure:")
    required_files = [
        'config.py', 'train_model.py', 'attendance_system.py',
        'requirements.txt', 'README.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
    
    # Check directories
    required_dirs = ['models', 'utils', 'database', 'logs', 'attendance']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - NOT FOUND")
    
    print("\nSystem check completed!")

if __name__ == "__main__":
    # Run system check
    run_system_check()
    
    # Run tests
    print("\nRunning tests...")
    unittest.main(verbosity=2) 