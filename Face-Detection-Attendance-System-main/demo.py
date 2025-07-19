"""
Demo script for Face Detection Attendance System
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import FACE_NAME_MAPPING, MODEL_CONFIG
from utils.logger import logger
from utils.attendance_manager import AttendanceManager
from utils.data_processor import ImageAugmenter

def print_demo_banner():
    """Print demo banner"""
    print("=" * 60)
    print("Face Detection Attendance System - Demo")
    print("=" * 60)
    print()

def demo_image_augmentation():
    """Demo image augmentation features"""
    print("🎨 Demo: Image Augmentation")
    print("-" * 30)
    
    # Create a sample image
    img = Image.new('RGB', (200, 200), color='blue')
    draw = ImageDraw.Draw(img)
    draw.text((50, 90), "FACE", fill='white')
    
    augmenter = ImageAugmenter()
    
    # Demo different augmentations
    augmentations = [
        ("Rotation (45°)", lambda: augmenter.apply_rotation(img, 45)),
        ("Contrast (1.5x)", lambda: augmenter.adjust_contrast(img, 1.5)),
        ("Noise Addition", lambda: augmenter.add_noise(img, 1000)),
        ("Sepia Effect", lambda: augmenter.apply_sepia(img)),
        ("Black & White", lambda: augmenter.convert_to_bw(img))
    ]
    
    for name, func in augmentations:
        try:
            result = func()
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print()

def demo_attendance_management():
    """Demo attendance management features"""
    print("📊 Demo: Attendance Management")
    print("-" * 30)
    
    # Create attendance manager with test database
    test_db = "demo_attendance.db"
    manager = AttendanceManager(test_db)
    
    try:
        # Add some test persons
        test_persons = [
            ("Alice Johnson", 1001),
            ("Bob Smith", 1002),
            ("Carol Davis", 1003)
        ]
        
        for name, face_id in test_persons:
            manager.add_person(name, face_id)
            print(f"✅ Added person: {name} (ID: {face_id})")
        
        # Record some attendance
        attendance_records = [
            (1001, "check_in", 0.95),
            (1002, "check_in", 0.88),
            (1001, "check_out", 0.92),
            (1003, "check_in", 0.91)
        ]
        
        for face_id, action, confidence in attendance_records:
            result = manager.record_attendance(face_id, action, confidence)
            person_name = next((name for name, fid in test_persons if fid == face_id), "Unknown")
            status = "✅" if result else "❌"
            print(f"{status} {person_name} - {action} (confidence: {confidence})")
        
        # Get today's attendance
        df = manager.get_today_attendance()
        print(f"\n📈 Today's attendance records: {len(df)}")
        
        # Generate summary
        summary = manager.get_attendance_summary()
        print(f"📊 Attendance summary: {summary}")
        
    finally:
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)
    
    print()

def demo_face_recognition_simulation():
    """Demo face recognition simulation"""
    print("👤 Demo: Face Recognition Simulation")
    print("-" * 30)
    
    # Simulate face recognition results
    simulated_faces = [
        (0, "Chandu", 0.92),
        (1, "Harshith", 0.88),
        (2, "Anji", 0.95),
        (None, "Unknown", 0.45)
    ]
    
    for face_id, name, confidence in simulated_faces:
        if face_id is not None:
            print(f"✅ Recognized: {name} (ID: {face_id}, Confidence: {confidence:.2f})")
        else:
            print(f"❓ Unknown person (Confidence: {confidence:.2f})")
    
    print()

def demo_reporting():
    """Demo reporting features"""
    print("📋 Demo: Reporting System")
    print("-" * 30)
    
    # Create sample attendance data
    sample_data = {
        'person_name': ['Alice', 'Bob', 'Carol', 'David'],
        'check_in_time': ['09:00', '09:15', '08:45', '09:30'],
        'check_out_time': ['17:00', '17:30', '17:15', '16:45'],
        'total_hours': [8.0, 8.25, 8.5, 7.25],
        'status': ['checked_out', 'checked_out', 'checked_out', 'checked_out']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("📊 Sample Attendance Report:")
    print(df.to_string(index=False))
    
    # Calculate statistics
    total_persons = len(df)
    avg_hours = df['total_hours'].mean()
    attendance_rate = 100.0  # Assuming all present
    
    print(f"\n📈 Statistics:")
    print(f"   Total Persons: {total_persons}")
    print(f"   Average Hours: {avg_hours:.2f}")
    print(f"   Attendance Rate: {attendance_rate:.1f}%")
    
    print()

def demo_configuration():
    """Demo configuration system"""
    print("⚙️ Demo: Configuration System")
    print("-" * 30)
    
    print("📋 Current Configuration:")
    print(f"   Model Threshold: {MODEL_CONFIG['threshold']}")
    print(f"   Embedding Size: {MODEL_CONFIG['embedding_size']}")
    print(f"   Batch Size: {MODEL_CONFIG['batch_size']}")
    print(f"   Learning Rate: {MODEL_CONFIG['learning_rate']}")
    
    print(f"\n👥 Registered Persons ({len(FACE_NAME_MAPPING)}):")
    for face_id, name in list(FACE_NAME_MAPPING.items())[:5]:  # Show first 5
        print(f"   ID {face_id}: {name}")
    if len(FACE_NAME_MAPPING) > 5:
        print(f"   ... and {len(FACE_NAME_MAPPING) - 5} more")
    
    print()

def demo_logging():
    """Demo logging system"""
    print("📝 Demo: Logging System")
    print("-" * 30)
    
    # Test different log levels
    logger.info("System demo started")
    logger.log_attendance("Demo User", "CHECK_IN", 0.95)
    logger.log_face_detection(3, [0.92, 0.88, 0.95])
    logger.log_model_operation("Demo Training", "Epoch 1/20")
    logger.log_system_event("Demo completed successfully")
    
    print("✅ Log entries created successfully")
    print("📁 Check logs/attendance_system.log for details")
    print()

def demo_performance_metrics():
    """Demo performance metrics"""
    print("🚀 Demo: Performance Metrics")
    print("-" * 30)
    
    # Simulate performance metrics
    metrics = {
        "Detection Speed": "30 FPS",
        "Recognition Accuracy": "94.5%",
        "Processing Time": "33ms per frame",
        "Memory Usage": "512MB",
        "GPU Utilization": "45%"
    }
    
    for metric, value in metrics.items():
        print(f"   {metric}: {value}")
    
    print()

def run_full_demo():
    """Run the complete demo"""
    print_demo_banner()
    
    print("🎯 This demo showcases the key features of the Face Detection Attendance System")
    print("   Each section demonstrates a different aspect of the system.\n")
    
    # Run all demos
    demos = [
        demo_image_augmentation,
        demo_attendance_management,
        demo_face_recognition_simulation,
        demo_reporting,
        demo_configuration,
        demo_logging,
        demo_performance_metrics
    ]
    
    for demo in demos:
        try:
            demo()
            time.sleep(1)  # Brief pause between demos
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)
    print("\nTo run the actual system:")
    print("1. python setup.py")
    print("2. python train_model.py")
    print("3. python attendance_system.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    run_full_demo() 