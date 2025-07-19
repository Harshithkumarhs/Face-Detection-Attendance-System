"""
Configuration settings for Face Detection Attendance System
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
ATTENDANCE_DIR = BASE_DIR / "attendance"
IMAGES_DIR = BASE_DIR / "images"

# Create directories if they don't exist
for directory in [DATABASE_DIR, MODELS_DIR, LOGS_DIR, ATTENDANCE_DIR, IMAGES_DIR]:
    directory.mkdir(exist_ok=True)

# Model settings
MODEL_CONFIG = {
    "model_path": str(MODELS_DIR / "siamese_model.pth"),
    "embedding_size": 512,
    "input_size": (224, 224),
    "threshold": 0.6,
    "margin": 1.0,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 20
}

# Face detection settings
FACE_DETECTION_CONFIG = {
    "min_face_size": 20,
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "confidence_threshold": 0.8,
    "expand_factor": 0.3
}

# Database settings
DATABASE_CONFIG = {
    "db_path": str(DATABASE_DIR / "attendance.db"),
    "csv_path": str(DATABASE_DIR / "image_paths_labels.csv"),
    "embeddings_path": str(DATABASE_DIR / "face_embeddings.csv"),
    "detected_faces_path": str(DATABASE_DIR / "detected_faces.csv")
}

# Attendance settings
ATTENDANCE_CONFIG = {
    "db_path": str(DATABASE_DIR / "attendance.db"),
    "check_in_time": "09:00",
    "check_out_time": "17:00",
    "late_threshold_minutes": 15,
    "attendance_file": str(ATTENDANCE_DIR / "attendance_log.csv"),
    "daily_report_file": str(ATTENDANCE_DIR / "daily_reports"),
    "monthly_report_file": str(ATTENDANCE_DIR / "monthly_reports")
}

# Camera settings
CAMERA_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "ip_camera_url": "http://192.168.249.211:8080/video"
}

# Logging settings
LOGGING_CONFIG = {
    "log_file": str(LOGS_DIR / "attendance_system.log"),
    "log_level": "INFO",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Face name mapping
FACE_NAME_MAPPING = {
    0: "Chandu",
    1: "Harshith", 
    2: "Anji",
    3: "Deekshith",
    4: "Giri",
    5: "Sumant",
    6: "Dileep",
    7: "Deviprasadh",
    8: "Channappa",
    9: "Dhanush B",
    10: "Chandraguptha",
    11: "Dhanusha U",
    12: "Bhuvan",
    13: "Nanda"
}

# Image processing settings
IMAGE_PROCESSING_CONFIG = {
    "augmentation_rotations": [30, -30, 15, -15],
    "contrast_factors": [0.8, 1.0, 1.2, 1.5],
    "noise_intensity": 10000,
    "box_size": (500, 500),
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp"]
}

# GUI settings
GUI_CONFIG = {
    "window_title": "Face Detection Attendance System",
    "window_size": "1200x800",
    "theme": "default",
    "refresh_rate": 1000  # milliseconds
}

# Performance settings
PERFORMANCE_CONFIG = {
    "max_faces_per_frame": 10,
    "processing_interval": 0.1,  # seconds
    "cache_size": 100,
    "enable_gpu": True
} 