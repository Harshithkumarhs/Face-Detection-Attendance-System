"""
Logging utility for Face Detection Attendance System
"""

import logging
import logging.handlers
import os
from datetime import datetime
from config import LOGGING_CONFIG

class AttendanceLogger:
    """Custom logger for the attendance system"""
    
    def __init__(self, name="AttendanceSystem"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers"""
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["log_file"],
            maxBytes=LOGGING_CONFIG["max_file_size"],
            backupCount=LOGGING_CONFIG["backup_count"]
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_attendance(self, person_name, action, confidence=None, timestamp=None):
        """Log attendance events"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_message = f"ATTENDANCE: {person_name} - {action}"
        if confidence is not None:
            log_message += f" (Confidence: {confidence:.3f})"
        
        self.info(log_message)
    
    def log_face_detection(self, num_faces, confidence_scores):
        """Log face detection events"""
        self.info(f"FACE_DETECTION: Detected {num_faces} faces with confidences: {confidence_scores}")
    
    def log_model_operation(self, operation, details=None):
        """Log model operations"""
        message = f"MODEL: {operation}"
        if details:
            message += f" - {details}"
        self.info(message)
    
    def log_system_event(self, event, details=None):
        """Log system events"""
        message = f"SYSTEM: {event}"
        if details:
            message += f" - {details}"
        self.info(message)

# Global logger instance
logger = AttendanceLogger() 