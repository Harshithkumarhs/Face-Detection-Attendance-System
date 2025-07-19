"""
Face Detection Utilities using OpenCV and Haar Cascades
"""

import cv2
import numpy as np
from PIL import Image
import torch
from config import FACE_DETECTION_CONFIG, FACE_NAME_MAPPING
from utils.logger import logger
import os

class FaceDetector:
    """
    Face detection and preprocessing utilities using OpenCV
    """
    
    def __init__(self):
        # Load OpenCV's pre-trained face detection model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            # Fallback to a different path
            cascade_path = 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            raise RuntimeError("Face cascade classifier not found")
        
        logger.log_system_event("FaceDetector initialized with OpenCV")
    
    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            List of face dictionaries with bounding boxes and confidence scores
        """
        try:
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Convert PIL to numpy
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image_array
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                image_gray,
                scaleFactor=FACE_DETECTION_CONFIG.get("scale_factor", 1.1),
                minNeighbors=FACE_DETECTION_CONFIG.get("min_neighbors", 5),
                minSize=(FACE_DETECTION_CONFIG.get("min_face_size", 20), 
                        FACE_DETECTION_CONFIG.get("min_face_size", 20))
            )
            
            # Convert to MTCNN-like format
            face_results = []
            for (x, y, w, h) in faces:
                face_results.append({
                    'box': [x, y, w, h],
                    'confidence': 0.9,  # OpenCV doesn't provide confidence scores
                    'keypoints': {
                        'left_eye': [x + w//3, y + h//3],
                        'right_eye': [x + 2*w//3, y + h//3],
                        'nose': [x + w//2, y + h//2],
                        'mouth_left': [x + w//4, y + 3*h//4],
                        'mouth_right': [x + 3*w//4, y + 3*h//4]
                    }
                })
            
            logger.log_face_detection(len(face_results), [face['confidence'] for face in face_results])
            
            return face_results
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face_region(self, image, face_box, expand_factor=0.3):
        """
        Extract face region from image with optional expansion
        
        Args:
            image: PIL Image or numpy array
            face_box: Dictionary with 'box' key containing [x, y, width, height]
            expand_factor: Factor to expand the bounding box
            
        Returns:
            PIL Image of the face region
        """
        try:
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = image
            
            x, y, width, height = face_box['box']
            
            # Expand the bounding box
            new_width = int(width * (1 + expand_factor))
            new_height = int(height * (1 + expand_factor))
            
            x_new = max(0, x - (new_width - width) // 2)
            y_new = max(0, y - (new_height - height) // 2)
            
            # Ensure coordinates are within image bounds
            img_width, img_height = image_pil.size
            x_new = max(0, min(x_new, img_width - 1))
            y_new = max(0, min(y_new, img_height - 1))
            new_width = min(new_width, img_width - x_new)
            new_height = min(new_height, img_height - y_new)
            
            # Extract face region
            face_region = image_pil.crop((x_new, y_new, x_new + new_width, y_new + new_height))
            
            return face_region, (x_new, y_new, new_width, new_height)
            
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            return None, None
    
    def draw_face_boxes(self, image, faces, recognized_faces=None):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: PIL Image or numpy array
            faces: List of detected faces
            recognized_faces: Dictionary mapping face indices to recognition results
            
        Returns:
            Image with drawn bounding boxes
        """
        try:
            if isinstance(image, np.ndarray):
                image_draw = image.copy()
            else:
                image_draw = np.array(image)
                if len(image_draw.shape) == 3 and image_draw.shape[2] == 3:
                    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_RGB2BGR)
            
            for i, face in enumerate(faces):
                x, y, width, height = face['box']
                confidence = face['confidence']
                
                # Expand bounding box
                expand_factor = FACE_DETECTION_CONFIG.get("expand_factor", 0.3)
                new_width = int(width * (1 + expand_factor))
                new_height = int(height * (1 + expand_factor))
                
                x_new = max(0, x - (new_width - width) // 2)
                y_new = max(0, y - (new_height - height) // 2)
                
                # Draw bounding box
                color = (0, 255, 0) if recognized_faces and i in recognized_faces else (0, 0, 255)
                cv2.rectangle(image_draw, (x_new, y_new), 
                            (x_new + new_width, y_new + new_height), color, 2)
                
                # Add label
                if recognized_faces and i in recognized_faces:
                    face_id, similarity = recognized_faces[i]
                    if face_id is not None:
                        name = FACE_NAME_MAPPING.get(face_id, f"Unknown_{face_id}")
                        label = f"{name} ({similarity:.2f})"
                    else:
                        label = f"Unknown ({similarity:.2f})"
                else:
                    label = f"Face ({confidence:.2f})"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(image_draw, (x_new, y_new - label_height - 10),
                            (x_new + label_width, y_new), color, -1)
                
                # Draw label text
                cv2.putText(image_draw, label, (x_new, y_new - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return image_draw
            
        except Exception as e:
            logger.error(f"Drawing face boxes failed: {e}")
            return image
    
    def preprocess_face(self, face_image, target_size=(224, 224)):
        """
        Preprocess face image for model input
        
        Args:
            face_image: PIL Image of face
            target_size: Target size for resizing
            
        Returns:
            Preprocessed tensor
        """
        try:
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            return transform(face_image)
            
        except Exception as e:
            logger.error(f"Face preprocessing failed: {e}")
            return None

class CameraManager:
    """
    Camera management for real-time face detection
    """
    
    def __init__(self, camera_index=0, ip_camera_url=None):
        self.camera_index = camera_index
        self.ip_camera_url = ip_camera_url
        self.cap = None
        self.face_detector = FaceDetector()
        
    def start_camera(self):
        """
        Start camera capture
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            if self.ip_camera_url:
                self.cap = cv2.VideoCapture(self.ip_camera_url)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.log_system_event("Camera started", f"Index: {self.camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"Camera start failed: {e}")
            return False
    
    def get_frame(self):
        """
        Get current frame from camera
        
        Returns:
            Frame as numpy array or None if failed
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            return None
        
        return frame
    
    def stop_camera(self):
        """
        Stop camera capture
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.log_system_event("Camera stopped")
    
    def __del__(self):
        self.stop_camera()

class FaceRecognitionPipeline:
    """
    Complete pipeline for face detection and recognition
    """
    
    def __init__(self, face_model, camera_manager=None):
        self.face_model = face_model
        self.camera_manager = camera_manager or CameraManager()
        self.face_detector = FaceDetector()
        self.known_embeddings = {}
        
    def load_known_faces(self, embeddings_path):
        """
        Load known face embeddings
        
        Args:
            embeddings_path: Path to embeddings CSV file
        """
        try:
            import pandas as pd
            df = pd.read_csv(embeddings_path)
            
            for _, row in df.iterrows():
                face_no = row['face_no']
                embedding = eval(row['embedding'])  # Convert string to list
                self.known_embeddings[face_no] = embedding
            
            logger.log_system_event("Known faces loaded", f"Count: {len(self.known_embeddings)}")
            
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
    
    def process_frame(self, frame, threshold=0.6):
        """
        Process a single frame for face detection and recognition
        
        Args:
            frame: Input frame
            threshold: Recognition threshold
            
        Returns:
            Tuple of (processed_frame, recognized_faces)
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return frame, {}
            
            recognized_faces = {}
            
            for i, face in enumerate(faces):
                # Extract face region
                face_image, bbox = self.face_detector.extract_face_region(frame, face)
                
                if face_image is None:
                    continue
                
                # Get face embedding
                embedding = self.face_model.get_face_embedding(face_image)
                
                # Find most similar face
                face_id, similarity = self.face_model.find_most_similar_face(
                    embedding, self.known_embeddings, threshold
                )
                
                recognized_faces[i] = (face_id, similarity)
            
            # Draw bounding boxes
            processed_frame = self.face_detector.draw_face_boxes(frame, faces, recognized_faces)
            
            return processed_frame, recognized_faces
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame, {}
    
    def run_realtime_detection(self, display=True):
        """
        Run real-time face detection and recognition
        
        Args:
            display: Whether to display the video feed
        """
        if not self.camera_manager.start_camera():
            return
        
        try:
            while True:
                frame = self.camera_manager.get_frame()
                if frame is None:
                    continue
                
                # Process frame
                processed_frame, recognized_faces = self.process_frame(frame)
                
                if display:
                    cv2.imshow('Face Detection Attendance System', processed_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                elif key == ord('s'):  # Save frame
                    cv2.imwrite('captured_frame.jpg', processed_frame)
                    logger.log_system_event("Frame saved", "captured_frame.jpg")
        
        except KeyboardInterrupt:
            logger.log_system_event("Realtime detection stopped by user")
        finally:
            self.camera_manager.stop_camera()
            if display:
                cv2.destroyAllWindows() 