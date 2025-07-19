"""
Main Face Detection Attendance System Application
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
import pandas as pd

from models.siamese_model import FaceRecognitionModel
from utils.face_detection import FaceRecognitionPipeline, CameraManager
from utils.attendance_manager import AttendanceManager
from utils.logger import logger
from config import MODEL_CONFIG, DATABASE_CONFIG, FACE_NAME_MAPPING, GUI_CONFIG

class AttendanceSystemGUI:
    """
    GUI for the Face Detection Attendance System
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUI_CONFIG["window_title"])
        self.root.geometry(GUI_CONFIG["window_size"])
        
        # Initialize components
        self.face_model = FaceRecognitionModel()
        self.attendance_manager = AttendanceManager()
        self.camera_manager = CameraManager()
        
        # Load model and embeddings
        self.load_model_and_embeddings()
        
        # GUI variables
        self.is_detecting = False
        self.current_frame = None
        self.known_embeddings = {}
        
        # Setup GUI
        self.setup_gui()
        
        logger.log_system_event("GUI initialized")
    
    def load_model_and_embeddings(self):
        """Load trained model and face embeddings"""
        try:
            # Load model
            if not self.face_model.load_model(MODEL_CONFIG["model_path"]):
                logger.error("Failed to load model")
                messagebox.showerror("Error", "Failed to load face recognition model")
                return False
            
            # Load embeddings
            try:
                embeddings_df = pd.read_csv(DATABASE_CONFIG["embeddings_path"])
                for _, row in embeddings_df.iterrows():
                    self.known_embeddings[row['face_no']] = eval(row['embedding'])
                logger.log_system_event("Embeddings loaded", f"Count: {len(self.known_embeddings)}")
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                messagebox.showwarning("Warning", "Failed to load face embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model and embeddings: {e}")
            return False
    
    def setup_gui(self):
        """Setup the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Registration controls
        registration_frame = ttk.LabelFrame(left_panel, text="Student Registration", padding="5")
        registration_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.register_btn = ttk.Button(registration_frame, text="Register New Student", command=self.register_student)
        self.register_btn.pack(fill=tk.X, pady=2)
        
        self.view_students_btn = ttk.Button(registration_frame, text="View Registered Students", command=self.view_students)
        self.view_students_btn.pack(fill=tk.X, pady=2)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(left_panel, text="Camera", padding="5")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(camera_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Attendance controls
        attendance_frame = ttk.LabelFrame(left_panel, text="Attendance", padding="5")
        attendance_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.check_in_btn = ttk.Button(attendance_frame, text="Check In", command=self.check_in)
        self.check_in_btn.pack(fill=tk.X, pady=2)
        
        self.check_out_btn = ttk.Button(attendance_frame, text="Check Out", command=self.check_out)
        self.check_out_btn.pack(fill=tk.X, pady=2)
        
        # Reports frame
        reports_frame = ttk.LabelFrame(left_panel, text="Reports", padding="5")
        reports_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.today_report_btn = ttk.Button(reports_frame, text="Today's Report", command=self.show_today_report)
        self.today_report_btn.pack(fill=tk.X, pady=2)
        
        self.daily_report_btn = ttk.Button(reports_frame, text="Generate Daily Report", command=self.generate_daily_report)
        self.daily_report_btn.pack(fill=tk.X, pady=2)
        
        self.monthly_report_btn = ttk.Button(reports_frame, text="Generate Monthly Report", command=self.generate_monthly_report)
        self.monthly_report_btn.pack(fill=tk.X, pady=2)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=MODEL_CONFIG["threshold"])
        ttk.Label(settings_frame, text="Recognition Threshold:").pack(anchor=tk.W)
        self.threshold_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL)
        self.threshold_scale.pack(fill=tk.X, pady=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(left_panel, text="Status", padding="5")
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(anchor=tk.W)
        
        # Right panel - Video and Information
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Video frame
        video_frame = ttk.LabelFrame(right_panel, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="No camera feed")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Information frame
        info_frame = ttk.LabelFrame(right_panel, text="Detection Information", padding="5")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        
        # Create treeview for detected faces
        columns = ('Name', 'Confidence', 'Status', 'Time')
        self.detection_tree = ttk.Treeview(info_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.detection_tree.heading(col, text=col)
            self.detection_tree.column(col, width=100)
        
        self.detection_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.detection_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.detection_tree.configure(yscrollcommand=scrollbar.set)
        
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
    
    def start_detection(self):
        """Start face detection"""
        if not self.camera_manager.start_camera():
            messagebox.showerror("Error", "Failed to start camera")
            return
        
        self.is_detecting = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Detection Active")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.log_system_event("Face detection started")
    
    def stop_detection(self):
        """Stop face detection"""
        self.is_detecting = False
        self.camera_manager.stop_camera()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Detection Stopped")
        
        # Clear video feed
        self.video_label.config(text="No camera feed")
        
        logger.log_system_event("Face detection stopped")
    
    def detection_loop(self):
        """Main detection loop"""
        while self.is_detecting:
            frame = self.camera_manager.get_frame()
            if frame is not None:
                # Process frame
                processed_frame, recognized_faces = self.process_frame(frame)
                
                # Update GUI
                self.update_video_feed(processed_frame)
                self.update_detection_info(recognized_faces)
            
            time.sleep(0.1)  # 10 FPS
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        try:
            # Detect faces
            faces = self.camera_manager.face_detector.detect_faces(frame)
            
            if not faces:
                return frame, {}
            
            recognized_faces = {}
            
            for i, face in enumerate(faces):
                # Extract face region
                face_image, bbox = self.camera_manager.face_detector.extract_face_region(frame, face)
                
                if face_image is None:
                    continue
                
                # Get face embedding
                embedding = self.face_model.get_face_embedding(face_image)
                
                # Find most similar face
                threshold = self.threshold_var.get()
                face_id, similarity = self.face_model.find_most_similar_face(
                    embedding, self.known_embeddings, threshold
                )
                
                recognized_faces[i] = (face_id, similarity)
            
            # Draw bounding boxes
            processed_frame = self.camera_manager.face_detector.draw_face_boxes(frame, faces, recognized_faces)
            
            return processed_frame, recognized_faces
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame, {}
    
    def update_video_feed(self, frame):
        """Update the video feed in GUI"""
        try:
            # Resize frame for display
            height, width = frame.shape[:2]
            max_size = 400
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference
            
        except Exception as e:
            logger.error(f"Failed to update video feed: {e}")
    
    def update_detection_info(self, recognized_faces):
        """Update detection information in GUI"""
        try:
            # Clear existing items
            for item in self.detection_tree.get_children():
                self.detection_tree.delete(item)
            
            # Add new detections
            for face_id, (person_id, confidence) in recognized_faces.items():
                if person_id is not None:
                    name = FACE_NAME_MAPPING.get(person_id, f"Unknown_{person_id}")
                    status = "Recognized"
                else:
                    name = "Unknown"
                    status = "Not Recognized"
                
                time_str = datetime.now().strftime("%H:%M:%S")
                
                self.detection_tree.insert('', 'end', values=(name, f"{confidence:.3f}", status, time_str))
            
        except Exception as e:
            logger.error(f"Failed to update detection info: {e}")
    
    def check_in(self):
        """Manual check-in for detected person"""
        try:
            # Get selected item from treeview
            selection = self.detection_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a person to check in")
                return
            
            item = self.detection_tree.item(selection[0])
            name = item['values'][0]
            
            # Find face_id from name
            face_id = None
            for fid, fname in FACE_NAME_MAPPING.items():
                if fname == name:
                    face_id = fid
                    break
            
            if face_id is None:
                messagebox.showerror("Error", "Could not identify person")
                return
            
            # Record attendance
            if self.attendance_manager.record_attendance(face_id, "check_in"):
                messagebox.showinfo("Success", f"{name} checked in successfully")
                self.status_label.config(text=f"{name} checked in")
            else:
                messagebox.showwarning("Warning", f"{name} already checked in today")
                
        except Exception as e:
            logger.error(f"Check-in failed: {e}")
            messagebox.showerror("Error", f"Check-in failed: {e}")
    
    def check_out(self):
        """Manual check-out for detected person"""
        try:
            # Get selected item from treeview
            selection = self.detection_tree.selection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a person to check out")
                return
            
            item = self.detection_tree.item(selection[0])
            name = item['values'][0]
            
            # Find face_id from name
            face_id = None
            for fid, fname in FACE_NAME_MAPPING.items():
                if fname == name:
                    face_id = fid
                    break
            
            if face_id is None:
                messagebox.showerror("Error", "Could not identify person")
                return
            
            # Record attendance
            if self.attendance_manager.record_attendance(face_id, "check_out"):
                messagebox.showinfo("Success", f"{name} checked out successfully")
                self.status_label.config(text=f"{name} checked out")
            else:
                messagebox.showwarning("Warning", f"No check-in record found for {name}")
                
        except Exception as e:
            logger.error(f"Check-out failed: {e}")
            messagebox.showerror("Error", f"Check-out failed: {e}")
    
    def show_today_report(self):
        """Show today's attendance report"""
        try:
            df = self.attendance_manager.get_today_attendance()
            
            if df.empty:
                messagebox.showinfo("Report", "No attendance records for today")
                return
            
            # Create report window
            report_window = tk.Toplevel(self.root)
            report_window.title("Today's Attendance Report")
            report_window.geometry("600x400")
            
            # Create treeview
            columns = ('Name', 'Check In', 'Check Out', 'Hours', 'Status')
            tree = ttk.Treeview(report_window, columns=columns, show='headings')
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
            
            # Add data
            for _, row in df.iterrows():
                check_in = row['check_in_time'] if pd.notna(row['check_in_time']) else ''
                check_out = row['check_out_time'] if pd.notna(row['check_out_time']) else ''
                hours = f"{row['total_hours']:.2f}" if pd.notna(row['total_hours']) else ''
                
                tree.insert('', 'end', values=(
                    row['person_name'],
                    check_in,
                    check_out,
                    hours,
                    row['status']
                ))
            
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
        except Exception as e:
            logger.error(f"Failed to show today's report: {e}")
            messagebox.showerror("Error", f"Failed to show report: {e}")
    
    def generate_daily_report(self):
        """Generate daily attendance report"""
        try:
            self.attendance_manager.generate_daily_report()
            messagebox.showinfo("Success", "Daily report generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def generate_monthly_report(self):
        """Generate monthly attendance report"""
        try:
            report_path = self.attendance_manager.generate_monthly_report()
            messagebox.showinfo("Success", f"Monthly report generated: {report_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate monthly report: {e}")
    
    def register_student(self):
        """Open student registration dialog"""
        registration_dialog = StudentRegistrationDialog(self.root, self.face_model, self.known_embeddings)
        self.root.wait_window(registration_dialog.dialog)
        # Reload embeddings after registration
        self.load_model_and_embeddings()
    
    def view_students(self):
        """Show registered students dialog"""
        students_dialog = ViewStudentsDialog(self.root, self.known_embeddings)
        self.root.wait_window(students_dialog.dialog)
        # Reload embeddings after dialog closes in case students were deleted
        self.load_model_and_embeddings()
    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_detection()
            logger.log_system_event("Application stopped by user")

class StudentRegistrationDialog:
    """Dialog for registering new students"""
    
    def __init__(self, parent, face_model, known_embeddings):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Register New Student")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.face_model = face_model
        self.known_embeddings = known_embeddings
        self.captured_image = None
        self.is_capturing = False
        self.cap = None
        self.camera_thread = None
        
        self.setup_dialog()
        
        # Bind window close event
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.dialog.destroy()
    
    def setup_dialog(self):
        """Setup the registration dialog"""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Student information
        info_frame = ttk.LabelFrame(main_frame, text="Student Information", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Student Name:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(info_frame, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        ttk.Label(info_frame, text="Student ID:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.id_var = tk.StringVar()
        self.id_entry = ttk.Entry(info_frame, textvariable=self.id_var, width=30)
        self.id_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        ttk.Label(info_frame, text="Department:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.dept_var = tk.StringVar()
        self.dept_entry = ttk.Entry(info_frame, textvariable=self.dept_var, width=30)
        self.dept_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Camera frame
        camera_frame = ttk.LabelFrame(main_frame, text="Capture Photo", padding="5")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.camera_label = ttk.Label(camera_frame, text="Click 'Test Camera' to begin")
        self.camera_label.pack(expand=True)
        
        # Camera controls - First row
        camera_controls1 = ttk.Frame(camera_frame)
        camera_controls1.pack(fill=tk.X, pady=(10, 5))
        
        self.test_camera_btn = ttk.Button(camera_controls1, text="Test Camera", command=self.test_camera)
        self.test_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.start_camera_btn = ttk.Button(camera_controls1, text="Start Camera", command=self.start_camera)
        self.start_camera_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Camera controls - Second row
        camera_controls2 = ttk.Frame(camera_frame)
        camera_controls2.pack(fill=tk.X, pady=(0, 5))
        
        self.capture_btn = ttk.Button(camera_controls2, text="Capture Photo", command=self.capture_photo, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_camera_btn = ttk.Button(camera_controls2, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_btn.pack(side=tk.LEFT)
        
        # Status label
        self.camera_status = ttk.Label(camera_frame, text="Camera ready")
        self.camera_status.pack(anchor=tk.W, pady=(5, 0))
        
        # Registration button
        self.register_btn = ttk.Button(main_frame, text="Register Student", command=self.register_student, state=tk.DISABLED)
        self.register_btn.pack(fill=tk.X, pady=(10, 0))
        
        # Initialize camera
        self.cap = None
        self.camera_thread = None
    
    def test_camera(self):
        """Test if camera is available"""
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.camera_status.config(text="Camera test successful - Ready to start")
                    messagebox.showinfo("Success", "Camera is working properly!")
                else:
                    self.camera_status.config(text="Camera test failed - no frame")
                    messagebox.showerror("Error", "Camera test failed - no frame captured")
            else:
                self.camera_status.config(text="Camera test failed - cannot open")
                messagebox.showerror("Error", "Camera test failed - cannot open camera")
        except Exception as e:
            self.camera_status.config(text=f"Camera test error: {e}")
            messagebox.showerror("Error", f"Camera test failed: {e}")
    
    def start_camera(self):
        """Start camera for photo capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.camera_status.config(text="Failed to open camera")
                messagebox.showerror("Error", "Failed to open camera")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_capturing = True
            self.start_camera_btn.config(state=tk.DISABLED)
            self.test_camera_btn.config(state=tk.DISABLED)
            self.capture_btn.config(state=tk.NORMAL)
            self.stop_camera_btn.config(state=tk.NORMAL)
            self.camera_status.config(text="Camera started - Click 'Capture Photo' when ready")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            self.camera_status.config(text=f"Camera start error: {e}")
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def camera_loop(self):
        """Camera capture loop"""
        while self.is_capturing and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = frame_pil.resize((320, 240))
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update GUI in main thread
                    self.dialog.after(0, self.update_camera_display, frame_tk)
                else:
                    self.dialog.after(0, lambda: self.camera_status.config(text="Camera error - no frame"))
                    break
            except Exception as e:
                self.dialog.after(0, lambda: self.camera_status.config(text=f"Camera error: {e}"))
                print(f"Camera error: {e}")
                break
            time.sleep(0.033)  # ~30 FPS
    
    def update_camera_display(self, frame_tk):
        """Update camera display in main thread"""
        try:
            self.camera_label.config(image=frame_tk)
            self.camera_label.image = frame_tk
        except Exception as e:
            print(f"Display update error: {e}")
    
    def capture_photo(self):
        """Capture photo for registration"""
        if self.cap is not None and self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.captured_image = frame.copy()
                self.camera_status.config(text="Photo captured successfully!")
                messagebox.showinfo("Success", "Photo captured successfully!")
                self.register_btn.config(state=tk.NORMAL)
                self.capture_btn.config(state=tk.DISABLED)
                
                # Show captured image
                frame_rgb = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((320, 240))
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.camera_label.config(image=frame_tk)
                self.camera_label.image = frame_tk
            else:
                self.camera_status.config(text="Failed to capture photo")
                messagebox.showerror("Error", "Failed to capture photo")
        else:
            self.camera_status.config(text="Camera not ready")
            messagebox.showerror("Error", "Camera not ready")
    
    def stop_camera(self):
        """Stop camera"""
        print("Stopping camera...")
        self.is_capturing = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Reset button states
        self.start_camera_btn.config(state=tk.NORMAL)
        self.test_camera_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_camera_btn.config(state=tk.DISABLED)
        self.camera_status.config(text="Camera stopped")
        self.camera_label.config(image="", text="Camera stopped - Click 'Start Camera' to begin again")
    
    def register_student(self):
        """Register the new student"""
        try:
            # Validate inputs
            name = self.name_var.get().strip()
            student_id = self.id_var.get().strip()
            department = self.dept_var.get().strip()
            
            if not name or not student_id or not department:
                messagebox.showerror("Error", "Please fill in all fields")
                return
            
            if self.captured_image is None:
                messagebox.showerror("Error", "Please capture a photo first")
                return
            
            # Generate face embedding
            face_embedding = self.face_model.extract_embedding(self.captured_image)
            if face_embedding is None:
                messagebox.showerror("Error", "No face detected in the captured image")
                return
            
            # Find next available face number
            next_face_no = max(self.known_embeddings.keys()) + 1 if self.known_embeddings else 0
            
            # Save embedding
            self.known_embeddings[next_face_no] = face_embedding.tolist()
            
            # Update face name mapping
            FACE_NAME_MAPPING[next_face_no] = name
            
            # Save to CSV
            self.save_embeddings()
            
            # Save student info
            self.save_student_info(next_face_no, name, student_id, department)
            
            messagebox.showinfo("Success", f"Student {name} registered successfully!")
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {e}")
    
    def save_embeddings(self):
        """Save embeddings to CSV"""
        import pandas as pd
        
        embeddings_data = []
        for face_no, embedding in self.known_embeddings.items():
            embeddings_data.append({
                'face_no': face_no,
                'embedding': str(embedding)
            })
        
        df = pd.DataFrame(embeddings_data)
        df.to_csv(DATABASE_CONFIG["embeddings_path"], index=False)
    
    def save_student_info(self, face_no, name, student_id, department):
        """Save student information"""
        import pandas as pd
        from datetime import datetime
        
        student_info = {
            'face_no': face_no,
            'name': name,
            'student_id': student_id,
            'department': department,
            'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Load existing data or create new
        try:
            df = pd.read_csv('database/students.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['face_no', 'name', 'student_id', 'department', 'registration_date'])
        
        # Add new student
        df = pd.concat([df, pd.DataFrame([student_info])], ignore_index=True)
        df.to_csv('database/students.csv', index=False)

class ViewStudentsDialog:
    """Dialog for viewing registered students"""
    
    def __init__(self, parent, known_embeddings):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Registered Students")
        self.dialog.geometry("600x450")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.known_embeddings = known_embeddings
        self.setup_dialog()
    
    def setup_dialog(self):
        """Setup the students view dialog"""
        # Main frame
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview
        columns = ('Face No', 'Name', 'Student ID', 'Department', 'Registration Date')
        self.students_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=100)
        
        self.students_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Delete button
        self.delete_btn = ttk.Button(button_frame, text="Delete Selected Student", command=self.delete_student, state=tk.DISABLED)
        self.delete_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Refresh button
        self.refresh_btn = ttk.Button(button_frame, text="Refresh List", command=self.load_students)
        self.refresh_btn.pack(side=tk.LEFT)
        
        # Bind selection event
        self.students_tree.bind('<<TreeviewSelect>>', self.on_student_select)
        
        # Load students
        self.load_students()
    
    def on_student_select(self, event):
        """Handle student selection"""
        selection = self.students_tree.selection()
        if selection:
            self.delete_btn.config(state=tk.NORMAL)
        else:
            self.delete_btn.config(state=tk.DISABLED)
    
    def delete_student(self):
        """Delete the selected student"""
        selection = self.students_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a student to delete")
            return
        
        # Get selected student info
        item = self.students_tree.item(selection[0])
        values = item['values']
        face_no = values[0]
        name = values[1]
        
        # Confirmation dialog
        result = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete student:\n\nName: {name}\nFace No: {face_no}\n\nThis action cannot be undone!"
        )
        
        if result:
            try:
                self.perform_deletion(face_no, name)
                messagebox.showinfo("Success", f"Student {name} has been deleted successfully!")
                self.load_students()  # Refresh the list
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete student: {e}")
    
    def perform_deletion(self, face_no, name):
        """Perform the actual deletion from all storage locations"""
        import pandas as pd
        import sqlite3
        from config import DATABASE_CONFIG, FACE_NAME_MAPPING
        
        # 1. Remove from students.csv
        try:
            df = pd.read_csv('database/students.csv')
            df = df[df['face_no'] != face_no]
            df.to_csv('database/students.csv', index=False)
        except FileNotFoundError:
            pass  # File doesn't exist
        
        # 2. Remove from face_embeddings.csv
        try:
            df = pd.read_csv(DATABASE_CONFIG["embeddings_path"])
            df = df[df['face_no'] != face_no]
            df.to_csv(DATABASE_CONFIG["embeddings_path"], index=False)
        except FileNotFoundError:
            pass  # File doesn't exist
        
        # 3. Remove from known_embeddings dictionary
        if face_no in self.known_embeddings:
            del self.known_embeddings[face_no]
        
        # 4. Remove from FACE_NAME_MAPPING
        if face_no in FACE_NAME_MAPPING:
            del FACE_NAME_MAPPING[face_no]
        
        # 5. Remove from SQLite database
        try:
            conn = sqlite3.connect(DATABASE_CONFIG["db_path"])
            cursor = conn.cursor()
            
            # Remove from persons table
            cursor.execute('DELETE FROM persons WHERE face_id = ?', (face_no,))
            
            # Optionally remove attendance records (uncomment if needed)
            # cursor.execute('DELETE FROM attendance WHERE person_id = ?', (face_no,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database deletion warning: {e}")
        
        # 6. Validate embeddings file integrity
        try:
            df = pd.read_csv(DATABASE_CONFIG["embeddings_path"])
            for _, row in df.iterrows():
                embedding = eval(row['embedding'])
                if len(embedding) != 512:
                    print(f"Warning: Embedding for face_no {row['face_no']} has incorrect size {len(embedding)}")
        except Exception as e:
            print(f"Embeddings validation warning: {e}")
        
        # Log the deletion
        from utils.logger import logger
        logger.log_system_event("Student deleted", f"Name: {name}, Face No: {face_no}")
    
    def load_students(self):
        """Load and display registered students"""
        try:
            import pandas as pd
            df = pd.read_csv('database/students.csv')
            
            # Clear existing items
            for item in self.students_tree.get_children():
                self.students_tree.delete(item)
            
            # Add students
            for _, row in df.iterrows():
                self.students_tree.insert('', 'end', values=(
                    row['face_no'],
                    row['name'],
                    row['student_id'],
                    row['department'],
                    row['registration_date']
                ))
                
        except FileNotFoundError:
            messagebox.showinfo("Info", "No students registered yet")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load students: {e}")

def main():
    """Main function to run the attendance system"""
    print("Starting Face Detection Attendance System...")
    
    try:
        app = AttendanceSystemGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 