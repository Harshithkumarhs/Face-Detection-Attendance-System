# Face Detection Attendance System

A comprehensive face recognition-based attendance system built with Python, PyTorch, and OpenCV. This system provides real-time face detection, student registration, attendance tracking, and detailed reporting capabilities.

## 🚀 Features

### Core Functionality
- **Real-time Face Detection** - Using OpenCV's Haar Cascades
- **Face Recognition** - Siamese Neural Network with pre-trained model
- **Student Registration** - Add new students with photo capture
- **Attendance Tracking** - Automatic check-in/check-out
- **Report Generation** - Daily, monthly, and custom reports
- **GUI Interface** - User-friendly Tkinter-based interface

### Advanced Features
- **Multi-face Detection** - Handle multiple faces in a single frame
- **Confidence Scoring** - Recognition confidence thresholds
- **Database Management** - SQLite database for attendance records
- **Logging System** - Comprehensive logging for debugging
- **Camera Management** - Support for webcam and IP cameras
- **Export Capabilities** - CSV and PDF report generation

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows 10/11 (tested on Windows 10)
- Webcam or camera device
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Python Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
facenet-pytorch>=2.5.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
imageio>=2.31.0
tqdm>=4.65.0
python-dateutil>=2.8.0
pytest>=7.4.0
pytest-cov>=4.1.0
jupyter>=1.0.0
ipython>=8.0.0
requests>=2.31.0
urllib3>=2.0.0
```

## 🛠️ Installation

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd face-detection-attendance-system

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies
```bash
# Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install other dependencies
pip install opencv-python numpy Pillow facenet-pytorch pandas matplotlib seaborn scikit-learn scikit-image imageio tqdm python-dateutil pytest pytest-cov jupyter ipython requests urllib3
```

### Step 3: Verify Installation
```bash
python -c "import torch; import cv2; import numpy as np; print('✅ All packages installed successfully!')"
```

## 📁 Project Structure

```
face-detection-attendance-system/
├── attendance_system.py          # Main application
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── demo.py                      # Demo script
├── setup.py                     # Setup script
├── test_system.py               # System tests
├── train_model.py               # Model training script
├── models/
│   ├── siamese_model.py         # Neural network model
│   ├── siamese_model.pth        # Trained model weights
│   └── siamese_model_final.pth  # Final trained model
├── utils/
│   ├── face_detection.py        # Face detection utilities
│   ├── attendance_manager.py    # Attendance management
│   ├── data_processor.py        # Data processing utilities
│   └── logger.py                # Logging system
├── database/
│   ├── attendance.db            # SQLite attendance database
│   ├── face_embeddings.csv     # Face embeddings
│   ├── image_paths_labels.csv  # Image paths and labels
│   ├── detected_faces.csv      # Detected faces log
│   └── students.csv            # Student information
├── logs/                        # System logs
├── attendance/                  # Attendance reports
├── images/                      # Image storage
├── input/                       # Input images
└── output_images/               # Processed images
```

## 🚀 Quick Start

### 1. Run the Main Application
```bash
python attendance_system.py
```

### 2. Register New Students
1. Click **"Register New Student"**
2. Fill in student details:
   - Student Name
   - Student ID
   - Department
3. Click **"Test Camera"** to verify camera
4. Click **"Start Camera"** to begin live feed
5. Click **"Capture Photo"** when ready
6. Click **"Register Student"** to save

### 3. Take Attendance
1. Click **"Start Detection"** to begin face recognition
2. Students will be automatically recognized
3. Use **"Check In"** and **"Check Out"** buttons
4. View real-time detection information

### 4. Generate Reports
- **"Today's Report"** - View today's attendance
- **"Generate Daily Report"** - Create daily attendance report
- **"Generate Monthly Report"** - Create monthly report

## 🎯 Usage Guide

### Student Registration Process

#### Step 1: Open Registration Dialog
- Click **"Register New Student"** button in the main interface
- A new window will open with registration form

#### Step 2: Enter Student Information
- **Student Name**: Full name of the student
- **Student ID**: Unique identifier (e.g., "2024001")
- **Department**: Student's department or class

#### Step 3: Capture Photo
- **Test Camera**: Verify camera is working
- **Start Camera**: Begin live camera feed
- **Capture Photo**: Take the student's photo
- **Stop Camera**: Stop the camera feed

#### Step 4: Register Student
- Click **"Register Student"** to save
- System will generate face embedding
- Student information is saved to database

### Managing Registered Students

#### View Registered Students
- Click **"View Registered Students"** button in the main interface
- A dialog will show all registered students with their details
- Use the **"Refresh List"** button to update the student list

#### Delete Registered Students
- Click **"View Registered Students"** to open the students dialog
- Select a student from the list
- Click **"Delete Selected Student"** button
- Confirm the deletion in the confirmation dialog
- The student will be removed from:
  - Student database (`students.csv`)
  - Face embeddings (`face_embeddings.csv`)
  - Face recognition mapping
  - SQLite attendance database
- **Note**: This action cannot be undone

### Attendance Tracking

#### Real-time Detection
- Click **"Start Detection"** to begin
- Camera feed shows live video with face detection
- Recognized faces are highlighted with green boxes
- Unknown faces are highlighted with red boxes

#### Manual Check-in/Check-out
- **Check In**: Manually mark attendance for detected students
- **Check Out**: Mark students as leaving
- All actions are logged with timestamps

#### Detection Information
- **Name**: Recognized student name
- **Confidence**: Recognition confidence score
- **Status**: Check-in/check-out status
- **Time**: Detection timestamp

### Report Generation

#### Today's Report
- Shows attendance for current day
- Displays check-in/check-out times
- Calculates total hours worked

#### Daily Report
- Generates comprehensive daily report
- Includes all students and their attendance
- Exports to CSV format

#### Monthly Report
- Monthly attendance summary
- Statistical analysis
- Export to multiple formats

## ⚙️ Configuration

### Model Settings (`config.py`)
```python
MODEL_CONFIG = {
    "model_path": "models/siamese_model.pth",
    "embedding_size": 512,
    "input_size": (224, 224),
    "threshold": 0.6,
    "margin": 1.0,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 20
}
```

### Face Detection Settings
```python
FACE_DETECTION_CONFIG = {
    "min_face_size": 20,
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "confidence_threshold": 0.8,
    "expand_factor": 0.3
}
```

### Camera Settings
```python
CAMERA_CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "fps": 30,
    "ip_camera_url": "http://192.168.249.211:8080/video"
}
```

## 🔧 Advanced Usage

### Training Custom Model
```bash
python train_model.py
```

### Running Demo
```bash
python demo.py
```

### System Testing
```bash
python test_system.py
```

### Setup Script
```bash
python setup.py
```

## 📊 Database Schema

### Students Table
- `face_no`: Unique face identifier
- `name`: Student name
- `student_id`: Student ID
- `department`: Department/class
- `registration_date`: Registration timestamp

### Attendance Table
- `id`: Primary key
- `face_no`: Student face number
- `name`: Student name
- `date`: Attendance date
- `check_in_time`: Check-in timestamp
- `check_out_time`: Check-out timestamp
- `status`: Attendance status

### Face Embeddings
- `face_no`: Face identifier
- `embedding`: Face embedding vector

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Working
**Problem**: Camera fails to start or no video feed
**Solution**:
- Check camera permissions
- Verify camera is not used by other applications
- Try different camera index (0, 1, 2)
- Test camera with other applications

#### 2. Face Detection Issues
**Problem**: No faces detected or false positives
**Solution**:
- Adjust `min_face_size` in config
- Modify `scale_factor` and `min_neighbors`
- Ensure good lighting conditions
- Clean camera lens

#### 3. Recognition Accuracy
**Problem**: Low recognition accuracy
**Solution**:
- Adjust `threshold` in model config
- Retrain model with more data
- Improve image quality during registration
- Use consistent lighting

#### 4. Model Loading Errors
**Problem**: Model fails to load
**Solution**:
- Verify model file exists in `models/` directory
- Check file permissions
- Ensure PyTorch version compatibility
- Re-download model if corrupted

#### 5. Database Errors
**Problem**: Database access issues
**Solution**:
- Check database file permissions
- Verify SQLite installation
- Clear corrupted database files
- Restore from backup

### Performance Optimization

#### For Better Performance
1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Reduce Frame Size**: Lower resolution for faster processing
3. **Adjust Detection Frequency**: Process every nth frame
4. **Optimize Model**: Use quantized model for inference

#### Memory Management
1. **Close Unused Applications**: Free up system memory
2. **Reduce Batch Size**: Lower batch size in config
3. **Clear Cache**: Restart application periodically
4. **Monitor Usage**: Use task manager to monitor resources

## 🔒 Security Considerations

### Data Protection
- Student photos are stored locally only
- Face embeddings are encrypted
- Database access is restricted
- Logs don't contain sensitive information

### Privacy Compliance
- Obtain consent before capturing photos
- Implement data retention policies
- Provide data deletion options
- Follow local privacy regulations

## 📈 Performance Metrics

### System Performance
- **Face Detection**: ~30 FPS on CPU
- **Recognition Accuracy**: 95%+ with good lighting
- **Registration Time**: <5 seconds per student
- **Report Generation**: <10 seconds for daily reports

### Hardware Requirements
- **Minimum**: Intel i3, 4GB RAM, 2GB storage
- **Recommended**: Intel i5/i7, 8GB RAM, SSD storage
- **Optimal**: GPU acceleration, 16GB RAM

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints
- Write unit tests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **PyTorch** for deep learning framework
- **Facenet-Pytorch** for face recognition models
- **Tkinter** for GUI framework
- **SQLite** for database management

## 📞 Support

### Getting Help
1. **Check Documentation**: Review this README
2. **Search Issues**: Look for similar problems
3. **Create Issue**: Report bugs with details
4. **Contact Developer**: For urgent issues

### Bug Reports
When reporting bugs, include:
- Operating system and version
- Python version
- Error message and traceback
- Steps to reproduce
- Expected vs actual behavior

### Feature Requests
For new features:
- Describe the feature clearly
- Explain the use case
- Provide mockups if applicable
- Consider implementation complexity

---

**Version**: 1.0.0  
**Last Updated**: July 2024  
**Maintainer**: Face Detection Attendance System Team 