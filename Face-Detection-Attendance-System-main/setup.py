"""
Setup script for Face Detection Attendance System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("Face Detection Attendance System - Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "models",
        "logs", 
        "attendance",
        "images",
        "database"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def check_existing_data():
    """Check for existing data files"""
    print("\nChecking existing data...")
    
    data_files = [
        "database/image_paths_labels.csv",
        "database/face_embeddings.csv",
        "siamese_model.pth"
    ]
    
    found_files = []
    for file in data_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"✅ Found {file}")
        else:
            print(f"⚠️  Missing {file}")
    
    if found_files:
        print(f"\nFound {len(found_files)} existing data files")
        return True
    else:
        print("\nNo existing data found - you'll need to train the model")
        return False

def setup_configuration():
    """Setup configuration files"""
    print("\nSetting up configuration...")
    
    # Check if config.py exists
    if not os.path.exists("config.py"):
        print("❌ config.py not found!")
        return False
    
    print("✅ Configuration file found")
    
    # Test configuration import
    try:
        from config import MODEL_CONFIG, DATABASE_CONFIG, FACE_NAME_MAPPING
        print("✅ Configuration loaded successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to load configuration: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    required_modules = [
        "torch",
        "torchvision", 
        "cv2",
        "mtcnn",
        "pandas",
        "PIL",
        "matplotlib",
        "tqdm"
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def setup_database():
    """Setup database"""
    print("\nSetting up database...")
    
    try:
        from utils.attendance_manager import AttendanceManager
        manager = AttendanceManager()
        print("✅ Database initialized")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")
    
    # Create sample CSV if it doesn't exist
    if not os.path.exists("database/image_paths_labels.csv"):
        print("Creating sample image paths CSV...")
        
        sample_data = [
            ["ours\\face_0\\class-faces_face0_original.jpg", "face_0"],
            ["ours\\face_1\\class-faces_face1_original.jpg", "face_1"],
            ["ours\\face_2\\class-faces_face2_original.jpg", "face_2"]
        ]
        
        import pandas as pd
        df = pd.DataFrame(sample_data, columns=['image_path', 'label'])
        df.to_csv("database/image_paths_labels.csv", index=False)
        print("✅ Created sample image paths CSV")
    
    return True

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests passed")
            return True
        else:
            print("❌ System tests failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("SETUP COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add face images to the 'input/' folder")
    print("2. Update face names in 'config.py' (FACE_NAME_MAPPING)")
    print("3. Run data processing:")
    print("   python -c \"from utils.data_processor import DatasetProcessor; "
          "DatasetProcessor('input/', 'database/ours/').process_images()\"")
    print("4. Train the model:")
    print("   python train_model.py")
    print("5. Run the attendance system:")
    print("   python attendance_system.py")
    print("\nFor more information, see README.md")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Setup configuration
    if not setup_configuration():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Create sample data
    create_sample_data()
    
    # Check existing data
    check_existing_data()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 