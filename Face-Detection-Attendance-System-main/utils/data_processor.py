"""
Data Processing Utilities for Face Recognition Training
"""

import os
import random
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import IMAGE_PROCESSING_CONFIG, MODEL_CONFIG
from utils.logger import logger

class ImageAugmenter:
    """
    Image augmentation utilities for training data generation
    """
    
    @staticmethod
    def fit_inside_box(image, box_size):
        """Resize image to fit inside a box while maintaining aspect ratio"""
        return ImageOps.contain(image, box_size)
    
    @staticmethod
    def add_noise(image, intensity=5000):
        """Add random noise to an image"""
        image = image.convert("RGB")
        pixels = image.load()
        
        for _ in range(intensity):
            x = random.randint(0, image.width - 1)
            y = random.randint(0, image.height - 1)
            noise_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            pixels[x, y] = noise_color
        
        return image
    
    @staticmethod
    def apply_sepia(image):
        """Apply sepia tone to the image"""
        image = image.convert("RGB")
        sepia = Image.new("RGB", image.size)
        pixels = sepia.load()
        original_pixels = image.load()
        
        for x in range(image.width):
            for y in range(image.height):
                r, g, b = original_pixels[x, y]
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
        
        return sepia
    
    @staticmethod
    def apply_rotation(image, angle):
        """Apply rotation to image"""
        return image.rotate(angle, expand=True)
    
    @staticmethod
    def adjust_contrast(image, factor):
        """Adjust contrast of image"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def convert_to_bw(image):
        """Convert image to black and white"""
        return image.convert("L").convert("RGB")

class DatasetProcessor:
    """
    Dataset processing and preparation utilities
    """
    
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.augmenter = ImageAugmenter()
        
    def process_images(self, rotations=None, contrast_factors=None, box_size=None):
        """
        Process images with augmentation
        
        Args:
            rotations: List of rotation angles
            contrast_factors: List of contrast adjustment factors
            box_size: Size for image resizing
        """
        if rotations is None:
            rotations = IMAGE_PROCESSING_CONFIG["augmentation_rotations"]
        if contrast_factors is None:
            contrast_factors = IMAGE_PROCESSING_CONFIG["contrast_factors"]
        if box_size is None:
            box_size = IMAGE_PROCESSING_CONFIG["box_size"]
        
        try:
            # List all image files
            image_files = [
                f for f in os.listdir(self.input_folder) 
                if f.lower().endswith(tuple(IMAGE_PROCESSING_CONFIG["supported_formats"]))
            ]
            
            if not image_files:
                logger.warning("No images found in input folder")
                return
            
            face_count = 0
            
            for image_file in image_files:
                image_path = os.path.join(self.input_folder, image_file)
                
                # Create output folder
                output_folder = os.path.join(self.output_folder, f"face_{face_count}")
                os.makedirs(output_folder, exist_ok=True)
                
                # Load image
                img = Image.open(image_path)
                img = self.augmenter.fit_inside_box(img, box_size)
                
                count = 0
                
                # Generate rotated images
                for angle in rotations:
                    rotated_img = self.augmenter.apply_rotation(img, angle)
                    rotated_img.save(os.path.join(output_folder, f"class-faces_face{face_count}_dec{count}.jpg"))
                    count += 1
                
                # Generate contrast-adjusted images
                for factor in contrast_factors:
                    contrast_img = self.augmenter.adjust_contrast(img, factor)
                    contrast_img.save(os.path.join(output_folder, f"class-faces_face{face_count}_dec{count}.jpg"))
                    count += 1
                
                # Add noise
                noisy_img = self.augmenter.add_noise(img.copy(), IMAGE_PROCESSING_CONFIG["noise_intensity"])
                noisy_img.save(os.path.join(output_folder, f"class-faces_face{face_count}_dec{count}.jpg"))
                count += 1
                
                # Apply sepia
                sepia_img = self.augmenter.apply_sepia(img.copy())
                sepia_img.save(os.path.join(output_folder, f"class-faces_face{face_count}_dec{count}.jpg"))
                count += 1
                
                # Convert to black and white
                bw_img = self.augmenter.convert_to_bw(img.copy())
                bw_img.save(os.path.join(output_folder, f"class-faces_face{face_count}_dec{count}.jpg"))
                count += 1
                
                # Save original
                img.save(os.path.join(output_folder, f"class-faces_face{face_count}_original.jpg"))
                
                logger.log_system_event("Image processed", f"File: {image_file}, Generated: {count} images")
                face_count += 1
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
    
    def generate_path_labels(self, output_file="path_labels.csv"):
        """
        Generate CSV file with image paths and labels
        
        Args:
            output_file: Output CSV file path
        """
        try:
            data = []
            
            for folder_name in os.listdir(self.output_folder):
                folder_path = os.path.join(self.output_folder, folder_name)
                
                if os.path.isdir(folder_path):
                    for image_name in os.listdir(folder_path):
                        image_path = os.path.join(folder_path, image_name)
                        relative_path = os.path.relpath(image_path, self.output_folder)
                        label = folder_name
                        
                        # Convert to Windows path format
                        relative_path = relative_path.replace(os.sep, '\\')
                        data.append([relative_path, label])
            
            df = pd.DataFrame(data, columns=['image_path', 'label'])
            df.to_csv(output_file, index=False)
            
            logger.log_system_event("Path labels generated", f"File: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate path labels: {e}")

class TripletDataset(Dataset):
    """
    Dataset for triplet training
    """
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self):
        """Get default image transformation"""
        return transforms.Compose([
            transforms.Resize(MODEL_CONFIG["input_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Get paths for anchor, positive, and negative images
        anchor_path = self.dataframe.iloc[idx]['anchor']
        positive_path = self.dataframe.iloc[idx]['positive']
        negative_path = self.dataframe.iloc[idx]['negative']
        
        # Load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative

class PairGenerator:
    """
    Generate training pairs for Siamese network
    """
    
    def __init__(self, path_df, database_path="database/"):
        self.path_df = path_df
        self.database_path = database_path
    
    def generate_random_indices(self, n, exclude, max_photos):
        """Generate random indices excluding a specific index"""
        numbers = [x for x in range(max_photos) if x != exclude]
        n = min(n, len(numbers))
        return random.sample(numbers, n)
    
    def create_triplet_pairs(self):
        """
        Create triplet pairs for training
        
        Returns:
            DataFrame with anchor, positive, and negative pairs
        """
        storage = []
        
        # Get unique labels
        labels = self.path_df['label'].unique()
        
        for label in labels:
            df = self.path_df[self.path_df['label'] == label].to_dict()
            df = df['image_path']
            original = list(df.values())[-1]  # Original image
            
            # Determine max photos for current face
            face_num = int(label.split('_')[1])
            max_photos = 16 if face_num <= 4 else 10
            
            # Generate positive pairs (same person, different images)
            for augmented in zip(list(df.values())[:-1], self.generate_random_indices(13, face_num, max_photos)):
                img_dict = {}
                img_dict['anchor'] = self.database_path + original
                img_dict['positive'] = self.database_path + augmented[0]
                
                # Find negative pair (different person)
                negative_label = f"face_{augmented[1]}"
                negative_df = self.path_df[self.path_df['label'] == negative_label]
                
                if not negative_df.empty:
                    img_dict['negative'] = self.database_path + list(negative_df.to_dict()['image_path'].values())[-1]
                    storage.append(img_dict)
                else:
                    logger.warning(f"No negative pair found for label {negative_label}")
        
        return pd.DataFrame(storage)
    
    def create_validation_pairs(self, num_pairs=100):
        """
        Create validation pairs for testing
        
        Args:
            num_pairs: Number of validation pairs to create
            
        Returns:
            DataFrame with validation pairs
        """
        storage = []
        labels = self.path_df['label'].unique()
        
        for _ in range(num_pairs):
            # Randomly select two different labels
            label1, label2 = random.sample(list(labels), 2)
            
            # Get images from each label
            df1 = self.path_df[self.path_df['label'] == label1]
            df2 = self.path_df[self.path_df['label'] == label2]
            
            if not df1.empty and not df2.empty:
                img1 = df1.iloc[random.randint(0, len(df1)-1)]['image_path']
                img2 = df2.iloc[random.randint(0, len(df2)-1)]['image_path']
                
                storage.append({
                    'image1': self.database_path + img1,
                    'image2': self.database_path + img2,
                    'same_person': False,
                    'label1': label1,
                    'label2': label2
                })
        
        return pd.DataFrame(storage)

class DataLoaderFactory:
    """
    Factory for creating data loaders
    """
    
    @staticmethod
    def create_triplet_loader(dataframe, batch_size=32, shuffle=True, num_workers=0):
        """
        Create triplet data loader
        
        Args:
            dataframe: DataFrame with triplet pairs
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader for triplet training
        """
        dataset = TripletDataset(dataframe)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        )
    
    @staticmethod
    def create_validation_loader(dataframe, batch_size=32, shuffle=False, num_workers=0):
        """
        Create validation data loader
        
        Args:
            dataframe: DataFrame with validation pairs
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader for validation
        """
        # Create a simple dataset for validation
        class ValidationDataset(Dataset):
            def __init__(self, dataframe, transform=None):
                self.dataframe = dataframe
                self.transform = transform or transforms.Compose([
                    transforms.Resize(MODEL_CONFIG["input_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            
            def __len__(self):
                return len(self.dataframe)
            
            def __getitem__(self, idx):
                row = self.dataframe.iloc[idx]
                img1 = Image.open(row['image1']).convert('RGB')
                img2 = Image.open(row['image2']).convert('RGB')
                
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                
                return img1, img2, row['same_person']
        
        dataset = ValidationDataset(dataframe)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers
        ) 