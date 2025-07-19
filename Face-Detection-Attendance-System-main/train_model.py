"""
Main Training Script for Face Recognition Model
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from datetime import datetime

from models.siamese_model import FaceRecognitionModel
from utils.data_processor import PairGenerator, DataLoaderFactory
from utils.logger import logger
from config import MODEL_CONFIG, DATABASE_CONFIG

class ModelTrainer:
    """
    Main training class for the face recognition model
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceRecognitionModel(self.device)
        self.loss_history = []
        self.val_loss_history = []
        
        logger.log_system_event("ModelTrainer initialized", f"Device: {self.device}")
    
    def load_data(self):
        """
        Load and prepare training data
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        try:
            # Load path dataframe
            path_df = pd.read_csv(DATABASE_CONFIG["csv_path"])
            logger.log_system_event("Data loaded", f"Records: {len(path_df)}")
            
            # Create pair generator
            pair_generator = PairGenerator(path_df, DATABASE_CONFIG["db_path"].replace("attendance.db", ""))
            
            # Generate training pairs
            train_pairs = pair_generator.create_triplet_pairs()
            logger.log_system_event("Training pairs generated", f"Pairs: {len(train_pairs)}")
            
            # Generate validation pairs
            val_pairs = pair_generator.create_validation_pairs(num_pairs=50)
            logger.log_system_event("Validation pairs generated", f"Pairs: {len(val_pairs)}")
            
            # Create data loaders
            train_loader = DataLoaderFactory.create_triplet_loader(
                train_pairs, 
                batch_size=MODEL_CONFIG["batch_size"],
                shuffle=True
            )
            
            val_loader = DataLoaderFactory.create_validation_loader(
                val_pairs,
                batch_size=MODEL_CONFIG["batch_size"],
                shuffle=False
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None, None
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            # Move to device
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Training step
            loss = self.model.train_step(anchor, positive, negative)
            running_loss += loss
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss:.4f}'})
        
        avg_loss = running_loss / len(train_loader)
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for anchor, positive, is_same in val_loader:
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                
                # Get embeddings
                anchor_emb = self.model.model(anchor)
                positive_emb = self.model.model(positive)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(anchor_emb, positive_emb)
                
                # Predict same person if similarity > threshold
                predictions = (similarity > MODEL_CONFIG["threshold"]).float()
                
                # Calculate accuracy
                correct_predictions += ((predictions == is_same.float()).sum().item())
                total_predictions += len(is_same)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        self.val_loss_history.append(accuracy)
        
        return accuracy
    
    def train(self, num_epochs=None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
        """
        if num_epochs is None:
            num_epochs = MODEL_CONFIG["num_epochs"]
        
        # Load data
        train_loader, val_loader = self.load_data()
        if train_loader is None:
            logger.error("Failed to load training data")
            return
        
        logger.log_system_event("Training started", f"Epochs: {num_epochs}")
        
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_accuracy = self.validate_epoch(val_loader)
            
            # Log progress
            logger.log_model_operation(
                f"Epoch {epoch+1}/{num_epochs}",
                f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.model.save_model(MODEL_CONFIG["model_path"])
                logger.log_model_operation("Best model saved", f"Accuracy: {val_accuracy:.4f}")
        
        # Save final model
        self.model.save_model(MODEL_CONFIG["model_path"].replace(".pth", "_final.pth"))
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.log_system_event("Training completed")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        try:
            plt.figure(figsize=(12, 5))
            
            # Training loss
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history, label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Validation accuracy
            plt.subplot(1, 2, 2)
            plt.plot(self.val_loss_history, label='Validation Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_curves.png')
            plt.show()
            
            logger.log_system_event("Training curves saved", "training_curves.png")
            
        except Exception as e:
            logger.error(f"Failed to plot training curves: {e}")
    
    def generate_embeddings(self):
        """
        Generate embeddings for known faces
        """
        try:
            # Load path dataframe
            path_df = pd.read_csv(DATABASE_CONFIG["csv_path"])
            
            # Load model
            if not self.model.load_model(MODEL_CONFIG["model_path"]):
                logger.error("Failed to load model for embedding generation")
                return
            
            # Get original images only
            original_paths_df = path_df[path_df['image_path'].str.contains("original", case=False)].copy()
            original_paths_df['image_path'] = original_paths_df['image_path'].apply(
                lambda x: DATABASE_CONFIG["db_path"].replace("attendance.db", "") + x
            )
            original_paths_df['face_no'] = original_paths_df['image_path'].str.extract(r'face_(\d+)').astype(int)
            
            # Generate embeddings
            embeddings = []
            for _, row in original_paths_df.iterrows():
                try:
                    from PIL import Image
                    image = Image.open(row['image_path']).convert('RGB')
                    embedding = self.model.get_face_embedding(image)
                    embeddings.append({
                        'face_no': row['face_no'],
                        'image_path': row['image_path'],
                        'embedding': embedding.tolist()
                    })
                    logger.log_model_operation("Embedding generated", f"Face {row['face_no']}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for {row['image_path']}: {e}")
            
            # Save embeddings
            embeddings_df = pd.DataFrame(embeddings)
            embeddings_df.to_csv(DATABASE_CONFIG["embeddings_path"], index=False)
            
            logger.log_system_event("Embeddings saved", DATABASE_CONFIG["embeddings_path"])
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")

def main():
    """Main function"""
    print("Face Recognition Model Training")
    print("=" * 40)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Generate embeddings
    print("Generating embeddings...")
    trainer.generate_embeddings()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 