"""
Siamese Neural Network Model for Face Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from config import MODEL_CONFIG
from utils.logger import logger

class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for face recognition using triplet loss
    """
    
    def __init__(self, embedding_size=512):
        super(SiameseNetwork, self).__init__()
        
        # Base model - InceptionResnetV1 pretrained on VGGFace2
        self.base_model = InceptionResnetV1(pretrained='vggface2')
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Additional layers for fine-tuning
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.embedding_layer = nn.Linear(512, embedding_size)
        
        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(embedding_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        logger.log_model_operation("Model initialized", f"Embedding size: {embedding_size}")
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_size)
        """
        # Base model forward pass
        x = self.base_model(x)
        
        # FC1 -> ReLU -> BatchNorm -> Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # FC2 -> ReLU -> BatchNorm -> Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Embedding layer -> ReLU -> BatchNorm -> Dropout
        x = self.embedding_layer(x)
        x = self.relu(x)
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        return x
    
    def get_embedding(self, x):
        """
        Get face embedding without training
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized embedding
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x)
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

class TripletLoss(nn.Module):
    """
    Triplet Loss for training the Siamese network
    """
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Triplet loss value
        """
        return self.triplet_loss(anchor, positive, negative)

class FaceRecognitionModel:
    """
    High-level interface for face recognition operations
    """
    
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiameseNetwork(MODEL_CONFIG["embedding_size"]).to(self.device)
        self.criterion = TripletLoss(MODEL_CONFIG["margin"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=MODEL_CONFIG["learning_rate"])
        
        logger.log_model_operation("FaceRecognitionModel initialized", f"Device: {self.device}")
    
    def load_model(self, model_path):
        """
        Load trained model weights
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.log_model_operation("Model loaded", f"Path: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def save_model(self, model_path):
        """
        Save model weights
        
        Args:
            model_path: Path to save the model
        """
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.log_model_operation("Model saved", f"Path: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def train_step(self, anchor, positive, negative):
        """
        Perform one training step
        
        Args:
            anchor: Anchor images
            positive: Positive images
            negative: Negative images
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get embeddings
        anchor_emb = self.model(anchor)
        positive_emb = self.model(positive)
        negative_emb = self.model(negative)
        
        # Compute loss
        loss = self.criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_face_embedding(self, face_image):
        """
        Get embedding for a single face image
        
        Args:
            face_image: PIL Image or tensor
            
        Returns:
            Face embedding
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(face_image, torch.Tensor):
                # Convert PIL image to tensor
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(MODEL_CONFIG["input_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                face_image = transform(face_image).unsqueeze(0)
            
            face_image = face_image.to(self.device)
            embedding = self.model(face_image)
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding.squeeze()
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score
        """
        return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    
    def find_most_similar_face(self, query_embedding, known_embeddings, threshold=0.6):
        """
        Find the most similar face from known embeddings
        
        Args:
            query_embedding: Query face embedding
            known_embeddings: Dictionary of known embeddings
            threshold: Similarity threshold
            
        Returns:
            Tuple of (face_id, similarity_score) or (None, max_similarity)
        """
        max_similarity = -1
        most_similar_face = None
        
        for face_id, known_embedding in known_embeddings.items():
            if isinstance(known_embedding, list):
                known_embedding = torch.tensor(known_embedding)
            
            similarity = self.compute_similarity(query_embedding, known_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_face = face_id
        
        if max_similarity > threshold:
            return most_similar_face, max_similarity
        else:
            return None, max_similarity 

    def extract_embedding(self, image):
        """
        Extracts a face embedding from an input image.
        Args:
            image: numpy array (BGR or RGB) or PIL Image
        Returns:
            embedding: numpy array or None if failed
        """
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = Image.fromarray(image[..., ::-1])  # BGR to RGB
            else:
                image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a numpy array or PIL Image")
        # Preprocess (resize, normalize, to tensor)
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(self.device)
        # Get embedding
        self.model.eval()
        with torch.no_grad():
            embedding = self.model(input_tensor)
        return embedding.cpu().numpy().flatten() 