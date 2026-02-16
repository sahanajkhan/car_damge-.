"""
Training and Model Configuration
"""
import torch

class Config:
    """Mask R-CNN Configuration for Car Damage Detection"""
    
    # Model Architecture
    NUM_CLASSES = 1 + 1  # Background + Damage
    
    # Training Parameters
    BATCH_SIZE = 2  # Small batch for memory efficiency
    NUM_EPOCHS = 30  # More training for better accuracy
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Detection Parameters
    MIN_CONFIDENCE = 0.22  # Lower threshold for small dataset (49 images)
    NMS_THRESHOLD = 0.3
    
    # Image Parameters
    MIN_SIZE = 800
    MAX_SIZE = 1333
    
    # Paths
    DATASET_PATH = "D:\\hero\\Automated-Car-Damage-Detection\\dataset"
    MODEL_SAVE_PATH = "D:\\hero\\autodamage-ai-v2\\models"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Class Names
    CLASS_NAMES = ['background', 'damage']
    
    # Damage Categories (for classification)
    DAMAGE_TYPES = {
        'Scratch': {'min_area': 0.001, 'max_area': 0.03, 'aspect_ratio_range': (2.5, 20)},
        'Dent': {'min_area': 0.01, 'max_area': 0.08, 'aspect_ratio_range': (0.5, 2.5)},
        'Shatter': {'min_area': 0.02, 'max_area': 0.15, 'aspect_ratio_range': (0.3, 3.0)},
        'Dislocation': {'min_area': 0.05, 'max_area': 0.25, 'aspect_ratio_range': (0.5, 3.0)}
    }

config = Config()
