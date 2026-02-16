"""
Train Mask R-CNN on Car Damage Dataset
Using PyTorch and torchvision (Python 3.12 compatible)
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import numpy as np

from dataset import CarDamageDataset, collate_fn
from config import config


def get_model(num_classes):
    """
    Create Mask R-CNN model with ResNet50-FPN backbone
    """
    # Load pretrained Mask R-CNN (COCO weights)
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in progress_bar:
        # Move to device
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track loss
        epoch_loss += losses.item()
        progress_bar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, device):
    """Validation step"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Validating'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Validation loss (model in eval mode returns predictions, not losses)
            # So we temporarily switch to train mode
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            model.eval()
    
    return val_loss / len(data_loader)


def main():
    print("🚀 Starting Mask R-CNN Training for Car Damage Detection\n")
    print(f"📊 Configuration:")
    print(f"   - Device: {config.DEVICE}")
    print(f"   - Epochs: {config.NUM_EPOCHS}")
    print(f"   - Batch Size: {config.BATCH_SIZE}")
    print(f"   - Learning Rate: {config.LEARNING_RATE}")
    print(f"   - Dataset: {config.DATASET_PATH}\n")
    
    # Create datasets
    train_dataset = CarDamageDataset(
        root_dir=config.DATASET_PATH,
        subset='train'
    )
    
    val_dataset = CarDamageDataset(
        root_dir=config.DATASET_PATH,
        subset='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    print("🔧 Building Mask R-CNN model...")
    model = get_model(num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )
    
    # Training loop
    print("\n🎯 Starting training...\n")
    best_val_loss = float('inf')
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, config.DEVICE, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, config.DEVICE)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"\n📈 Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"   - Train Loss: {train_loss:.4f}")
        print(f"   - Val Loss: {val_loss:.4f}")
        print(f"   - Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.MODEL_SAVE_PATH, 'damage_detector_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_path)
            print(f"✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: epoch_{epoch}.pth")
    
    # Save final model
    final_path = os.path.join(config.MODEL_SAVE_PATH, 'damage_detector_final.pth')
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    print(f"\n✨ Training complete!")
    print(f"📦 Best model saved to: {model_path}")
    print(f"📦 Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
