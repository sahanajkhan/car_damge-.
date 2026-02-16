"""
Train a ResNet50 damage severity classifier on the Car-Damage-Detection dataset
Classes: Minor, Moderate, Severe, No Damage
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import json

# =========================================
# Configuration
# =========================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Car-Damage-Detection-data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 4  # minor, moderate, severe, no_damage
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names matching folder order
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage", "No Damage"]

# =========================================
# Data Transforms
# =========================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    """Build ResNet50 with custom classifier head"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers (keep fine-tuning last layers)
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    # Replace final classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    
    return model


def train():
    print(f"\n{'='*60}")
    print(f"🚗 Car Damage Severity Classifier Training")
    print(f"{'='*60}")
    print(f"📱 Device: {DEVICE}")
    print(f"📁 Dataset: {DATA_DIR}")
    print(f"🏷️  Classes: {CLASS_NAMES}")
    print(f"{'='*60}\n")
    
    # Load datasets
    train_dir = os.path.join(DATA_DIR, "1. Training")
    val_dir = os.path.join(DATA_DIR, "2. Validation")
    test_dir = os.path.join(DATA_DIR, "3. Testing")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    
    # Print class mapping
    print(f"📋 Class mapping: {train_dataset.class_to_idx}")
    print(f"📊 Training:   {len(train_dataset)} images")
    print(f"📊 Validation: {len(val_dataset)} images")
    print(f"📊 Testing:    {len(test_dataset)} images\n")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    # Build model
    model = build_model().to(DEVICE)
    
    # Class weights for imbalanced data
    class_counts = [365, 427, 414, 768]  # minor, moderate, severe, no_damage
    total = sum(class_counts)
    class_weights = torch.tensor([total / (NUM_CLASSES * c) for c in class_counts]).to(DEVICE)
    print(f"⚖️  Class weights: {class_weights.tolist()}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", 
                    bar_format='{l_bar}{bar:30}{r_bar}')
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        scheduler.step()
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_DIR, "damage_classifier_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'class_to_idx': train_dataset.class_to_idx,
            }, save_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.1f}%)")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(MODEL_DIR, f"classifier_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, save_path)
            print(f"   💾 Checkpoint saved: epoch_{epoch+1}.pth")
    
    # --- Final Test ---
    print(f"\n{'='*60}")
    print(f"🧪 Testing on {len(test_dataset)} unseen images...")
    print(f"{'='*60}")
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(MODEL_DIR, "damage_classifier_best.pth"), 
                                  map_location=DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Overall Accuracy: {test_acc:.1f}%")
    print(f"\n   Per-class accuracy:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"   {CLASS_NAMES[i]:20s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    print(f"\n✨ Training complete!")
    print(f"📦 Best model: {os.path.join(MODEL_DIR, 'damage_classifier_best.pth')}")
    print(f"📊 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"📊 Test Accuracy: {test_acc:.1f}%")


if __name__ == "__main__":
    train()
