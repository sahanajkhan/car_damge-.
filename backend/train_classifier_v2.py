"""
Train ResNet50 damage classifier - V2 (High Accuracy)
Fixes:
  1. Unfreeze ALL layers (model was underfitting)
  2. Lower learning rate with warmup
  3. Stronger augmentation + MixUp
  4. Label smoothing
  5. More epochs (40)
  6. Lower dropout (was causing underfitting)
"""
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets, models
from tqdm import tqdm
import numpy as np

# =========================================
# Configuration
# =========================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "Car-Damage-Detection-data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.0003  # Lower LR for full fine-tuning
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage", "No Damage"]
MIXUP_ALPHA = 0.2  # MixUp augmentation strength

# =========================================
# Data Transforms - Stronger augmentation
# =========================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def mixup_data(x, y, alpha=0.2):
    """MixUp augmentation - blends pairs of images and labels"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_model():
    """Build ResNet50 - unfreeze ALL layers for full fine-tuning"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # UNFREEZE ALL LAYERS (previous version froze too many, causing underfitting)
    for param in model.parameters():
        param.requires_grad = True
    
    # Custom classifier head (reduced dropout - was causing underfitting)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.15),
        nn.Linear(256, NUM_CLASSES)
    )
    
    return model


def get_weighted_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train():
    print(f"\n{'='*60}")
    print(f"🚗 Car Damage Classifier V2 (High Accuracy)")
    print(f"{'='*60}")
    print(f"📱 Device: {DEVICE}")
    print(f"📁 Dataset: {DATA_DIR}")
    print(f"🏷️  Classes: {CLASS_NAMES}")
    print(f"🔧 Full fine-tuning: ALL layers unfrozen")
    print(f"🔧 MixUp alpha: {MIXUP_ALPHA}")
    print(f"🔧 Label smoothing: 0.1")
    print(f"{'='*60}\n")
    
    # Load datasets
    train_dir = os.path.join(DATA_DIR, "1. Training")
    val_dir = os.path.join(DATA_DIR, "2. Validation")
    test_dir = os.path.join(DATA_DIR, "3. Testing")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    
    print(f"📋 Class mapping: {train_dataset.class_to_idx}")
    print(f"📊 Training:   {len(train_dataset)} images")
    print(f"📊 Validation: {len(val_dataset)} images")
    print(f"📊 Testing:    {len(test_dataset)} images\n")
    
    # Weighted sampler for balanced batches
    sampler = get_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    
    # Build model
    model = build_model().to(DEVICE)
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📐 Parameters: {trainable:,} trainable / {total:,} total")
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer - different LR for backbone vs classifier head
    backbone_params = [p for n, p in model.named_parameters() 
                       if 'fc' not in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() 
                   if 'fc' in n and p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained layers
        {'params': head_params, 'lr': LEARNING_RATE},            # Higher LR for new head
    ], weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE],
        epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_acc = 0.0
    patience = 0
    max_patience = 10
    
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
            
            # Apply MixUp
            if random.random() < 0.5:  # 50% chance of MixUp
                images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                # Track accuracy using primary label
                _, predicted = outputs.max(1)
                train_correct += (lam * predicted.eq(labels_a).sum().item() + 
                                 (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        train_loss /= train_total
        train_acc = 100. * train_correct / train_total
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        
        # Per-class val accuracy
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"   {CLASS_NAMES[i]:20s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
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
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(MODEL_DIR, f"classifier_v2_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, save_path)
            print(f"   💾 Checkpoint saved")
    
    # === Final Test ===
    print(f"\n{'='*60}")
    print(f"🧪 Testing on {len(test_dataset)} unseen images...")
    print(f"{'='*60}")
    
    best_checkpoint = torch.load(os.path.join(MODEL_DIR, "damage_classifier_best.pth"),
                                  map_location=DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    # Confusion matrix
    confusion = [[0]*NUM_CLASSES for _ in range(NUM_CLASSES)]
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                confusion[label][pred] += 1
                if pred == label:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n🎯 FINAL TEST RESULTS:")
    print(f"   Overall Accuracy: {test_acc:.1f}%")
    print(f"\n   Per-class accuracy:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"   {CLASS_NAMES[i]:20s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    print(f"\n   Confusion Matrix:")
    print(f"   {'':20s} | {'Minor':>8s} | {'Moderate':>8s} | {'Severe':>8s} | {'No Dmg':>8s}")
    print(f"   {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for i in range(NUM_CLASSES):
        row = " | ".join(f"{confusion[i][j]:>8d}" for j in range(NUM_CLASSES))
        print(f"   {CLASS_NAMES[i]:20s} | {row}")
    
    print(f"\n✨ Training complete!")
    print(f"📦 Best model saved to: {os.path.join(MODEL_DIR, 'damage_classifier_best.pth')}")
    print(f"📊 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"📊 Test Accuracy: {test_acc:.1f}%")


if __name__ == "__main__":
    train()
