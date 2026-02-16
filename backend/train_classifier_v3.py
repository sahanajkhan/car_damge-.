"""
Train EfficientNet-V2-S damage classifier - V3 (Production)
Key changes from V2:
  1. EfficientNet-V2-S backbone (much stronger than ResNet50)
  2. 384x384 input (captures fine damage details)
  3. NO MixUp (was hurting - blending severity levels is contradictory)
  4. Focal Loss (focuses on hard examples like Minor vs Moderate)
  5. Test Time Augmentation (TTA) for evaluation
  6. Cosine annealing with warm restarts
  7. Gradient accumulation for effective batch size 32
  8. Lighter augmentation (less aggressive, more realistic)
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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

IMG_SIZE = 384          # EfficientNet-V2-S native resolution
BATCH_SIZE = 8          # Smaller batch (larger images), use grad accumulation
ACCUM_STEPS = 4         # Effective batch size = 8 * 4 = 32
NUM_EPOCHS = 50
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage", "No Damage"]

# =========================================
# Focal Loss - focuses on hard examples
# =========================================
class FocalLoss(nn.Module):
    """Focal Loss to focus training on hard-to-classify examples"""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =========================================
# Data Transforms - Appropriate for damage detection
# =========================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # 416x416
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA transforms for test-time augmentation
tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slightly different crop
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Slightly zoomed
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


def build_model():
    """Build EfficientNet-V2-S with custom classifier head"""
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    # Unfreeze all layers for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Custom classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.15),
        nn.Linear(512, NUM_CLASSES)
    )
    
    return model


def get_weighted_sampler(dataset):
    """Create weighted sampler to handle class imbalance"""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    print(f"\n📊 Class distribution: {dict(zip(CLASS_NAMES, class_counts))}")
    
    # Stronger weights for underrepresented classes
    class_weights = 1.0 / (class_counts ** 0.75)  # Smoothed inverse frequency
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"⚖️  Sampling weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in class_weights]))}")
    
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def evaluate_with_tta(model, test_dir, class_names):
    """Evaluate using Test Time Augmentation - average predictions across augmentations"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Load test dataset with basic transform to get labels
    basic_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    
    print(f"\n🔄 Running TTA with {len(tta_transforms)} augmentations...")
    
    for idx in tqdm(range(len(basic_dataset)), desc="TTA Evaluation"):
        img_path, label = basic_dataset.samples[idx]
        all_labels.append(label)
        
        # Load raw image
        from PIL import Image
        raw_img = Image.open(img_path).convert('RGB')
        
        # Average predictions across all TTA transforms
        avg_probs = torch.zeros(NUM_CLASSES).to(DEVICE)
        for tta_t in tta_transforms:
            img_tensor = tta_t(raw_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                avg_probs += probs.squeeze(0)
        
        avg_probs /= len(tta_transforms)
        pred = avg_probs.argmax().item()
        all_preds.append(pred)
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_acc = (all_preds == all_labels).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TEST RESULTS (with TTA):")
    print(f"   Overall Accuracy: {overall_acc:.1f}%\n")
    print(f"   Per-class accuracy:")
    
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean() * 100
            correct = (all_preds[mask] == i).sum()
            total = mask.sum()
            print(f"   {name:20s}: {acc:.1f}% ({correct}/{total})")
    
    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    header = "".join([f" | {name:>8s}" for name in ["Minor", "Moderate", "Severe", "No Dmg"]])
    print(f"   {'':20s}{header}")
    print(f"   {'-'*21}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*9}")
    
    for i, name in enumerate(class_names):
        row = f"   {name:20s}"
        for j in range(NUM_CLASSES):
            count = ((all_labels == i) & (all_preds == j)).sum()
            row += f" | {count:>8d}"
        print(row)
    
    return overall_acc


def train():
    print(f"\n{'='*60}")
    print(f"🚗 Car Damage Classifier V3 (EfficientNet-V2-S)")
    print(f"{'='*60}")
    print(f"📱 Device: {DEVICE}")
    print(f"📐 Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📦 Effective batch size: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"📁 Dataset: {DATA_DIR}")
    print(f"🏷️  Classes: {CLASS_NAMES}")
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "1. Training"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "2. Validation"),
        transform=val_transforms
    )
    
    print(f"\n📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Class mapping: {train_dataset.class_to_idx}")
    
    # Weighted sampler for class imbalance
    sampler = get_weighted_sampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Build model
    model = build_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: EfficientNet-V2-S")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Compute class weights for Focal Loss
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    # Inverse frequency weights, normalized
    alpha = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    alpha = alpha / alpha.sum() * NUM_CLASSES
    alpha = alpha.to(DEVICE)
    print(f"   Focal Loss alpha: {alpha.cpu().numpy()}")
    
    # Focal Loss with label smoothing
    criterion = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.05)
    
    # Discriminative learning rates
    # Backbone gets lower LR, classifier head gets higher LR
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': head_params, 'lr': 5e-4, 'weight_decay': 0.001},
    ])
    
    # Cosine annealing with warm restarts (restart every 15 epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    # Warmup for first 3 epochs
    warmup_epochs = 3
    warmup_factor = 0.1
    
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    print(f"   Warmup: {warmup_epochs} epochs")
    print(f"   Early stopping patience: {patience}")
    print(f"   Gradient accumulation: {ACCUM_STEPS} steps")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Warmup LR scaling
        if epoch < warmup_epochs:
            warmup_lr_scale = warmup_factor + (1.0 - warmup_factor) * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * warmup_lr_scale / (warmup_factor + (1.0 - warmup_factor) * (max(0, epoch-1) / warmup_epochs)) if epoch > 0 else pg['lr'] * warmup_factor
        
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                     desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels) / ACCUM_STEPS
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUM_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}", 
                           acc=f"{100.*correct/total:.1f}%")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Update scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Validation
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
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Print epoch results
        print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        
        for i, name in enumerate(CLASS_NAMES):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"   {name:20s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model_path = os.path.join(MODEL_DIR, "damage_classifier_best.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'val_accuracy': val_acc,
                'epoch': epoch + 1,
                'architecture': 'efficientnet_v2_s',
                'img_size': IMG_SIZE,
            }, model_path)
            print(f"   💾 Checkpoint saved (val_acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    # Load best model for testing
    print(f"\n{'='*60}")
    print(f"🧪 Testing on unseen images...")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(MODEL_DIR, "damage_classifier_best.pth"), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Standard evaluation (no TTA)
    test_dir = os.path.join(DATA_DIR, "3. Testing")
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing (standard)"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"\n🎯 TEST RESULTS (standard):")
    print(f"   Overall Accuracy: {test_acc:.1f}%\n")
    print(f"   Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"   {name:20s}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    # Confusion matrix
    print(f"\n   Confusion Matrix:")
    header = "".join([f" | {name:>8s}" for name in ["Minor", "Moderate", "Severe", "No Dmg"]])
    print(f"   {'':20s}{header}")
    print(f"   {'-'*21}+{'-'*10}+{'-'*10}+{'-'*10}+{'-'*9}")
    for i, name in enumerate(CLASS_NAMES):
        row = f"   {name:20s}"
        for j in range(NUM_CLASSES):
            count = ((all_labels == i) & (all_preds == j)).sum()
            row += f" | {count:>8d}"
        print(row)
    
    # TTA evaluation
    print(f"\n{'='*60}")
    tta_acc = evaluate_with_tta(model, test_dir, CLASS_NAMES)
    
    print(f"\n{'='*60}")
    print(f"✨ Training complete!")
    print(f"📦 Best model saved to: {os.path.join(MODEL_DIR, 'damage_classifier_best.pth')}")
    print(f"📊 Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"📊 Test Accuracy (standard): {test_acc:.1f}%")
    print(f"📊 Test Accuracy (TTA): {tta_acc:.1f}%")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # Faster training
    
    train()
