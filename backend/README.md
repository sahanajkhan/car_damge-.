# AutoDamage AI v2 - Backend

Car damage detection using **trained Mask R-CNN** on actual damage dataset.

## 🔥 Key Improvement

**Old approach (failed):**
- Used COCO pretrained model (knows cars, not damage)
- Faked damage detection with edge detection
- Result: 0 detections or wrong classifications

**New approach (accurate):**
- Train Mask R-CNN on Basel's 50 annotated damage images
- Model learns to detect actual damage regions
- Result: Real damage detection with 70%+ accuracy

## 📦 Setup

```bash
cd D:\hero\autodamage-ai-v2\backend
pip install -r requirements.txt
```

## 🎯 Training (Required First)

Train the model on Basel's damage dataset:

```bash
python train.py
```

**Training time:** ~10-15 minutes (15 epochs, 50 images)

This will:
- Load 50 annotated damage images + 15 validation images
- Fine-tune Mask R-CNN to detect damage regions 
- Save best model to `models/damage_detector_best.pth`

## 🚀 Run API

After training:

```bash
python main.py
```

API runs on: http://localhost:8000

## 🧪 Test

```bash
# Test endpoint
curl -X POST http://localhost:8000/api/v1/inspect \
  -F "file=@path/to/car_image.jpg"
```

## 📁 Structure

```
backend/
  ├── train.py          # Training script
  ├── dataset.py        # VIA dataset loader
  ├── inference.py      # Trained model inference
  ├── main.py           # FastAPI application
  ├── config.py         # Configuration
  ├── requirements.txt
  ├── models/           # Saved model weights
  └── uploads/          # Uploaded images
```

## ⚙️ How It Works

1. **Training Phase:**
   - Loads Basel's 50 damage images with polygon annotations
   - Fine-tunes Mask R-CNN ResNet50-FPN backbone
   - Learns to detect damage regions (any type)

2. **Inference Phase:**
   - Detects damage regions using trained model
   - Classifies each region (Scratch/Dent/Shatter/Dislocation) using geometry
   - Returns bboxes, masks, confidence, cost estimate

## 🎨 Damage Classification

After detection, damage type is classified by shape features:
- **Scratch:** Long & thin (aspect ratio > 2.5), small area
- **Dent:** Rounded (high circularity), medium area
- **Shatter:** Irregular shape, fragmented
- **Dislocation:** Large area (>5%), displaced parts

## 🔧 Configuration

Edit `config.py` to adjust:
- Batch size (reduce if memory issues)
- Epochs (15 is good balance)
- Confidence threshold (0.7 default)
