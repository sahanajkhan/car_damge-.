#  AI-Driven Vehicle Damage Detection & Intelligent Repair Assessment

> Developed for Hero VIDA Campus Challenge 2026

An AI-powered system that automatically detects vehicle damage, classifies severity, and estimates repair costs using deep learning and computer vision.

---

##  Overview

Manual vehicle inspection is time-consuming and error-prone. This project provides an automated AI solution that analyzes vehicle images and generates:

- ✅ Damage detection
- ✅ Severity classification (Minor, Moderate, Severe, No Damage)
- ✅ Repair urgency prediction
- ✅ Estimated repair cost
- ✅ Explainable AI (Bounding Box & Heatmap)

This system can be used in:

- Insurance claim automation  
- Vehicle service centers  
- Automated inspection systems  

---

## Model Development

We used pretrained deep learning models as baseline:

- MobileNetV2  
- ResNet50  
- EfficientNet-B0  

We improved their accuracy by:

- Fine-tuning on a custom vehicle damage dataset  
- Applying data augmentation  
- Optimizing severity classification  
- Testing on real-world images  

This transfer learning approach helps the model learn damage-specific features efficiently.

---

##  Model Performance

**Overall Test Accuracy: 68.7%**

### Class-wise Accuracy:

| Class | Accuracy |
|------|----------|
| No Damage | 91.1% |
| Severe Damage | 79.6% |
| Minor Damage | 71.1% |
| Moderate Damage | 21.1% |

Tested on **227 real-world vehicle images**

---

##  Tech Stack

**AI / ML**

- Python
- PyTorch
- Torchvision
- OpenCV

**Frontend**

- HTML
- CSS
- JavaScript

**Backend**

- Node.js / Flask

---

##  System Architecture

```
Vehicle Image
     ↓
Pretrained Model (EfficientNet-B0)
     ↓
Fine-Tuning
     ↓
Damage Classification
     ↓
Severity Prediction
     ↓
Repair Cost Estimation
     ↓
Result Display
```

---

##  Project Structure

```
vehicle-damage-detection/

├── dataset/
├── models/
├── train.py
├── test.py
├── predict.py
├── frontend/
├── backend/
└── README.md
```

---

##  Installation

Clone repository:

```bash
git clone https://github.com/yourusername/vehicle-damage-detection.git
```

Go to project folder:

```bash
cd vehicle-damage-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Run Project

### Run prediction

```bash
python predict.py
```

### Train model

```bash
python train.py
```

---

##  Example Output

**Input:** Vehicle Image  

**Output:**

- Damage: Severe Dent  
- Confidence: 94%  
- Repair Cost: ₹6000  
- Urgency: High  

---

## 🌟 Key Features

- Automated vehicle damage detection  
- Severity classification  
- Repair cost estimation  
- Explainable AI output  
- Custom trained deep learning model  

---

##  Future Improvements

- Improve accuracy to 90%+
- Deploy as web application
- Mobile app integration
- Real-time detection


