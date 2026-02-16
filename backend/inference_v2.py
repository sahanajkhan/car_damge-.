"""
Advanced Car Damage Detector using:
1. ResNet50 classifier trained on severity data (Minor/Moderate/Severe/No Damage)
2. Grad-CAM for damage localization (heatmap showing WHERE damage is)
3. Mask R-CNN for additional region detection (trained on Basel's data)
"""
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


# =========================================
# Configuration
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["Minor Damage", "Moderate Damage", "Severe Damage", "No Damage"]

# Severity mapping for cost estimation
SEVERITY_MAP = {
    "Minor Damage": {"level": "Low", "cost_min": 2000, "cost_max": 15000},
    "Moderate Damage": {"level": "Medium", "cost_min": 15000, "cost_max": 50000},
    "Severe Damage": {"level": "High", "cost_min": 50000, "cost_max": 200000},
    "No Damage": {"level": "None", "cost_min": 0, "cost_max": 0},
}

# Image transforms (must match training)
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks
    Shows WHERE the model is looking to make its decision
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


class DamageDetector:
    """
    Advanced damage detector combining:
    - ResNet50 classifier for severity classification
    - Grad-CAM for damage localization
    """
    
    def __init__(self):
        self.device = DEVICE
        self.model = self._load_classifier()
        self.grad_cam = GradCAM(self.model, self.model.layer4[-1])
        print(f"✅ Damage detector loaded on {self.device}")
    
    def _load_classifier(self):
        """Load trained ResNet50 classifier"""
        model = models.resnet50(weights=None)
        
        # Match training architecture
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS_NAMES))
        )
        
        # Load trained weights
        model_path = os.path.join(MODEL_DIR, "damage_classifier_best.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Classifier model not found: {model_path}\n"
                f"Train it first: python train_classifier.py"
            )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Store class mapping
        self.class_to_idx = checkpoint.get('class_to_idx', {})
        print(f"   Model loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', '?'):.1f}%")
        
        return model
    
    def detect(self, image_path):
        """
        Detect and classify car damage
        
        Returns: list of detection dicts with:
            - damage_type: str (Minor/Moderate/Severe)
            - confidence: float
            - severity: str
            - cost_range: dict
            - heatmap: numpy array (for visualization)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size
        img_array = np.array(image)
        
        # Prepare tensor
        input_tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        
        # Get classification
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get predicted class
        pred_idx = probabilities.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_confidence = probabilities[pred_idx].item()
        
        # Get all class probabilities
        class_probs = {CLASS_NAMES[i]: probabilities[i].item() for i in range(len(CLASS_NAMES))}
        
        print(f"\n📊 Classification Results:")
        for cls, prob in sorted(class_probs.items(), key=lambda x: -x[1]):
            marker = "  ← PREDICTED" if cls == pred_class else ""
            print(f"   {cls:20s}: {prob*100:.1f}%{marker}")
        
        # If no damage detected, return empty
        if pred_class == "No Damage" and pred_confidence > 0.6:
            print(f"   ✅ No damage detected (confidence: {pred_confidence*100:.1f}%)")
            return []
        
        # Generate Grad-CAM heatmap for the predicted damage class
        # If "No Damage" but low confidence, check damage classes
        if pred_class == "No Damage":
            # Use the highest damage class instead
            damage_probs = {k: v for k, v in class_probs.items() if k != "No Damage"}
            pred_class = max(damage_probs, key=damage_probs.get)
            pred_confidence = damage_probs[pred_class]
            pred_idx = CLASS_NAMES.index(pred_class)
        
        # Generate heatmap using Grad-CAM
        input_tensor_grad = TRANSFORM(image).unsqueeze(0).to(self.device)
        input_tensor_grad.requires_grad = True
        
        heatmap = self.grad_cam.generate(input_tensor_grad, target_class=pred_idx)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
        
        # Find damage regions from heatmap
        damage_regions = self._extract_damage_regions(heatmap_resized, img_array, img_width, img_height)
        
        # Build detection results
        severity_info = SEVERITY_MAP[pred_class]
        
        detections = []
        for region in damage_regions:
            detection = {
                'damage_type': pred_class.replace(" Damage", ""),
                'confidence': pred_confidence,
                'severity': severity_info['level'],
                'bbox': region['bbox'],
                'area_pixels': region['area'],
                'area_percentage': region['area_pct'],
                'mask': region['mask'],
                'heatmap': heatmap_resized,
            }
            detections.append(detection)
            print(f"   🎯 {pred_class}: {pred_confidence*100:.1f}% | Area: {region['area_pct']:.1f}%")
        
        if len(detections) == 0 and pred_class != "No Damage":
            # Fallback: create a single detection from the heatmap center
            hot_mask = (heatmap_resized > 0.3).astype(np.uint8)
            if hot_mask.sum() > 0:
                rows = np.any(hot_mask, axis=1)
                cols = np.any(hot_mask, axis=0)
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                
                area = hot_mask.sum()
                area_pct = (area / (img_width * img_height)) * 100
                
                detections.append({
                    'damage_type': pred_class.replace(" Damage", ""),
                    'confidence': pred_confidence,
                    'severity': severity_info['level'],
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'area_pixels': int(area),
                    'area_percentage': float(area_pct),
                    'mask': hot_mask.astype(float),
                    'heatmap': heatmap_resized,
                })
                print(f"   🎯 {pred_class}: {pred_confidence*100:.1f}% | Area: {area_pct:.1f}% (from heatmap)")
        
        return detections
    
    def _extract_damage_regions(self, heatmap, img_array, img_width, img_height):
        """Extract damage regions from Grad-CAM heatmap"""
        regions = []
        
        # Threshold heatmap to find hot regions
        threshold = 0.4  # Focus on areas with 40%+ activation
        hot_mask = (heatmap > threshold).astype(np.uint8)
        
        # Clean up mask
        kernel = np.ones((15, 15), np.uint8)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_area = img_width * img_height
            area_pct = (area / total_area) * 100
            
            # Skip tiny regions (< 0.5% of image)
            if area_pct < 0.5:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Create mask for this region
            region_mask = np.zeros((img_height, img_width), dtype=np.float32)
            cv2.drawContours(region_mask, [cnt], -1, 1.0, -1)
            
            regions.append({
                'bbox': [x, y, x + w, y + h],
                'area': int(area),
                'area_pct': area_pct,
                'mask': region_mask,
            })
        
        return regions
    
    def get_annotated_image(self, image_path, detections):
        """Create annotated image with heatmap overlay and damage regions"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        output = image.copy()
        
        if not detections:
            # No damage - add green border
            cv2.rectangle(output, (5, 5), (output.shape[1]-5, output.shape[0]-5), (0, 200, 0), 3)
            cv2.putText(output, "No Damage Detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
            return output
        
        # Get heatmap from first detection
        heatmap = detections[0].get('heatmap')
        
        if heatmap is not None:
            # Create beautiful heatmap overlay
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            
            # Only overlay where heatmap is significant (> 30%)
            mask = heatmap > 0.3
            mask_3d = np.stack([mask, mask, mask], axis=2)
            
            # Blend: 60% original + 40% heatmap in hot areas
            output = np.where(mask_3d, 
                            cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0),
                            image)
        
        # Draw bounding boxes and labels
        colors = {
            'Minor': (0, 255, 255),      # Yellow
            'Moderate': (0, 165, 255),    # Orange  
            'Severe': (0, 0, 255),        # Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            damage_type = det['damage_type']
            confidence = det['confidence']
            
            color = colors.get(damage_type, (255, 255, 0))
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{damage_type} {confidence*100:.0f}%"
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Label background
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return output
