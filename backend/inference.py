"""
Inference Engine - Car Damage Detection with Trained Mask R-CNN
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2

from config import config


class DamageDetector:
    """
    Trained Mask R-CNN inference for car damage detection
    """
    
    def __init__(self, model_path=None):
        """
        Load trained model
        """
        self.device = config.DEVICE
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✅ Damage detector loaded on {self.device}")
    
    def _load_model(self, model_path=None):
        """Load trained Mask R-CNN model"""
        # Create model architecture
        model = maskrcnn_resnet50_fpn(weights=None)
        
        # Replace predictors
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, config.NUM_CLASSES)
        
        # Load weights
        if model_path is None:
            model_path = os.path.join(config.MODEL_SAVE_PATH, 'damage_detector_best.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nPlease train the model first: python train.py")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def detect(self, image_path):
        """
        Detect damage in image
        
        Returns:
            list of detections with damage type classification
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        # Convert to tensor
        image_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # DEBUG: Show ALL raw model outputs
        all_scores = predictions['scores'].cpu().numpy()
        all_labels = predictions['labels'].cpu().numpy()
        print(f"\n🔍 RAW MODEL OUTPUT: {len(all_scores)} total detections")
        for i in range(min(10, len(all_scores))):  # Show top 10
            print(f"   Detection {i+1}: confidence={all_scores[i]:.4f}, label={all_labels[i]}")
        
        # Filter by confidence (including medium-confidence for clustering)
        high_conf_indices = predictions['scores'] > config.MIN_CONFIDENCE
        medium_conf_indices = predictions['scores'] > (config.MIN_CONFIDENCE * 0.6)  # 60% of threshold
        
        boxes = predictions['boxes'][high_conf_indices].cpu().numpy()
        scores = predictions['scores'][high_conf_indices].cpu().numpy()
        masks = predictions['masks'][high_conf_indices].cpu().numpy()
        labels = predictions['labels'][high_conf_indices].cpu().numpy()
        
        # Also get medium confidence detections for potential merging
        medium_boxes = predictions['boxes'][medium_conf_indices].cpu().numpy()
        medium_scores = predictions['scores'][medium_conf_indices].cpu().numpy()
        medium_masks = predictions['masks'][medium_conf_indices].cpu().numpy()
        
        print(f"✓ Found {len(boxes)} high-confidence detections (>{config.MIN_CONFIDENCE})")
        print(f"✓ Found {len(medium_boxes)} medium-confidence detections (>{config.MIN_CONFIDENCE * 0.6})")
        print(f"   (Rejected {len(all_scores) - len(medium_boxes)} low-confidence detections)\n")
        
        # Load image for visual analysis
        img_array = np.array(image)
        
        # First pass: expand and merge nearby detections
        if len(boxes) > 0:
            boxes, scores, masks = self._merge_nearby_detections(
                boxes, scores, masks, medium_boxes, medium_scores, medium_masks, img_width, img_height
            )
        
        # Process each detection with advanced filtering
        detections = []
        for i in range(len(boxes)):
            bbox = boxes[i]
            mask = masks[i, 0]  # Remove channel dimension
            score = scores[i]
            
            # Calculate area first for early filtering
            area_pixels = np.sum(mask > 0.5)
            area_percentage = (area_pixels / (img_width * img_height)) * 100
            
            # FILTER 1: Ignore very small regions (likely logos, reflections, artifacts)
            # Relaxed minimum to catch more damage
            if area_percentage < 0.05:
                print(f"   ⊘ Region {i+1}: Ignored (too small: {area_percentage:.2f}%)")
                continue
            
            # FILTER 2: Check if region looks like actual damage (not clean surface)
            is_damage, damage_score = self._verify_damage_region(
                img_array, bbox, mask, area_percentage
            )
            
            # Relaxed threshold - accept if it looks somewhat like damage
            if not is_damage and damage_score < 30:
                print(f"   ⊘ Region {i+1}: Ignored (doesn't look like damage: score={damage_score:.2f})")
                continue
            
            print(f"   ✓ Region {i+1} passed filters: damage_score={damage_score:.1f}, area={area_percentage:.2f}%")
            
            # Classify damage type based on shape features
            damage_type, damage_confidence = self._classify_damage(
                bbox, mask, img_width, img_height
            )
            
            # FILTER 3: Require minimum combined confidence
            # Lowered threshold - trust model's detection if it passed visual verification
            combined_confidence = (score * 0.6) + (damage_confidence * 0.4)
            if combined_confidence < 0.35:
                print(f"   ⊘ Region {i+1}: Low confidence ({combined_confidence:.2f})")
                continue
            
            detection = {
                'damage_type': damage_type,
                'confidence': float(combined_confidence),
                'damage_confidence': float(damage_confidence),
                'bbox': bbox.tolist(),
                'area_pixels': int(area_pixels),
                'area_percentage': float(area_percentage),
                'mask': mask
            }
            
            detections.append(detection)
            
            print(f"   ✓ Region {i+1}: {damage_type} | Conf: {combined_confidence:.2f} | Area: {area_percentage:.1f}%")
        
        return detections
    
    def _verify_damage_region(self, image, bbox, mask, area_percentage):
        """
        Verify if a detected region actually looks like damage
        Returns: (is_damage: bool, damage_score: float)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return False, 0.0
        
        # Convert to different color spaces for analysis
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Calculate damage indicators
        damage_score = 0.0
        
        # 1. Edge density (damage has irregular edges)
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density > 0.15:  # High edge density
            damage_score += 30
        elif edge_density > 0.08:
            damage_score += 15
        
        # 2. Color variance (damage often has color changes)
        color_std = np.std(roi)
        if color_std > 40:  # High variance suggests damage
            damage_score += 25
        elif color_std > 25:
            damage_score += 10
        
        # 3. Texture roughness (damage has rough texture)
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        if texture_variance > 500:
            damage_score += 25
        elif texture_variance > 200:
            damage_score += 10
        
        # 4. Dark regions (scratches/dents often create shadows)
        dark_pixels = np.sum(roi_gray < 80) / roi_gray.size
        if  0.1 < dark_pixels < 0.6:  # Some darkness but not entire region
            damage_score += 20
        
        # Relaxed threshold: Need at least 35 points to consider as damage
        # Lower threshold to catch more real damage (was missing actual dents)
        is_damage = damage_score >= 35
        
        return is_damage, damage_score
    
    def _merge_nearby_detections(self, boxes, scores, masks, medium_boxes, medium_scores, medium_masks, img_width, img_height):
        """
        Merge nearby detections to capture full damage area
        Expands small detections and clusters nearby regions
        """
        if len(boxes) == 0:
            return boxes, scores, masks
        
        merged_boxes = []
        merged_scores = []
        merged_masks = []
        used_indices = set()
        
        for i in range(len(boxes)):
            if i in used_indices:
                continue
            
            current_box = boxes[i]
            current_score = scores[i]
            current_mask = masks[i, 0]
            
            # Find all nearby detections (including medium confidence)
            x1, y1, x2, y2 = current_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Check for nearby medium-confidence detections within 150 pixels
            nearby_masks = [current_mask]
            nearby_scores = [current_score]
            
            for j in range(len(medium_boxes)):
                mx1, my1, mx2, my2 = medium_boxes[j]
                mcx, mcy = (mx1 + mx2) / 2, (my1 + my2) / 2
                
                # Calculate distance
                dist = np.sqrt((cx - mcx)**2 + (cy - mcy)**2)
                
                if dist < 150:  # Within 150 pixels
                    nearby_masks.append(medium_masks[j, 0])
                    nearby_scores.append(medium_scores[j])
            
            # Merge masks by combining them
            if len(nearby_masks) > 1:
                combined_mask = np.zeros_like(current_mask)
                for mask in nearby_masks:
                    combined_mask = np.maximum(combined_mask, mask)
                
                # Expand the merged region slightly (dilate)
                kernel = np.ones((15, 15), np.uint8)
                combined_mask = cv2.dilate((combined_mask > 0.5).astype(np.uint8), kernel, iterations=1).astype(float)
                
                # Recalculate bounding box from merged mask
                rows = np.any(combined_mask > 0.5, axis=1)
                cols = np.any(combined_mask > 0.5, axis=0)
                
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    
                    new_box = np.array([x1, y1, x2, y2])
                    new_score = max(nearby_scores)  # Use highest confidence
                    
                    merged_boxes.append(new_box)
                    merged_scores.append(new_score)
                    merged_masks.append(combined_mask)
                    used_indices.add(i)
            else:
                # Single detection - expand it slightly
                kernel = np.ones((10, 10), np.uint8)
                expanded_mask = cv2.dilate((current_mask > 0.5).astype(np.uint8), kernel, iterations=1).astype(float)
                
                # Recalculate bbox
                rows = np.any(expanded_mask > 0.5, axis=1)
                cols = np.any(expanded_mask > 0.5, axis=0)
                
                if rows.any() and cols.any():
                    y1, y2 = np.where(rows)[0][[0, -1]]
                    x1, x2 = np.where(cols)[0][[0, -1]]
                    
                    merged_boxes.append(np.array([x1, y1, x2, y2]))
                    merged_scores.append(current_score)
                    merged_masks.append(expanded_mask)
                    used_indices.add(i)
        
        if len(merged_boxes) == 0:
            return boxes, scores, masks
        
        # Convert back to arrays
        merged_boxes = np.array(merged_boxes)
        merged_scores = np.array(merged_scores)
        merged_masks = np.array(merged_masks)[:, np.newaxis, :, :]  # Add channel dimension back
        
        print(f"🔗 Merged {len(boxes)} detections into {len(merged_boxes)} larger regions")
        
        return merged_boxes, merged_scores, merged_masks
    
    def _classify_damage(self, bbox, mask, img_width, img_height):
        """
        Improved damage classification based on actual damage characteristics
        """
        # Calculate features
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / max(height, 1)
        
        # Calculate mask properties  
        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 'Dent', 0.5
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        area_percentage = area / (img_width * img_height)
        perimeter = cv2.arcLength(cnt, True)
        
        # Shape analysis
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        
        # Solidity (how filled vs convex hull)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        
        # Bounding box fill
        bbox_area = width * height
        fill_ratio = area / (bbox_area + 1e-6)
        
        # Scoring system based on REAL damage characteristics
        scores = {
            'Scratch': 0,
            'Dent': 0,
            'Shatter': 0,
            'Dislocation': 0
        }
        
        # SCRATCH: Linear, elongated, small
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            scores['Scratch'] += 45
        if area_percentage < 0.015:  # < 1.5%
            scores['Scratch'] += 35
        if circularity < 0.25:
            scores['Scratch'] += 15
        if solidity > 0.7:
            scores['Scratch'] += 5
        
        # DENT: Rounded/irregular, medium size, filled
        if 0.6 <= aspect_ratio <= 1.8:
            scores['Dent'] += 40
        if 0.004 <= area_percentage <= 0.04:  # 0.4-4%
            scores['Dent'] += 40
        if 0.35 <= circularity <= 0.75:
            scores['Dent'] += 15
        if solidity > 0.55:
            scores['Dent'] += 5
        
        # SHATTER: Fragmented, irregular, not solid
        if circularity < 0.3:
            scores['Shatter'] += 30
        if solidity < 0.6:
            scores['Shatter'] += 40
        if area_percentage > 0.015:
            scores['Shatter'] += 20
        if aspect_ratio > 1.5 or aspect_ratio < 0.65:
            scores['Shatter'] += 10
        
        # DISLOCATION: Large, solid piece
        if area_percentage > 0.035:  # > 3.5%
            scores['Dislocation'] += 50
        if solidity > 0.65:
            scores['Dislocation'] += 30
        if 0.7 <= aspect_ratio <= 1.5:
            scores['Dislocation'] += 20
        
        # Select damage type
        damage_type = max(scores, key=scores.get)
        max_score = scores[damage_type]
        confidence = min(max_score / 100.0, 0.95)
        
        # Fallback for low scores - choose based on most distinctive feature
        if max_score < 35:
            if area_percentage > 0.04:
                damage_type = 'Dislocation'
                confidence = 0.65
            elif aspect_ratio > 2.5 or aspect_ratio < 0.4:
                damage_type = 'Scratch'
                confidence = 0.60
            else:
                damage_type = 'Dent'
                confidence = 0.70
        
        return damage_type, confidence
    
    def get_annotated_image(self, image_path, detections):
        """
        Create annotated image with damage regions highlighted
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Color map for damage types
        color_map = {
            'Scratch': (255, 100, 100),    # Red
            'Dent': (100, 255, 100),       # Green
            'Shatter': (255, 165, 0),      # Orange
            'Dislocation': (255, 255, 100) # Yellow
        }
        
        overlay = image.copy()
        
        for i, det in enumerate(detections):
            # Get color
            color = color_map.get(det['damage_type'], (200, 200, 200))
            
            # Draw mask
            mask = (det['mask'] > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Apply colored mask
            overlay[mask_resized > 0] = overlay[mask_resized > 0] * 0.5 + np.array(color) * 0.5
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{det['damage_type']} {det['confidence']:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert back to BGR for saving
        annotated = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        return annotated
