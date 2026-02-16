"""
FastAPI Application - Car Damage Detection API v3
Uses ResNet50 Classifier + Grad-CAM for accurate damage detection
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Initialize FastAPI
app = FastAPI(
    title="AutoDamage AI v3",
    description="Car Damage Detection with ResNet50 Classifier + Grad-CAM",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Initialize detector
detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector
    try:
        from inference_v3 import DamageDetector
        detector = DamageDetector()
    except FileNotFoundError as e:
        print(f"⚠️  Model not found: {e}")
        print(f"    Train first: python train_classifier.py")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        import traceback
        traceback.print_exc()


# Response Models
class Detection(BaseModel):
    damage_type: str
    confidence: float
    bbox: List[float]
    area_pixels: int
    area_percentage: float
    severity: Optional[str] = None


class InspectionResponse(BaseModel):
    success: bool
    total_damages: int
    detections: List[Detection]
    severity: str
    estimated_cost_min: int
    estimated_cost_max: int
    annotated_image_url: str


# Cost Estimation
COST_MAP = {
    'Minor':    (2000, 15000),
    'Moderate': (15000, 50000),
    'Severe':   (50000, 200000),
}


def estimate_cost(detections):
    """Estimate damage cost based on severity"""
    if not detections:
        return "None", 0, 0
    
    # Use the most severe detection
    severity_order = {'Severe': 3, 'Moderate': 2, 'Minor': 1}
    worst = max(detections, key=lambda d: severity_order.get(d['damage_type'], 0))
    
    damage_type = worst['damage_type']
    cost_min, cost_max = COST_MAP.get(damage_type, (2000, 15000))
    
    # Scale by area
    total_area = sum(d['area_percentage'] for d in detections)
    area_factor = 1 + (total_area / 20)
    
    severity = worst.get('severity', 'Low')
    
    return severity, int(cost_min * area_factor), int(cost_max * area_factor)


# API Endpoints
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AutoDamage AI v3 - ResNet50 Classifier + Grad-CAM",
        "model_loaded": detector is not None
    }


@app.post("/api/v1/inspect", response_model=InspectionResponse)
async def inspect_damage(file: UploadFile = File(...)):
    """Detect car damage from uploaded image"""
    if detector is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Train first: python train_classifier.py"
        )
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Save uploaded file
    file_id = uuid.uuid4()
    file_extension = Path(file.filename).suffix
    saved_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"\n📸 Processing: {saved_path}")
    
    try:
        # Run detection
        detections = detector.detect(str(saved_path))
        
        # Generate annotated image
        annotated_image = detector.get_annotated_image(str(saved_path), detections)
        annotated_path = UPLOAD_DIR / f"{file_id}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_image)
        
        # Estimate cost
        severity, cost_min, cost_max = estimate_cost(detections)
        
        # Prepare response
        response = InspectionResponse(
            success=True,
            total_damages=len(detections),
            detections=[
                Detection(
                    damage_type=d['damage_type'],
                    confidence=d['confidence'],
                    bbox=[float(b) for b in d['bbox']],
                    area_pixels=d['area_pixels'],
                    area_percentage=d['area_percentage'],
                    severity=d.get('severity', severity)
                )
                for d in detections
            ],
            severity=severity,
            estimated_cost_min=cost_min,
            estimated_cost_max=cost_max,
            annotated_image_url=f"/uploads/{file_id}_annotated.jpg"
        )
        
        print(f"✅ Detected {len(detections)} damage regions")
        print(f"   Severity: {severity} | Cost: ₹{cost_min:,} - ₹{cost_max:,}\n")
        
        return response
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    return {
        "model_status": "loaded" if detector is not None else "not_loaded",
        "model_type": "ResNet50 Classifier + Grad-CAM",
        "supported_classes": ["Minor Damage", "Moderate Damage", "Severe Damage", "No Damage"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
