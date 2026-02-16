"""
FastAPI Application - Car Damage Detection API
"""
import os
import uuid
from pathlib import Path
from typing import List

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import DamageDetector
from config import config


# Initialize FastAPI
app = FastAPI(
    title="AutoDamage AI v2",
    description="Car Damage Detection with Trained Mask R-CNN",
    version="2.0.0"
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
        detector = DamageDetector()
    except FileNotFoundError as e:
        print(f"⚠️  Model not found. Please train first: python train.py")
        print(f"    Error: {e}")


# Response Models
class Detection(BaseModel):
    damage_type: str
    confidence: float
    damage_confidence: float
    bbox: List[float]
    area_pixels: int
    area_percentage: float


class InspectionResponse(BaseModel):
    success: bool
    total_damages: int
    detections: List[Detection]
    severity: str
    estimated_cost_min: int
    estimated_cost_max: int
    annotated_image_url: str


# Severity and Cost Estimation
def estimate_severity(detections):
    """Estimate overall damage severity"""
    if not detections:
        return "None", 0, 0
    
    total_area = sum(d['area_percentage'] for d in detections)
    num_damages = len(detections)
    
    # Cost ranges (INR)
    cost_per_type = {
        'Scratch': (1500, 12000),
        'Dent': (2500, 20000),
        'Shatter': (5000, 35000),
        'Dislocation': (5000, 40000)
    }
    
    # Calculate total cost
    total_cost_min = 0
    total_cost_max = 0
    
    for det in detections:
        damage_type = det['damage_type']
        min_cost, max_cost = cost_per_type.get(damage_type, (2000, 15000))
        
        # Scale by area
        area_factor = 1 + (det['area_percentage'] / 10)  # 10% area = 2x cost
        
        total_cost_min += int(min_cost * area_factor)
        total_cost_max += int(max_cost * area_factor)
    
    # Determine severity
    if total_area < 5 and num_damages <= 2:
        severity = "Low"
    elif total_area < 15 and num_damages <= 4:
        severity = "Medium"
    else:
        severity = "High"
    
    return severity, total_cost_min, total_cost_max


# API Endpoints
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "ok",
        "message": "AutoDamage AI v2 - Trained Mask R-CNN",
        "model_loaded": detector is not None
    }


@app.post("/api/v1/inspect", response_model=InspectionResponse)
async def inspect_damage(file: UploadFile = File(...)):
    """
    Detect car damage from uploaded image
    """
    if detector is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please train the model first: python train.py"
        )
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPG, PNG, BMP, or WEBP")
    
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
        
        # Estimate severity and cost
        severity, cost_min, cost_max = estimate_severity(detections)
        
        # Prepare response
        response = InspectionResponse(
            success=True,
            total_damages=len(detections),
            detections=[
                Detection(
                    damage_type=d['damage_type'],
                    confidence=d['confidence'],
                    damage_confidence=d['damage_confidence'],
                    bbox=d['bbox'],
                    area_pixels=d['area_pixels'],
                    area_percentage=d['area_percentage']
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
        print(f"❌ Error during detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get detection statistics"""
    return {
        "model_status": "loaded" if detector is not None else "not_loaded",
        "device": str(config.DEVICE),
        "min_confidence": config.MIN_CONFIDENCE,
        "supported_damage_types": ["Scratch", "Dent", "Shatter", "Dislocation"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
