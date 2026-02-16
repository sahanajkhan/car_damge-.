# Start Training Script
# Train Mask R-CNN on Basel's Car Damage Dataset

Write-Host "🚀 AutoDamage AI v2 - Training Script" -ForegroundColor Cyan
Write-Host ""

# Check if dataset exists
$datasetPath = "D:\hero\Automated-Car-Damage-Detection\dataset\train"
if (-not (Test-Path $datasetPath)) {
    Write-Host "❌ Dataset not found at: $datasetPath" -ForegroundColor Red
    exit 1
}

Write-Host "📊 Dataset found: $datasetPath" -ForegroundColor Green
Write-Host ""

# Navigate to backend directory
Set-Location "D:\hero\autodamage-ai-v2\backend"

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt -q

Write-Host ""
Write-Host "🎯 Starting training..." -ForegroundColor Cyan
Write-Host "   This will take ~10-15 minutes" -ForegroundColor Gray
Write-Host "   Model will be saved to: models/damage_detector_best.pth" -ForegroundColor Gray
Write-Host ""

# Run training
python train.py

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Training complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start API: python main.py" -ForegroundColor White
    Write-Host "  2. Test: .\test_api.ps1" -ForegroundColor White
} else {
    Write-Host "❌ Training failed" -ForegroundColor Red
}
