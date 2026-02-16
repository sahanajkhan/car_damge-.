# Test Trained Model API
# Test damage detection with trained Mask R-CNN

Write-Host "🧪 Testing Trained Damage Detector" -ForegroundColor Cyan
Write-Host ""

# Test image path
$testImage = "D:\hero\car-damage-detecting-MaskRCNN\customImages\unseen-data\160.jpg"

if (-not (Test-Path $testImage)) {
    Write-Host "❌ Test image not found: $testImage" -ForegroundColor Red
    exit 1
}

Write-Host "📸 Test image: 10.jpg" -ForegroundColor Green
Write-Host ""

# Make API request
Write-Host "🔄 Sending request to API..." -ForegroundColor Yellow

$response = curl.exe -X POST http://localhost:8000/api/v1/inspect `
    -F "file=@$testImage" `
    -H "accept: application/json" `
    --silent

if ($LASTEXITCODE -eq 0) {
    $result = $response | ConvertFrom-Json
    
    Write-Host ""
    Write-Host "✅ Detection Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Results:" -ForegroundColor Cyan
    Write-Host "   Total Damages: $($result.total_damages)" -ForegroundColor White
    Write-Host "   Severity: $($result.severity)" -ForegroundColor Yellow
    Write-Host "   Cost Range: ₹$($result.estimated_cost_min) - ₹$($result.estimated_cost_max)" -ForegroundColor White
    Write-Host ""
    
    if ($result.total_damages -gt 0) {
        Write-Host "🔍 Detected Damages:" -ForegroundColor Cyan
        foreach ($det in $result.detections) {
            Write-Host "   - $($det.damage_type): $([math]::Round($det.confidence * 100))% confidence | Area: $([math]::Round($det.area_percentage, 2))%" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "🖼️  Annotated image:" -ForegroundColor Cyan
    $imageUrl = "http://localhost:8000$($result.annotated_image_url)"
    Write-Host "   $imageUrl" -ForegroundColor Blue
    
    # Open in browser
    Start-Process $imageUrl
    
} else {
    Write-Host "❌ API request failed" -ForegroundColor Red
    Write-Host "   Make sure the API is running: python main.py" -ForegroundColor Yellow
}
