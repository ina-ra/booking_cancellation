param(
    [switch]$SkipAirflow
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $projectRoot
try {
    if ($SkipAirflow) {
        Write-Host "Starting local stack without Airflow..." -ForegroundColor Cyan
        docker compose -f docker-compose.local.yml up --build -d postgres minio init-db seed-model app
    }
    else {
        Write-Host "Starting local stack with Airflow..." -ForegroundColor Cyan
        docker compose -f docker-compose.local.yml up --build -d
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start local infrastructure via docker compose."
    }

    Write-Host ""
    Write-Host "Local stack is ready." -ForegroundColor Green
    Write-Host "App UI/API:" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:8000"
    Write-Host "Swagger:" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:8000/docs"
    Write-Host "MinIO console:" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:9001"
    if (-not $SkipAirflow) {
        Write-Host "Airflow UI:" -ForegroundColor Green
        Write-Host "  http://127.0.0.1:8081"
        Write-Host "Airflow login:" -ForegroundColor Green
        Write-Host "  Read AIRFLOW_ADMIN_USERNAME / AIRFLOW_ADMIN_PASSWORD from .env"
    }
}
catch {
    Write-Host ""
    Write-Host "Bootstrap failed." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    throw
}
finally {
    Pop-Location
}
