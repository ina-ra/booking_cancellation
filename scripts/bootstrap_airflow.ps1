param()

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $projectRoot
try {
    Write-Host "Starting local Airflow flow..." -ForegroundColor Cyan
    docker compose -f docker-compose.local.yml up --build -d postgres minio init-db seed-model airflow-standalone
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Airflow."
    }

    Write-Host ""
    Write-Host "Airflow is starting." -ForegroundColor Green
    Write-Host "UI:" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:8081"
    Write-Host "Login:" -ForegroundColor Green
    Write-Host "  Read AIRFLOW_ADMIN_USERNAME / AIRFLOW_ADMIN_PASSWORD from .env"
    Write-Host "Backfill example:" -ForegroundColor Green
    Write-Host "  docker compose -f docker-compose.local.yml exec airflow-standalone airflow dags backfill booking_batch_scoring --start-date 2026-04-14 --end-date 2026-04-16"
}
finally {
    Pop-Location
}
