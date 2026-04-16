param(
    [switch]$SkipTraining,
    [switch]$SkipAirflow
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot "venv\\Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment was not found at venv\\Scripts\\python.exe. Create it before bootstrap."
}

Push-Location $projectRoot
try {
    if (-not $SkipAirflow) {
        Write-Host "Building application image for DockerOperator..." -ForegroundColor Cyan
        docker build -t booking-cancellation-app:latest .
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build booking-cancellation-app:latest."
        }
    }

    Write-Host "Starting local Postgres and MinIO..." -ForegroundColor Cyan
    docker compose -f docker-compose.local.yml up -d postgres minio
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start local infrastructure via docker compose."
    }

    Write-Host "Initializing Postgres schema..." -ForegroundColor Cyan
    & $pythonExe -m src.interfaces.cli.init_db_cli
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize Postgres schema."
    }

    if (-not $SkipTraining) {
        Write-Host "Training model and uploading artifacts to S3..." -ForegroundColor Cyan
        & $pythonExe -m src.interfaces.cli.train_models_cli
        if ($LASTEXITCODE -ne 0) {
            throw "Training or S3 artifact upload failed."
        }
    }

    if (-not $SkipAirflow) {
        Write-Host "Starting Airflow after database and model initialization..." -ForegroundColor Cyan
        docker compose -f docker-compose.local.yml up -d airflow-standalone
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to start Airflow."
        }
    }

    Write-Host ""
    Write-Host "Local stack is ready." -ForegroundColor Green
    Write-Host "Start API with:" -ForegroundColor Green
    Write-Host "  py -m uvicorn src.interfaces.main:app --reload"
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
