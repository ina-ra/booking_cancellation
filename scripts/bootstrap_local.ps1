param(
    [switch]$SkipTraining,
    [switch]$SkipAirflow,
    [string]$ImageTarPath = ".\docker-images\booking-cancellation-app-latest.tar",
    [switch]$AllowBuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonExe = Join-Path $projectRoot "venv\\Scripts\\python.exe"
$resolvedImageTarPath = Join-Path $projectRoot $ImageTarPath
$imageTag = "booking-cancellation-app:latest"

function Ensure-AppImage {
    param(
        [string]$ImageTag,
        [string]$TarPath,
        [bool]$CanBuild
    )

    docker image inspect $ImageTag *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Using existing Docker image $ImageTag." -ForegroundColor Cyan
        return
    }

    if (Test-Path $TarPath) {
        Write-Host "Loading Docker image from tar archive..." -ForegroundColor Cyan
        docker load -i $TarPath
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to load Docker image from $TarPath."
        }

        docker image inspect $ImageTag *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker archive $TarPath was loaded, but image $ImageTag is still unavailable."
        }

        return
    }

    if ($CanBuild) {
        Write-Host "Tar archive was not found; building application image..." -ForegroundColor Cyan
        docker build -t $ImageTag .
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build $ImageTag."
        }

        return
    }

    throw "Docker image $ImageTag was not found locally, and tar archive $TarPath does not exist. Run scripts\\export_app_image.ps1 on a machine where the image is already built, or rerun bootstrap with -AllowBuild."
}

function Wait-ForContainerHealth {
    param(
        [string]$ContainerName,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        $status = docker inspect $ContainerName --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $status = $status.Trim()
            if ($status -eq "healthy" -or $status -eq "running") {
                Write-Host "$ContainerName is ready." -ForegroundColor Cyan
                return
            }
        }

        Start-Sleep -Seconds 2
    }

    throw "Container $ContainerName did not become ready within $TimeoutSeconds seconds."
}

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment was not found at venv\\Scripts\\python.exe. Create it before bootstrap."
}

Push-Location $projectRoot
try {
    if (-not $SkipAirflow) {
        Ensure-AppImage -ImageTag $imageTag -TarPath $resolvedImageTarPath -CanBuild $AllowBuild.IsPresent
    }

    Write-Host "Starting local Postgres and MinIO..." -ForegroundColor Cyan
    docker compose -f docker-compose.local.yml up -d postgres minio
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start local infrastructure via docker compose."
    }

    Write-Host "Waiting for Postgres to become healthy..." -ForegroundColor Cyan
    Wait-ForContainerHealth -ContainerName "booking-cancellation-postgres" -TimeoutSeconds 90

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
    if (Test-Path $resolvedImageTarPath) {
        Write-Host "Image tar archive:" -ForegroundColor Green
        Write-Host "  $resolvedImageTarPath"
    }
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
