param(
    [string]$ImageTarPath = ".\docker-images\booking-cancellation-app-latest.tar",
    [switch]$AllowBuild
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
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

Push-Location $projectRoot
try {
    Ensure-AppImage -ImageTag $imageTag -TarPath $resolvedImageTarPath -CanBuild $AllowBuild.IsPresent

    Write-Host "Starting Airflow standalone..." -ForegroundColor Cyan
    docker compose -f docker-compose.local.yml up -d airflow-standalone
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Airflow."
    }

    Write-Host ""
    Write-Host "Airflow is starting." -ForegroundColor Green
    Write-Host "UI:" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:8081"
    Write-Host "Login:" -ForegroundColor Green
    Write-Host "  Read AIRFLOW_ADMIN_USERNAME / AIRFLOW_ADMIN_PASSWORD from .env"
    if (Test-Path $resolvedImageTarPath) {
        Write-Host "Image tar archive:" -ForegroundColor Green
        Write-Host "  $resolvedImageTarPath"
    }
    Write-Host "Backfill example:" -ForegroundColor Green
    Write-Host "  docker compose -f docker-compose.local.yml exec airflow-standalone airflow dags backfill booking_batch_scoring --start-date 2026-04-14 --end-date 2026-04-16"
}
finally {
    Pop-Location
}
