param(
    [string]$OutputPath = ".\docker-images\booking-cancellation-app-latest.tar",
    [switch]$BuildIfMissing
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedOutputPath = Join-Path $projectRoot $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutputPath
$imageTag = "booking-cancellation-app:latest"

Push-Location $projectRoot
try {
    docker image inspect $imageTag *> $null
    if ($LASTEXITCODE -ne 0) {
        if (-not $BuildIfMissing) {
            throw "Docker image $imageTag was not found locally. Build it first or rerun this script with -BuildIfMissing."
        }

        Write-Host "Docker image is missing; building it before export..." -ForegroundColor Cyan
        docker build -t $imageTag .
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build $imageTag."
        }
    }

    if (-not (Test-Path $outputDirectory)) {
        New-Item -ItemType Directory -Path $outputDirectory | Out-Null
    }

    Write-Host "Saving Docker image to tar archive..." -ForegroundColor Cyan
    docker save -o $resolvedOutputPath $imageTag
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to save $imageTag to $resolvedOutputPath."
    }

    Write-Host ""
    Write-Host "Docker image archive is ready." -ForegroundColor Green
    Write-Host "Path:" -ForegroundColor Green
    Write-Host "  $resolvedOutputPath"
}
finally {
    Pop-Location
}
