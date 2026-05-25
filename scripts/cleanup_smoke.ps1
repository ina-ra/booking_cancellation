param(
    [string]$ComposeProjectName = "booking_cancellation_smoke",
    [string]$EnvFile = ".env.smoke"
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $projectRoot
try {
    $env:APP_ENV_FILE = $EnvFile

    Write-Host "Cleaning up smoke stack..." -ForegroundColor Cyan
    docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml down -v --remove-orphans
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clean up smoke stack."
    }

    Write-Host ""
    Write-Host "Smoke stack has been removed." -ForegroundColor Green
}
finally {
    Remove-Item Env:APP_ENV_FILE -ErrorAction SilentlyContinue
    Pop-Location
}
