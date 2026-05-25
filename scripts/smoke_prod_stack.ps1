param(
    [string]$ComposeProjectName = "booking_cancellation_smoke",
    [string]$EnvFile = ".env.smoke",
    [int]$AppPort = 8010,
    [int]$PostgresPublishedPort = 5542,
    [int]$MinioApiPort = 9100,
    [int]$MinioConsolePort = 9101,
    [int]$MaxAttempts = 36,
    [int]$DelaySeconds = 5,
    [switch]$KeepStack
)

$ErrorActionPreference = "Stop"
$previousAppEnvFile = $env:APP_ENV_FILE
$stackStarted = $false

function Invoke-Compose {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed: $($Arguments -join ' ')"
    }
}

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Push-Location $projectRoot
try {
    Write-Host "Preparing smoke-test env file..." -ForegroundColor Cyan

    Copy-Item .env.prod.example $EnvFile -Force
    (Get-Content $EnvFile) `
        -replace 'POSTGRES_PASSWORD=change-me', 'POSTGRES_PASSWORD=smoke-postgres-password' `
        -replace 'S3_SECRET_KEY=change-me', 'S3_SECRET_KEY=smoke-minio-secret' `
        -replace 'MINIO_ROOT_PASSWORD=change-me', 'MINIO_ROOT_PASSWORD=smoke-minio-secret' `
        -replace 'AIRFLOW_ADMIN_PASSWORD=change-me', 'AIRFLOW_ADMIN_PASSWORD=smoke-airflow-password' `
        -replace 'POSTGRES_PUBLISHED_PORT=5432', "POSTGRES_PUBLISHED_PORT=$PostgresPublishedPort" `
        -replace 'MINIO_API_PORT=9000', "MINIO_API_PORT=$MinioApiPort" `
        -replace 'MINIO_CONSOLE_PORT=9001', "MINIO_CONSOLE_PORT=$MinioConsolePort" `
        -replace 'APP_PORT=8000', "APP_PORT=$AppPort" `
        | Set-Content $EnvFile

    $env:APP_ENV_FILE = $EnvFile

    Write-Host "Resetting previous smoke stack..." -ForegroundColor Cyan
    Invoke-Compose -Arguments @("down", "-v", "--remove-orphans")

    Write-Host "Starting production-like smoke stack..." -ForegroundColor Cyan
    $stackStarted = $true
    Invoke-Compose -Arguments @("up", "--build", "-d", "postgres", "minio", "init-db", "seed-model", "app")

    $readyUrl = "http://127.0.0.1:$AppPort/ready"
    $frontendHealthUrl = "http://127.0.0.1:$AppPort/frontend-api/health"
    $rootUrl = "http://127.0.0.1:$AppPort/"

    $readyPayload = $null
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $readyResponse = Invoke-WebRequest -Uri $readyUrl -UseBasicParsing -TimeoutSec 5
            if ($readyResponse.StatusCode -eq 200) {
                $readyPayload = $readyResponse.Content | ConvertFrom-Json
                break
            }
        }
        catch {
            Write-Host "Waiting for /ready ($attempt/$MaxAttempts)..." -ForegroundColor Yellow
            Start-Sleep -Seconds $DelaySeconds
        }
    }

    if ($null -eq $readyPayload) {
        Write-Host ""
        Write-Host "Smoke test failed. Dumping compose status and logs..." -ForegroundColor Red
        Invoke-Compose -Arguments @("ps")
        docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml logs --no-color
        if ($LASTEXITCODE -ne 0) {
            throw "Smoke test failed and logs could not be collected."
        }
        throw "Timed out waiting for $readyUrl"
    }

    $frontendHealthResponse = Invoke-WebRequest -Uri $frontendHealthUrl -UseBasicParsing -TimeoutSec 5
    $rootResponse = Invoke-WebRequest -Uri $rootUrl -UseBasicParsing -TimeoutSec 5

    Write-Host ""
    Write-Host "Smoke test passed." -ForegroundColor Green
    Write-Host "Readiness:" -ForegroundColor Green
    $readyPayload | ConvertTo-Json -Depth 4
    Write-Host "Frontend health status code: $($frontendHealthResponse.StatusCode)" -ForegroundColor Green
    Write-Host "Root page status code: $($rootResponse.StatusCode)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Stack endpoints:" -ForegroundColor Green
    Write-Host "  App/UI/API: $rootUrl"
    Write-Host "  Swagger: http://127.0.0.1:$AppPort/docs"
    Write-Host "  Ready: $readyUrl"
    Write-Host "  MinIO API: http://127.0.0.1:$MinioApiPort"
    Write-Host "  MinIO Console: http://127.0.0.1:$MinioConsolePort"
    Write-Host ""
    Invoke-Compose -Arguments @("ps")
}
catch {
    if ($stackStarted) {
        Write-Host ""
        Write-Host "Smoke stack status:" -ForegroundColor Yellow
        docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml ps
        Write-Host ""
        Write-Host "Smoke stack logs:" -ForegroundColor Yellow
        docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml logs --no-color
    }

    Write-Host ""
    Write-Host "Production-like smoke test failed." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    throw
}
finally {
    if ($null -eq $previousAppEnvFile) {
        Remove-Item Env:APP_ENV_FILE -ErrorAction SilentlyContinue
    }
    else {
        $env:APP_ENV_FILE = $previousAppEnvFile
    }

    if (-not $KeepStack) {
        Write-Host ""
        Write-Host "Cleaning up smoke stack..." -ForegroundColor Cyan
        docker compose -p $ComposeProjectName --env-file $EnvFile -f docker-compose.prod.yml down -v --remove-orphans
    }
    else {
        Write-Host ""
        Write-Host "Keeping smoke stack running because -KeepStack was provided." -ForegroundColor Yellow
    }

    Pop-Location
}
