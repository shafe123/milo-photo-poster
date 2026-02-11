# Deploy Azure Function App
# Usage: .\deploy.ps1 -FunctionAppName "milo-photo-poster"

param(
    [Parameter(Mandatory = $false)]
    [string]$FunctionAppName = "milo-photo-poster"
)

Write-Host "Deploying to Azure Function App: $FunctionAppName" -ForegroundColor Cyan
Write-Host ""

# Check if Azure Functions Core Tools is installed
try {
    $funcVersion = func --version
    Write-Host "✓ Azure Functions Core Tools version: $funcVersion" -ForegroundColor Green
}
catch {
    Write-Error "Azure Functions Core Tools not found. Install from: https://learn.microsoft.com/azure/azure-functions/functions-run-local"
    exit 1
}

# Run the deployment
Write-Host ""
Write-Host "Starting deployment..." -ForegroundColor Cyan
Write-Host ""

try {
    func azure functionapp publish $FunctionAppName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Deployment completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Cyan
        Write-Host "1. Deploy settings: .\deploy-settings.ps1" -ForegroundColor White
        Write-Host "2. Monitor logs: func azure functionapp logstream $FunctionAppName" -ForegroundColor White
        Write-Host "3. Test the function in Azure Portal or wait for the scheduled trigger" -ForegroundColor White
    }
    else {
        Write-Error "Deployment failed with exit code $LASTEXITCODE"
        exit 1
    }
}
catch {
    Write-Error "Deployment error: $_"
    exit 1
}
