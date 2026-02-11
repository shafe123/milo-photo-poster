# Deploy local.settings.json to Azure Function App
# Usage: .\deploy-settings.ps1 -ResourceGroup "milo-photos-rg" -FunctionAppName "milo-photo-poster"

param(
    [Parameter(Mandatory = $false)]
    [string]$ResourceGroup = "milo-photos-rg",
    
    [Parameter(Mandatory = $false)]
    [string]$FunctionAppName = "milo-photo-poster",
    
    [Parameter(Mandatory = $false)]
    [string]$SettingsFile = "local.settings.json"
)

# Check if local.settings.json exists
if (-not (Test-Path $SettingsFile)) {
    Write-Error "Settings file '$SettingsFile' not found!"
    exit 1
}

Write-Host "Reading settings from $SettingsFile..." -ForegroundColor Cyan

# Read and parse the JSON file
$settings = Get-Content $SettingsFile | ConvertFrom-Json

# Settings to exclude (Azure manages these automatically)
$excludeSettings = @(
    "FUNCTIONS_WORKER_RUNTIME",
    "FUNCTIONS_EXTENSION_VERSION",
    "WEBSITE_CONTENTAZUREFILECONNECTIONSTRING",
    "WEBSITE_CONTENTSHARE"
)

# Build the settings array for Azure CLI
$settingsArray = @()

foreach ($property in $settings.Values.PSObject.Properties) {
    $key = $property.Name
    $value = $property.Value
    
    # Skip excluded settings
    if ($excludeSettings -contains $key) {
        Write-Host "Skipping $key (managed by Azure)" -ForegroundColor Yellow
        continue
    }
    
    # Add to settings array
    $settingsArray += "$key=$value"
    Write-Host "Adding: $key" -ForegroundColor Green
}

if ($settingsArray.Count -eq 0) {
    Write-Warning "No settings to deploy!"
    exit 0
}

Write-Host "`nDeploying $($settingsArray.Count) settings to Function App '$FunctionAppName'..." -ForegroundColor Cyan

# Deploy settings to Azure
try {
    $result = az functionapp config appsettings set `
        --resource-group $ResourceGroup `
        --name $FunctionAppName `
        --settings @settingsArray `
        2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Settings deployed successfully!" -ForegroundColor Green
    }
    else {
        Write-Error "Failed to deploy settings: $result"
        exit 1
    }
}
catch {
    Write-Error "Error deploying settings: $_"
    exit 1
}

Write-Host "`nNote: Sensitive values have been deployed. Consider using Azure Key Vault for production." -ForegroundColor Yellow
