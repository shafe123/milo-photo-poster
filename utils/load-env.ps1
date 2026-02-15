# Load environment variables from local.settings.json into current PowerShell session
# Usage: . .\utils\load-env.ps1
# Note: Use dot-sourcing (. .\script.ps1) to set variables in the current session

param(
    [Parameter(Mandatory = $false)]
    [string]$SettingsFile = "local.settings.json"
)

# Check if local.settings.json exists
if (-not (Test-Path $SettingsFile)) {
    Write-Error "Settings file '$SettingsFile' not found!"
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Expected file at: $(Join-Path (Get-Location) $SettingsFile)" -ForegroundColor Yellow
    return
}

Write-Host "Loading environment variables from $SettingsFile..." -ForegroundColor Cyan

try {
    # Read and parse the JSON file
    $settings = Get-Content $SettingsFile | ConvertFrom-Json
    
    # Counter for loaded variables
    $count = 0
    
    # Iterate through all values and set environment variables
    foreach ($property in $settings.Values.PSObject.Properties) {
        $key = $property.Name
        $value = $property.Value
        
        # Set environment variable in current process
        [System.Environment]::SetEnvironmentVariable($key, $value, [System.EnvironmentVariableTarget]::Process)
        
        Write-Host "  ✓ $key" -ForegroundColor Green
        $count++
    }
    
    Write-Host ""
    Write-Host "✓ Loaded $count environment variables into current session" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: These variables are only set for the current PowerShell session." -ForegroundColor Yellow
    Write-Host "They will not persist after you close this terminal." -ForegroundColor Yellow
}
catch {
    Write-Error "Failed to load settings: $_"
}
