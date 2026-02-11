#!/bin/bash
# Deploy local.settings.json to Azure Function App
# Usage: ./deploy-settings.sh [resource-group] [function-app-name]

RESOURCE_GROUP="${1:-milo-photos-rg}"
FUNCTION_APP_NAME="${2:-milo-photo-poster}"
SETTINGS_FILE="local.settings.json"

# Check if local.settings.json exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Error: Settings file '$SETTINGS_FILE' not found!"
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Install with: apt-get install jq (or brew install jq on macOS)"
    exit 1
fi

echo "Reading settings from $SETTINGS_FILE..."

# Settings to exclude (Azure manages these automatically)
EXCLUDE_SETTINGS="FUNCTIONS_WORKER_RUNTIME|FUNCTIONS_EXTENSION_VERSION|WEBSITE_CONTENTAZUREFILECONNECTIONSTRING|WEBSITE_CONTENTSHARE"

# Extract settings from JSON and build settings array
SETTINGS_ARRAY=()

while IFS="=" read -r key value; do
    # Skip excluded settings
    if echo "$key" | grep -E "^($EXCLUDE_SETTINGS)$" > /dev/null; then
        echo "Skipping $key (managed by Azure)"
        continue
    fi
    
    SETTINGS_ARRAY+=("$key=$value")
    echo "Adding: $key"
done < <(jq -r '.Values | to_entries | .[] | "\(.key)=\(.value)"' "$SETTINGS_FILE")

if [ ${#SETTINGS_ARRAY[@]} -eq 0 ]; then
    echo "Warning: No settings to deploy!"
    exit 0
fi

echo ""
echo "Deploying ${#SETTINGS_ARRAY[@]} settings to Function App '$FUNCTION_APP_NAME'..."

# Deploy settings to Azure
if az functionapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$FUNCTION_APP_NAME" \
    --settings "${SETTINGS_ARRAY[@]}" \
    > /dev/null 2>&1; then
    echo ""
    echo "✓ Settings deployed successfully!"
else
    echo "Error: Failed to deploy settings"
    exit 1
fi

echo ""
echo "Note: Sensitive values have been deployed. Consider using Azure Key Vault for production."
