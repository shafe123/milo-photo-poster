#!/bin/bash
# Load environment variables from local.settings.json into current shell session
# Usage: source ./utils/load-env.sh
# Note: Must use 'source' or '.' to set variables in the current session

SETTINGS_FILE="${1:-local.settings.json}"

# Check if local.settings.json exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Error: Settings file '$SETTINGS_FILE' not found!"
    echo "Current directory: $(pwd)"
    return 1 2>/dev/null || exit 1
fi

echo "Loading environment variables from $SETTINGS_FILE..."

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed."
    echo "Install with: sudo apt install jq (Ubuntu/Debian) or brew install jq (macOS)"
    return 1 2>/dev/null || exit 1
fi

# Parse JSON and export environment variables
count=0
while IFS="=" read -r key value; do
    export "$key=$value"
    echo "  ✓ $key"
    ((count++))
done < <(jq -r '.Values | to_entries[] | "\(.key)=\(.value)"' "$SETTINGS_FILE")

echo ""
echo "✓ Loaded $count environment variables into current session"
echo ""
echo "Note: These variables are only set for the current shell session."
echo "They will not persist after you close this terminal."
