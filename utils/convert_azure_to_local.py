import json

# Load Azure app settings exported from Azure CLI
with open('azure_appsettings.json', 'r') as f:
    azure_settings = json.load(f)

# Convert to local.settings.json format
local_settings = {
    "IsEncrypted": False,
    "Values": {
        "AzureWebJobsStorage": "<your-azurewebjobs-storage-connection-string>",
        "FUNCTIONS_WORKER_RUNTIME": "python"
    }
}

# Add all Azure settings as key-value pairs
for setting in azure_settings:
    key = setting.get('name')
    value = setting.get('value')
    if key and value:
        local_settings["Values"][key] = value

# Write to local.settings.json
with open('local.settings.json', 'w') as f:
    json.dump(local_settings, f, indent=2)

print("local.settings.json created!")
