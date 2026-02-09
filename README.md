# Milo Photo Poster

An Azure Functions application that automatically posts a daily photo of Milo the cat using the Postly API.

## Overview

This application runs as a scheduled Azure Function that:
1. **Analyzes recent photos** from Azure Blob Storage using Azure Computer Vision API
2. **Selects the most appealing photo** based on quality, composition, and cat-related content
3. **Falls back to AI generation** using Azure OpenAI DALL-E if no suitable photos are found
4. **Generates witty captions** using GPT-4, tailored to the day, season, holidays, and photo content
5. **Posts to social media** via the Postly API with AI-generated captions that reflect Milo's grumpy personality

The function runs daily at 10:00 AM UTC, ensuring Milo gets his daily spotlight! 🐱

## Features

- **Smart Photo Selection**: Uses Azure Computer Vision to score photos based on quality, composition, and relevance
- **AI-Powered Appearance Analysis**: Uses GPT-4 Vision to analyze actual Milo photos and extract detailed visual characteristics
- **Mood-Based AI Generation**: When no suitable photos are found, DALL-E 3 generates images of Milo with random moods (happy, playful, sleepy, curious, gloomy, angry, regal, cozy) based on his actual appearance
- **Context-Aware Caption Generation**: AI-generated witty captions that adapt to day of week, season, holidays, and photo content, reflecting Milo's grumpy personality
- **Automated Posting**: Seamless integration with Postly API for social media management
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Configurable**: Flexible settings for storage containers, scoring parameters, and scheduling

## Prerequisites

Before deploying this application, you'll need:

1. **Azure Subscription** - [Create a free account](https://azure.microsoft.com/free/)
2. **Azure Storage Account** - For storing Milo's photos
3. **Azure Computer Vision Resource** - For analyzing photo quality
4. **Azure OpenAI Service** - With two deployments:
   - **DALL-E 3** for AI image generation
   - **GPT-4 Vision** (gpt-4o or gpt-4-turbo-vision) for analyzing Milo's appearance
5. **Postly Account** - [Sign up at Postly.ai](https://postly.ai/) and obtain API credentials
6. **Azure Functions Core Tools** (for local development) - [Installation guide](https://learn.microsoft.com/azure/azure-functions/functions-run-local)
7. **Python 3.9-3.11** - Azure Functions currently supports Python 3.9, 3.10, and 3.11

## Setup Instructions

### 1. Create Azure Resources

#### Storage Account
```bash
# Create resource group
az group create --name milo-photos-rg --location eastus

# Create storage account
az storage account create \
  --name milophotosstg \
  --resource-group milo-photos-rg \
  --location eastus \
  --sku Standard_LRS

# Create blob container
az storage container create \
  --name milo-photos \
  --account-name milophotosstg
```

#### Computer Vision Resource
```bash
az cognitiveservices account create \
  --name milo-computer-vision \
  --resource-group milo-photos-rg \
  --kind ComputerVision \
  --sku F0 \
  --location eastus
```

#### Azure OpenAI Resource
```bash
# Create Azure OpenAI resource
az cognitiveservices account create \
  --name milo-openai \
  --resource-group milo-photos-rg \
  --kind OpenAI \
  --sku S0 \
  --location eastus

# Deploy models (use Azure Portal for this step)
# Go to Azure OpenAI Studio > Deployments > Create new deployments
# 1. Select: dall-e-3, Name: dall-e-3
# 2. Select: gpt-4o (or gpt-4-turbo-vision), Name: gpt-4o
```

#### Function App
```bash
az functionapp create \
  --resource-group milo-photos-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name milo-photo-poster \
  --storage-account milophotosstg \
  --os-type linux
```

### 2. Configure Application Settings

Get your connection strings and keys:

```bash
# Storage connection string
az storage account show-connection-string \
  --name milophotosstg \
  --resource-group milo-photos-rg

# Computer Vision endpoint and key
az cognitiveservices account show \
  --name milo-computer-vision \
  --resource-group milo-photos-rg
az cognitiveservices account keys list \
  --name milo-computer-vision \
  --resource-group milo-photos-rg

# Azure OpenAI endpoint and key
az cognitiveservices account show \
  --name milo-openai \
  --resource-group milo-photos-rg
az cognitiveservices account keys list \
  --name milo-openai \
  --resource-group milo-photos-rg
```

Configure the Function App settings:

```bash
az functionapp config appsettings set \
  --name milo-photo-poster \
  --resource-group milo-photos-rg \
  --settings \
    AZURE_STORAGE_CONNECTION_STRING="<your-storage-connection-string>" \
    BLOB_CONTAINER_NAME="milo-photos" \
    COMPUTER_VISION_ENDPOINT="https://<region>.api.cognitive.microsoft.com/" \
    COMPUTER_VISION_KEY="<your-cv-key>" \
    OPENAI_API_KEY="<your-openai-key>" \
    OPENAI_ENDPOINT="https://<your-openai-resource>.openai.azure.com/" \
    OPENAI_DEPLOYMENT_NAME="dall-e-3" \
    OPENAI_GPT4V_DEPLOYMENT_NAME="gpt-4o" \
    POSTLY_API_KEY="<your-postly-api-key>" \
    POSTLY_WORKSPACE_ID="<your-postly-workspace-id>" \
    POSTLY_TARGET_PLATFORMS="<account-id-1>,<account-id-2>" \
    DAYS_TO_CHECK="1"
```

### 3. Upload Milo Photos to Blob Storage

#### Using Azure Portal
1. Navigate to your Storage Account in the Azure Portal
2. Go to "Containers" and select "milo-photos"
3. Click "Upload" and select your photos
4. Upload JPG, PNG, or other image formats

#### Using Azure Storage Explorer
1. Download [Azure Storage Explorer](https://azure.microsoft.com/features/storage-explorer/)
2. Connect to your storage account
3. Navigate to the "milo-photos" container
4. Drag and drop photos to upload

#### Using Azure CLI
```bash
az storage blob upload \
  --account-name milophotosstg \
  --container-name milo-photos \
  --name photo1.jpg \
  --file /path/to/photo1.jpg
```

## Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/shafe123/milo-photo-poster.git
cd milo-photo-poster
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Local Settings
Copy the example settings file and fill in your values:

```bash
cp local.settings.json.example local.settings.json
# Edit local.settings.json with your actual credentials
```

### 5. Run Locally
```bash
func start
```

The function will start locally. To test the timer trigger immediately without waiting for the scheduled time, you can use:

```bash
# The function will execute based on its schedule
# For immediate testing, modify the timer trigger temporarily or use manual invocation
```

### 6. Manual Testing
You can test individual components:

```python
# Test blob storage connection
from azure.storage.blob import BlobServiceClient
client = BlobServiceClient.from_connection_string("<connection-string>")
container = client.get_container_client("milo-photos")
for blob in container.list_blobs():
    print(blob.name)
```

## Deployment

### Option 1: Deploy via Azure CLI
```bash
# From the project root directory
func azure functionapp publish milo-photo-poster
```

### Option 2: Deploy via VS Code
1. Install the [Azure Functions extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions)
2. Open the project in VS Code
3. Click the Azure icon in the sidebar
4. Sign in to your Azure account
5. Right-click your Function App and select "Deploy to Function App"

### Option 3: Deploy via GitHub Actions
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Azure Functions

on:
  push:
    branches:
      - main

env:
  AZURE_FUNCTIONAPP_NAME: milo-photo-poster
  AZURE_FUNCTIONAPP_PACKAGE_PATH: '.'
  PYTHON_VERSION: '3.11'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - name: 'Checkout GitHub Action'
      uses: actions/checkout@v3

    - name: Setup Python ${{ env.PYTHON_VERSION }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}

    - name: 'Install dependencies'
      run: |
        pip install -r requirements.txt

    - name: 'Deploy to Azure Functions'
      uses: Azure/functions-action@v1
      with:
        app-name: ${{ env.AZURE_FUNCTIONAPP_NAME }}
        package: ${{ env.AZURE_FUNCTIONAPP_PACKAGE_PATH }}
        publish-profile: ${{ secrets.AZURE_FUNCTIONAPP_PUBLISH_PROFILE }}
```

## Monitoring

### View Logs in Azure Portal
1. Navigate to your Function App in the Azure Portal
2. Click on "Functions" > "daily_milo_post"
3. Click "Monitor" to see execution history
4. Click on individual executions to see detailed logs

### Live Streaming Logs
```bash
func azure functionapp logstream milo-photo-poster
```

### Application Insights
For advanced monitoring, enable Application Insights:

```bash
az monitor app-insights component create \
  --app milo-photo-poster-insights \
  --location eastus \
  --resource-group milo-photos-rg

# Link to Function App
az functionapp config appsettings set \
  --name milo-photo-poster \
  --resource-group milo-photos-rg \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY="<instrumentation-key>"
```

## How It Works

### Photo Selection Algorithm

The function uses a sophisticated scoring system to select the best photo:

1. **Recent Photos**: Scans blob storage for photos modified in the last N days (default: 7)
2. **Computer Vision Analysis**: Each photo is analyzed for:
   - Overall description and confidence
   - Tags (looking for cat-related content)
   - Adult/racy content (filtered out)
   - Image type (prefers photographs over clip art)
   - Color information
3. **Appeal Score Calculation**:
   - Base score from description confidence (0-30 points)
   - Bonus for cat-related tags (0-20 points)
   - Penalty for inappropriate content (-50 points)
   - Penalty for clip art or line drawings (-20/-15 points)
   - Bonus for color images (+10 points)
4. **Selection**: Photo with highest score (minimum 30 points) is selected

### AI Fallback

If no suitable photo is found in blob storage:

#### Step 1: Analyze Milo's Appearance with GPT-4 Vision
- Retrieves 2-3 recent Milo photos from blob storage
- Sends them to GPT-4 Vision (gpt-4o) to "see" what Milo actually looks like
- GPT-4 Vision creates a detailed physical description including:
  - Fur color and pattern (e.g., "orange tabby with bold dark stripes")
  - Distinctive markings (e.g., "M-shaped marking on forehead", "white paws")
  - Eye color
  - Fur length and texture
  - Unique features
- Example output: "a fluffy orange tabby cat with bold dark stripes, white paws, and bright green eyes"

#### Step 2: Generate Image with DALL-E 3
- Randomly selects from 8 different moods: happy, playful, sleepy, curious, gloomy, angry, regal, or cozy
- Incorporates Milo's actual appearance description from GPT-4 Vision into the prompt
- Example prompt: "A high-quality photo of Milo, a fluffy orange tabby cat with bold dark stripes and white paws, looking playful..."
- Generated in 1024x1024 HD quality with natural, photorealistic style
- Result: AI-generated images that actually resemble Milo, not just a generic cat!

**Why this works:** While DALL-E 3 can't directly see photos, GPT-4 Vision can. By using GPT-4 Vision as a "bridge" to describe Milo's appearance, we ensure DALL-E 3 generates images that match how Milo really looks.

### Caption Generation

Captions are dynamically generated using AI to keep content fresh and engaging:

1. **Context Collection**: Gathers current temporal context including:
   - Day of week (Monday through Sunday)
   - Season (winter, spring, summer, fall for Northern Hemisphere)
   - Notable holidays (New Year's Day, Valentine's Day, St. Patrick's Day, April Fool's Day, Halloween, Thanksgiving, Christmas, New Year's Eve)

2. **Image Analysis**: Uses Computer Vision description of the selected photo (when available)

3. **AI Caption Generation**: Uses Azure OpenAI GPT-4 to create:
   - Short, witty captions (max 15 words)
   - Grumpy but lovable personality matching Milo's character
   - Context-aware references to day, season, or holidays
   - Fallback captions if API is unavailable

4. **Caption Format**:
   - Prefix: "Daily Milo! 😾" (grumpy cat emoji)
   - Middle: AI-generated witty caption
   - Suffix: "#Milo #Cats #GrumpyCat"
   - Example: "Daily Milo! 😾 Another Monday means another judgmental stare. #Milo #Cats #GrumpyCat"

### Postly Integration

The selected or generated image is posted via the Postly API using a two-step process:

1. **Upload Image**: The image is uploaded to Postly's `/files` endpoint, which returns a URL
2. **Create Post**: A post is created using the uploaded image URL, caption, and target platform accounts
3. **Publish**: The post is published to the specified social media platforms

#### Getting Postly Target Platform IDs

The `POSTLY_TARGET_PLATFORMS` environment variable should contain comma-separated account IDs for the social media accounts you want to post to. To get these IDs:

1. Log in to your Postly account
2. Navigate to your workspace settings
3. Find the connected social media accounts
4. Copy the account IDs for the platforms you want to post to
5. Set them as a comma-separated list: `account-id-1,account-id-2,account-id-3`

**Note**: If `POSTLY_TARGET_PLATFORMS` is not set, the post will be created in the workspace but may not be automatically published to specific platforms. Consult the [Postly API documentation](https://docs.postly.ai/) for more details on managing target platforms.

## Cost Optimization Tips

1. **Storage**: Use Standard LRS for blob storage (~$0.02/GB/month)
2. **Computer Vision**: Free tier includes 5,000 transactions/month (sufficient for daily use)
3. **Azure OpenAI**: Pay-per-use; only charged when generating AI images
4. **Function App**: Consumption plan charges only for execution time (minimal cost for daily function)
5. **Application Insights**: Configure sampling to reduce costs (already enabled in host.json)

**Estimated Monthly Cost**: $5-15 depending on AI image generation frequency

## Troubleshooting

### Function Not Triggering
- Check the timer expression in `function_app.py`
- Verify the Function App is running (not stopped)
- Check Application Insights for execution logs

### Blob Storage Access Issues
- Verify `AZURE_STORAGE_CONNECTION_STRING` is correct
- Ensure the container name matches `BLOB_CONTAINER_NAME`
- Check that the container exists and has photos

### Computer Vision Errors
- Verify endpoint URL format: `https://<region>.api.cognitive.microsoft.com/`
- Ensure API key is valid
- Check that images are publicly accessible or use SAS tokens

### Postly API Errors
- Verify API key and workspace ID
- Check Postly API documentation for endpoint changes
- Review response error messages in logs

### AI Image Generation Failures
- Ensure DALL-E 3 deployment exists in Azure OpenAI
- Verify deployment name matches `OPENAI_DEPLOYMENT_NAME`
- Check quota limits in Azure OpenAI resource

## Security Best Practices

### Using Azure Key Vault

Store sensitive credentials in Azure Key Vault:

```bash
# Create Key Vault
az keyvault create \
  --name milo-photo-vault \
  --resource-group milo-photos-rg \
  --location eastus

# Store secrets
az keyvault secret set --vault-name milo-photo-vault --name "PostlyApiKey" --value "<your-api-key>"

# Grant Function App access
az functionapp identity assign \
  --name milo-photo-poster \
  --resource-group milo-photos-rg

# Update app settings to reference Key Vault
az functionapp config appsettings set \
  --name milo-photo-poster \
  --resource-group milo-photos-rg \
  --settings POSTLY_API_KEY="@Microsoft.KeyVault(SecretUri=https://milo-photo-vault.vault.azure.net/secrets/PostlyApiKey/)"
```

### Using Managed Identities

Configure managed identity for Azure services:

```bash
# Enable system-assigned managed identity
az functionapp identity assign \
  --name milo-photo-poster \
  --resource-group milo-photos-rg

# Grant access to Storage Account
az role assignment create \
  --assignee <principal-id> \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/<subscription-id>/resourceGroups/milo-photos-rg/providers/Microsoft.Storage/storageAccounts/milophotosstg"
```

Then update code to use `DefaultAzureCredential` instead of connection strings.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Support

For issues or questions:
- Open an issue in this repository
- Check Azure Functions documentation: https://docs.microsoft.com/azure/azure-functions/
- Check Postly API documentation: https://postly.ai/docs

## Acknowledgments

- Azure Functions team for the serverless platform
- Azure Cognitive Services for Computer Vision capabilities
- OpenAI and Microsoft for DALL-E integration
- Postly for the social media API