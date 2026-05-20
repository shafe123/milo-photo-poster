# Milo Photo Poster

An Azure Functions application that automatically posts a daily photo of Milo the cat using the Postly API.

## Overview

This application runs as a scheduled Azure Function that:
1. **Detects Milo** using Azure Custom Vision to distinguish Milo from other cats
2. **Analyzes recent photos** from Azure Blob Storage using Azure Computer Vision API
3. **Selects the most appealing photo** based on quality, composition, and cat-related content
4. **Falls back to AI generation** using FLUX.2-pro if no suitable photos are found
5. **Generates witty captions** using GPT-4, tailored to the day, season, holidays, and photo content
6. **Posts to social media** via the Postly API with AI-generated captions that reflect Milo's grumpy personality

The function runs daily at 10:00 AM UTC, ensuring Milo gets his daily spotlight! 🐱

## Features

- **Milo Detection**: Uses Azure Custom Vision to identify Milo and filter out photos of other cats (like Emilio)
- **Smart Photo Selection**: Uses Azure Computer Vision to score photos based on quality, composition, and relevance
- **Duplicate Prevention**: Tracks posted photos to avoid reposting the same image within a configurable timeframe
- **AI-Powered Appearance Analysis**: Uses GPT-4 Vision to analyze actual Milo photos and extract detailed visual characteristics
- **Mood-Based AI Generation**: When no suitable photos are found, FLUX.2-pro generates photorealistic images of Milo with random moods (happy, playful, sleepy, curious, gloomy, angry, regal, cozy) based on his actual appearance
- **Context-Aware Caption Generation**: AI-generated witty captions that adapt to day of week, real-time weather conditions, holidays, and photo content, reflecting Milo's grumpy personality
- **Real-Time Weather Integration**: Fetches current weather from OpenWeatherMap One Call API 3.0 to create timely, relevant captions with actual conditions
- **Automated Posting**: Seamless integration with Postly API for social media management
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Configurable**: Flexible settings for storage containers, scoring parameters, and scheduling

## Prerequisites

Before deploying this application, you'll need:

1. **Azure Subscription** - [Create a free account](https://azure.microsoft.com/free/)
2. **Azure Storage Account** - For storing Milo's photos
3. **Azure Computer Vision Resource** - For analyzing photo quality
4. **Azure Custom Vision Resource** - For detecting Milo in photos and distinguishing him from other cats
5. **Azure OpenAI Service** - With GPT-4 Vision deployment:
   - **GPT-4 Vision** (gpt-4o or gpt-4-turbo-vision) for analyzing Milo's appearance and generating captions
6. **Black Forest Labs API Key** - [Sign up at Black Forest Labs](https://api.bfl.ml/) for FLUX.2-pro image generation
7. **OpenWeatherMap API Key (optional)** - [Sign up for free](https://openweathermap.org/api) and subscribe to One Call API 3.0 for real-time weather data in captions
8. **Postly Account** - [Sign up at Postly.ai](https://postly.ai/) and obtain API credentials
9. **Azure Functions Core Tools** (for local development) - [Installation guide](https://learn.microsoft.com/azure/azure-functions/functions-run-local)
10. **Python 3.9-3.11** - Azure Functions currently supports Python 3.9, 3.10, and 3.11

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

#### Custom Vision Resource
```bash
# Create Custom Vision training resource
az cognitiveservices account create \
  --name milo-custom-vision \
  --resource-group milo-photos-rg \
  --kind CustomVision.Training \
  --sku F0 \
  --location eastus

# Create Custom Vision prediction resource
az cognitiveservices account create \
  --name milo-custom-vision-prediction \
  --resource-group milo-photos-rg \
  --kind CustomVision.Prediction \
  --sku F0 \
  --location eastus

# Train your Custom Vision model:
# 1. Go to https://www.customvision.ai/
# 2. Create a new Classification project (Multiclass)
# 3. Upload training images with tags: "milo" (Milo only), "emilio" (Emilio only), "both" (both cats), "neither" (no cats)
# 4. Train the model
# 5. Publish the iteration
# 6. Note the Project ID and Iteration Name for configuration
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

# Deploy GPT-4 Vision model (use Azure Portal for this step)
# Go to Azure OpenAI Studio > Deployments > Create new deployment
# Select: gpt-4o (or gpt-4-turbo-vision), Name: gpt-4o
```

#### Black Forest Labs FLUX API
```bash
# Sign up for FLUX API access:
# 1. Go to https://api.bfl.ml/
# 2. Create an account
# 3. Obtain your API key
# 4. Note the API endpoint URL (typically https://api.bfl.ml/v1/flux-pro)
```

#### Function App (Flex Consumption Plan)
```bash
# Create Function App with Flex Consumption plan
az functionapp create \
  --resource-group milo-photos-rg \
  --name milo-photo-poster \
  --storage-account milophotosstg \
  --flexconsumption-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4

# (Optional) Configure instance scaling limits
az functionapp config set \
  --resource-group milo-photos-rg \
  --name milo-photo-poster \
  --minimum-elastic-instance-count 0 \
  --maximum-elastic-instance-count 3
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

#### Option 1: Deploy from local.settings.json (Recommended)

Use the provided deployment script to automatically upload all settings from your `local.settings.json`:

**PowerShell (Windows):**
```powershell
.\deploy-settings.ps1 -ResourceGroup "milo-photos-rg" -FunctionAppName "milo-photo-poster"
```

**Bash (Linux/macOS):**
```bash
chmod +x deploy-settings.sh
./deploy-settings.sh milo-photos-rg milo-photo-poster
```

The script will automatically:
- Read all settings from `local.settings.json`
- Skip Azure-managed settings (FUNCTIONS_WORKER_RUNTIME, etc.)
- Deploy all application settings to your Function App

#### Option 2: Manual Configuration

```bash
az functionapp config appsettings set \
  --name milo-photo-poster \
  --resource-group milo-photos-rg \
  --settings \
    AZURE_STORAGE_CONNECTION_STRING="<connection-string>" \
    BLOB_CONTAINER_NAME="milo-photos" \
    COMPUTER_VISION_ENDPOINT="<endpoint>" \
    COMPUTER_VISION_KEY="<key>" \
    # ... add all other settings
```

## Environment Variables Reference

The following environment variables must be set in Azure Function App Settings or local.settings.json:

### Required Settings
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Storage account connection string
- `BLOB_CONTAINER_NAME` - Container name for photos (default: "milo-photos")
- `COMPUTER_VISION_ENDPOINT` - Azure Computer Vision API endpoint
- `COMPUTER_VISION_KEY` - Computer Vision API key

### Custom Vision (Milo Detection)
- `CUSTOM_VISION_PREDICTION_ENDPOINT` - Azure Custom Vision prediction endpoint
- `CUSTOM_VISION_PREDICTION_KEY` - Custom Vision prediction key
- `CUSTOM_VISION_PROJECT_ID` - Custom Vision project ID
- `CUSTOM_VISION_ITERATION_NAME` - Published iteration name (default: "Iteration1")
- `CUSTOM_VISION_TRAINING_ENDPOINT` - Azure Custom Vision training endpoint (for utility scripts)
- `CUSTOM_VISION_TRAINING_KEY` - Custom Vision training key (for utility scripts)
- `CUSTOM_VISION_PREDICTION_RESOURCE_ID` - Full Azure resource ID for prediction resource (for utility scripts)
- `MILO_CONFIDENCE_THRESHOLD` - Minimum confidence to consider Milo present (default: "0.7")
- `REQUIRE_MILO_IN_PHOTO` - Filter out photos without Milo (default: "true")

### Image Generation (FLUX)
- `OPENAI_IMAGE_MODEL` - Image model name (default: "flux-2")
- `OPENAI_IMAGE_API_KEY` - Black Forest Labs FLUX API key
- `FLUX_API_URL` - FLUX API endpoint (e.g., "https://api.bfl.ml/v1/flux-pro")

### Text Generation (Azure OpenAI GPT-4)
- `OPENAI_TEXT_MODEL` - Text model deployment name (default: "gpt-4o")
- `OPENAI_TEXT_API_KEY` - Azure OpenAI API key
- `OPENAI_TEXT_ENDPOINT` - Azure OpenAI endpoint

### Weather Integration (Optional)
- `WEATHER_API_KEY` - OpenWeatherMap One Call API 3.0 key (optional)
- `WEATHER_LAT` - Latitude for weather data (default: "40.4406" for Pittsburgh)
- `WEATHER_LON` - Longitude for weather data (default: "-79.9959" for Pittsburgh)

### Postly API (Social Media Posting)
- `POSTLY_API_KEY` - Postly API authentication key
- `POSTLY_WORKSPACE_ID` - Postly workspace identifier
- `POSTLY_TARGET_PLATFORMS` - Comma-separated list of Postly account IDs (optional, defaults to "all")
- `POSTLY_BLUESKY_ACCOUNT_ID` - Specific Bluesky account ID (optional)
- `POSTLY_INSTAGRAM_ACCOUNT_ID` - Specific Instagram account ID (optional)

### Function Behavior
- `DAYS_TO_CHECK` - Number of days to look back for photos (default: "7")
- `MAX_PHOTOS_TO_ANALYZE` - Maximum photos to analyze per run (default: "10")
- `POSTED_HISTORY_DAYS` - Days to remember posted photos (default: "30")

### Azure Functions Settings (Auto-configured)
- `FUNCTIONS_WORKER_RUNTIME` - Should be "python"
- `FUNCTIONS_EXTENSION_VERSION` - Should be "~4"
- `WEBSITE_CONTENTAZUREFILECONNECTIONSTRING` - Auto-configured by Azure
- `WEBSITE_CONTENTSHARE` - Auto-configured by Azure
- `APPLICATIONINSIGHTS_CONNECTION_STRING` - Auto-configured if Application Insights is enabled

**Do not commit values/secrets to the repository.**

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

### 7. Code Quality Checks
Run the quality checks locally:

```bash
ruff check .
ruff format . --diff
mypy function_app.py tests
bandit -c pyproject.toml -r .
pytest -m "not integration"
pytest -m integration
```

Enable pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

## Utility Scripts

The `utils/` directory contains helpful scripts for managing and testing the application:

### Photo Management
- **`download_sample_photos.py`** - Download sample photos from Azure Blob Storage for testing
- **`download_negative_cat_images.py`** - Download negative training examples (non-Milo cats) for Custom Vision training
- **`analyze_milo_photos.py`** - Batch analyze photos with Computer Vision and Custom Vision to see scoring and detection results

### Custom Vision Training
- **`test_custom_vision_api.py`** - Test Custom Vision API connectivity and analyze individual photos
  - Useful for debugging Milo detection issues
  - Shows confidence scores for each tag (milo, emilio, both, neither)

### Environment Configuration
- **`convert_azure_to_local.py`** - Convert Azure Function App settings to local.settings.json format
  - Run: `python utils/convert_azure_to_local.py <function-app-name> <resource-group>`
- **`load-env.ps1`** / **`load-env.sh`** - Load environment variables from local.settings.json into your shell
  - PowerShell: `. .\utils\load-env.ps1`
  - Bash: `source utils/load-env.sh`

### Deployment
- **`deploy-settings.ps1`** / **`deploy-settings.sh`** - Deploy local.settings.json to Azure Function App
  - PowerShell: `.\utils\deploy-settings.ps1 -ResourceGroup "rg-name" -FunctionAppName "app-name"`
  - Bash: `./utils/deploy-settings.sh rg-name app-name`
- **`deploy.ps1`** - Full deployment script for code and settings

## Deployment

### Option 1: Deploy via Azure CLI (Recommended for Flex Consumption)
```bash
# From the project root directory
func azure functionapp publish milo-photo-poster
```

**Note for Flex Consumption Plans**: This is currently the most reliable deployment method. The Azure Functions Core Tools handle Flex Consumption deployments correctly.

### Option 2: Deploy via VS Code

**⚠️ Known Issue with Flex Consumption Plans**: The Azure Functions VS Code extension may show "Failed to get status of deployment" errors when deploying to Flex Consumption plans. This is a known limitation. If you encounter this error:

1. Use **Option 1** (Azure CLI) instead, or
2. The deployment may have succeeded despite the error - check the Azure Portal to verify
3. Alternatively, deploy from the terminal within VS Code using `func azure functionapp publish milo-photo-poster`

**If using VS Code extension:**
1. Install the [Azure Functions extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions)
2. Open the project in VS Code
3. Click the Azure icon in the sidebar
4. Sign in to your Azure account
5. Right-click your Function App and select "Deploy to Function App"

### Option 3: Deploy via GitHub Actions (Automated)

The repository includes automated CI/CD and dependency management:

#### CI/CD Pipeline ([`.github/workflows/cicd.yml`](.github/workflows/cicd.yml))
This workflow runs quality checks and automatically deploys to Azure when code is pushed to `main`:

**Quality Checks** (run on all PRs and main pushes):
- ✓ Ruff linting and formatting
- ✓ mypy type checking
- ✓ Bandit security scanning
- ✓ Unit tests
- ✓ Integration tests

**Deployment** (only on main branch pushes, after all checks pass):
- ✓ Uses Azure Functions Core Tools for reliable Flex Consumption deployment
- ✓ Proper Python dependency handling with virtual environment
- ✓ OIDC authentication (no publish profiles needed)
- ✓ Can also be manually triggered via workflow_dispatch

**Note**: Application settings (API keys, connection strings) are **not** included in the deployment. Manage them separately using:
```powershell
.\deploy-settings.ps1
```

#### GitHub Actions Version Updater ([`.github/workflows/update-actions.yml`](.github/workflows/update-actions.yml))
This workflow automatically checks for updates to GitHub Actions used in the repository and creates pull requests with version updates:

- ✓ Runs weekly on Sundays
- ✓ Checks all workflow files for action updates
- ✓ Creates PRs with updated versions
- ✓ Can also be manually triggered

**Setup Required**: Create a Personal Access Token (PAT) for this workflow:

1. Go to GitHub Settings > Developer settings > Personal access tokens > [Fine-grained tokens](https://github.com/settings/tokens?type=beta)
2. Click "Generate new token"
3. Set the following:
   - **Token name**: `GitHub Actions Version Updater`
   - **Repository access**: Only select repositories > `milo-photo-poster`
   - **Repository permissions**:
     - Contents: Read and write
     - Workflows: Read and write
     - Pull requests: Read and write
     - Metadata: Read-only (auto-selected)
4. Click "Generate token" and copy it
5. Go to your repository Settings > Secrets and variables > Actions
6. Click "New repository secret"
7. Name: `GH_ACTIONS_UPDATE_TOKEN`
8. Value: Paste your token
9. Click "Add secret"

To manually trigger an update check:
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "Update GitHub Actions Versions"
4. Click "Run workflow"

#### Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml))
Dependabot automatically monitors and updates Python dependencies:

**Python Dependencies** (requirements.txt):
- ✓ Checks weekly on Sundays
- ✓ Groups minor and patch updates together to reduce PR noise
- ✓ Labels PRs with `dependencies` and `python`

**No setup required** - Dependabot is enabled automatically on GitHub repositories with a `dependabot.yml` file.

**Note**: GitHub Actions updates are handled by the separate GitHub Actions Version Updater workflow above.

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
2. **Duplicate Avoidance**: Filters out photos that have been posted within the last M days (default: 30) to prevent the same photo from being selected repeatedly
3. **Milo Detection**: Uses Azure Custom Vision to identify whether Milo is present in the photo:
   - Classifies photos as: "milo" (Milo only), "emilio" (Emilio only), "both" (both cats), or "neither" (no cats)
   - Filters out photos without Milo if `REQUIRE_MILO_IN_PHOTO` is enabled
   - Caches detection results in blob metadata to avoid repeated API calls
   - Automatically re-analyzes if Custom Vision iteration changes
4. **Computer Vision Analysis**: Each remaining photo is analyzed for:
   - Overall description and confidence
   - Tags (looking for cat-related content)
   - Adult/racy content (filtered out)
   - Image type (prefers photographs over clip art)
   - Color information
5. **Appeal Score Calculation**:
   - Base score from description confidence (0-30 points)
   - Bonus for cat-related tags (0-20 points)
   - Penalty for inappropriate content (-50 points)
   - Penalty for clip art or line drawings (-20/-15 points)
   - Bonus for color images (+10 points)
6. **Selection**: Photo with highest score (minimum 30 points) is selected

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
- Stores the description in a `milo_description.txt` file for reuse
- Example output: "a fluffy orange tabby cat with bold dark stripes, white paws, and bright green eyes"

#### Step 2: Generate Image with FLUX.2-pro
- Randomly selects from 8 different moods: happy, playful, sleepy, curious, gloomy, angry, regal, or cozy
- Incorporates Milo's actual appearance description from GPT-4 Vision into the prompt
- Example prompt: "A high-quality photo of Milo, a fluffy orange tabby cat with bold dark stripes and white paws, looking playful..."
- Uses Black Forest Labs FLUX.2-pro API to generate photorealistic 1024x1024 images
- Optionally includes a reference photo to improve consistency
- Result: AI-generated images that actually resemble Milo, not just a generic cat!

**Why this works:** While FLUX can't directly see photos, GPT-4 Vision can. By using GPT-4 Vision as a "bridge" to describe Milo's appearance, we ensure FLUX generates images that match how Milo really looks. The optional reference image further improves consistency.

### Caption Generation

Captions are dynamically generated using AI to keep content fresh and engaging:

1. **Context Collection**: Gathers current temporal and environmental context including:
   - Day of week (Monday through Sunday) - included 40% of the time for variety
   - Real-time weather conditions via OpenWeatherMap One Call API 3.0 - included 60% of the time when API key is configured
     - Current temperature and weather description (e.g., "clear sky", "light rain")
     - Location-specific data based on WEATHER_LAT and WEATHER_LON settings
   - Notable holidays (always included when applicable) - New Year's Day, Valentine's Day, St. Patrick's Day, April Fool's Day, Halloween, Thanksgiving, Christmas, New Year's Eve

2. **Image Analysis**: Uses Computer Vision description of the selected photo (when available)

3. **AI Caption Generation**: Uses Azure OpenAI GPT-4 to create:
   - Short, witty captions (max 15 words)
   - Grumpy but lovable personality matching Milo's character
   - Varied captions - context elements are probabilistically included to prevent repetitive patterns
   - Real weather-aware references when available (e.g., "it's 45°F and rainy, perfect nap weather")
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
3. **Custom Vision**: Free tier includes 10,000 predictions/month (more than sufficient for daily use)
4. **Azure OpenAI (GPT-4 Vision)**: Pay-per-use; only charged when analyzing Milo's appearance or generating captions
5. **FLUX API**: Pay-per-use from Black Forest Labs; check their pricing at https://api.bfl.ml/pricing
6. **Function App**: Consumption/Flex plan charges only for execution time (minimal cost for daily function)
7. **Application Insights**: Configure sampling to reduce costs (already enabled in host.json)

**Estimated Monthly Cost**: $10-25 depending on AI image generation frequency and API usage

## Troubleshooting

### Deployment Issues

#### "Failed to get status of deployment" (VS Code Extension)
This is a known issue with Flex Consumption plans and the VS Code extension. **Solution:**
- Use `func azure functionapp publish milo-photo-poster` instead
- Or use the `.\deploy.ps1` script
- Check Azure Portal - the deployment may have succeeded despite the error

#### GitHub Actions Deployment Failures
If the GitHub Actions workflow fails:
1. Verify your Azure credentials (client-id, tenant-id, subscription-id) are correctly set in repository secrets
2. Check that the Function App name matches in the workflow file
3. Ensure Azure Functions Core Tools can access your subscription
4. Review the workflow logs in the "Actions" tab on GitHub

#### "No package found" or Build Errors
- Ensure `requirements.txt` is in the repository root
- Verify all dependencies are compatible with Python 3.11
- Check that `.funcignore` isn't excluding necessary files

#### Settings Not Applied After Deployment
Application settings are managed separately from code deployment:
- Run `.\deploy-settings.ps1` after deploying code
- Verify settings in Azure Portal → Function App → Configuration
- Restart the Function App if settings were just updated

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

### Custom Vision Errors (Milo Detection)
- Verify `CUSTOM_VISION_PREDICTION_ENDPOINT`, `CUSTOM_VISION_PREDICTION_KEY`, and `CUSTOM_VISION_PROJECT_ID` are correct
- Ensure the iteration specified in `CUSTOM_VISION_ITERATION_NAME` is published
- Check that the Custom Vision model is trained with at least the tags: "milo", "emilio", "both", "neither"
- Adjust `MILO_CONFIDENCE_THRESHOLD` if too many photos are filtered out (or too few)
- Set `REQUIRE_MILO_IN_PHOTO=false` to disable filtering if you want to post any cat photo
- Use the utility script `utils/test_custom_vision_api.py` to test Custom Vision independently

### Postly API Errors
- Verify API key and workspace ID
- Check Postly API documentation for endpoint changes
- Review response error messages in logs

### AI Image Generation Failures
- Verify `FLUX_API_URL` and `OPENAI_IMAGE_API_KEY` are correctly configured
- Check Black Forest Labs API status and quota limits
- Ensure reference images are accessible (if using reference image feature)
- Review FLUX API response errors in logs for specific issues

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
- Azure Cognitive Services for Computer Vision and Custom Vision capabilities
- Black Forest Labs for FLUX.2-pro image generation
- OpenAI and Microsoft for GPT-4 Vision integration
- Postly for the social media API
