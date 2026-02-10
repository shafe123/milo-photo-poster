# GitHub Copilot Instructions

## Project Overview

This is an Azure Functions application (Python-based) that automatically posts a daily photo of Milo the cat using the Postly API. The function runs on a scheduled timer, analyzes photos from Azure Blob Storage using Computer Vision, and falls back to AI-generated images via DALL-E when needed.

## Technology Stack

- **Runtime**: Python 3.9-3.11
- **Framework**: Azure Functions v4
- **Cloud Services**: 
  - Azure Blob Storage (photo storage)
  - Azure Computer Vision API (image analysis)
  - Azure OpenAI (DALL-E 3 for AI image generation)
  - Postly API (social media posting)
- **Key Libraries**:
  - `azure-functions` - Azure Functions SDK
  - `azure-storage-blob` - Blob storage client
  - `azure-cognitiveservices-vision-computervision` - Computer Vision client
  - `openai` - Azure OpenAI client
  - `requests` - HTTP client
  - `Pillow` - Image processing

## Commands

### Local Development
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure local settings
cp local.settings.json.example local.settings.json
# Edit local.settings.json with your credentials

# Run locally
func start
```

### Deployment
```bash
# Deploy to Azure
func azure functionapp publish milo-photo-poster
```

### Testing
- Currently no automated test suite exists
- Manual testing involves running the function locally and verifying:
  - Blob storage connectivity
  - Computer Vision API integration
  - OpenAI DALL-E image generation
  - Postly API posting

## Code Style and Conventions

### Python Style
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Maximum line length: ~100 characters (for readability)
- Use docstrings for all functions with Args and Returns sections
- Prefer descriptive variable names over short names

### Naming Conventions
- **Functions**: `snake_case` (e.g., `select_best_photo`, `calculate_appeal_score`)
- **Classes**: `PascalCase` (though none currently exist)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MIN_ACCEPTABLE_SCORE`, `DAYS_TO_CHECK`)
- **Variables**: `snake_case`

### Error Handling
- Use try-except blocks for external API calls
- Log errors with descriptive messages using `logging.error()`
- Gracefully handle failures (e.g., fallback to AI generation when no photos found)
- Always include context in error messages

### Logging
- Use Python's `logging` module (already configured by Azure Functions)
- Log levels:
  - `logging.info()` - General progress and status updates
  - `logging.warning()` - Non-critical issues (e.g., skipping a corrupted image)
  - `logging.error()` - Errors that prevent normal operation
- Include relevant context in log messages (e.g., blob names, scores, API endpoints)

## Project Structure

```
milo-photo-poster/
├── .github/              # GitHub configuration
├── function_app.py       # Main Azure Function code (all logic in one file)
├── host.json            # Azure Functions host configuration
├── requirements.txt     # Python dependencies
├── .funcignore         # Files to ignore during deployment
├── .gitignore          # Git ignore patterns
├── local.settings.json.example  # Template for local configuration
└── README.md           # Documentation
```

## Key Functions and Logic

### Main Entry Point
- `daily_milo_post()` - Timer-triggered Azure Function (runs at 10:00 AM UTC daily)
  - Validates configuration
  - Attempts to select best photo from blob storage
  - Falls back to AI generation if needed
  - Posts to Postly API

### Photo Selection Pipeline
1. `get_recent_blobs()` - Retrieves images modified in last N days
2. `analyze_image_quality()` - Uses Computer Vision to analyze each image
3. `calculate_appeal_score()` - Scores images based on quality, cat content, etc.
4. `select_best_photo()` - Returns highest-scoring image above threshold

### Fallback Mechanism
- `generate_ai_image()` - Generates cat photo using DALL-E if no suitable photos found

### Publishing
- `post_to_postly()` - Uploads image and creates post via Postly API

## Environment Variables and Configuration

All configuration via environment variables (set in Azure Function App Settings or `local.settings.json`):

### Required Settings
- `AZURE_STORAGE_CONNECTION_STRING` - Azure Storage account connection string
- `BLOB_CONTAINER_NAME` - Container name for photos (default: "milo-photos")
- `COMPUTER_VISION_ENDPOINT` - Azure Computer Vision API endpoint
- `COMPUTER_VISION_KEY` - Computer Vision API key
- `OPENAI_API_KEY` - Azure OpenAI API key
- `OPENAI_ENDPOINT` - Azure OpenAI endpoint
- `OPENAI_DEPLOYMENT_NAME` - DALL-E deployment name (default: "dall-e-3")
- `POSTLY_API_KEY` - Postly API authentication key
- `POSTLY_WORKSPACE_ID` - Postly workspace identifier
- `POSTLY_TARGET_PLATFORMS` - Comma-separated list of Postly account IDs to post to (optional)
- `DAYS_TO_CHECK` - Number of days to look back for photos (default: 7)

### Never Commit
- API keys, connection strings, or secrets
- `local.settings.json` (contains sensitive credentials)
- `.venv/` directory

## Security and Boundaries

### Prohibited Actions
- **Never** commit secrets, API keys, or connection strings to the repository
- **Never** modify or expose sensitive environment variables
- **Never** introduce security vulnerabilities (SQL injection, XSS, etc.)
- **Never** bypass authentication or authorization checks

### Best Practices
- Store all secrets in Azure Key Vault (see README for setup)
- Use Managed Identities instead of connection strings where possible
- Validate and sanitize all external inputs
- Keep dependencies up to date for security patches
- Follow principle of least privilege for Azure resource access

### Allowed Modifications
- Function logic and business rules
- Image scoring algorithm
- Logging and monitoring
- Documentation
- Dependencies (after security review)

## Testing Guidelines

### When Adding Tests (Future)
- Place tests in a `tests/` directory
- Use `pytest` as the testing framework
- Mock external Azure services (Storage, Computer Vision, OpenAI, Postly)
- Test individual functions in isolation
- Include integration tests for the full pipeline
- Aim for >80% code coverage for critical paths

### Manual Testing Checklist
- [ ] Function starts without errors locally
- [ ] Blob storage connection successful
- [ ] Computer Vision API analyzes images correctly
- [ ] Photo scoring algorithm works as expected
- [ ] AI image generation functions when no photos available
- [ ] Postly API posting succeeds with valid credentials
- [ ] Error handling works for various failure scenarios

## Git Workflow

### Branching
- `main` - Production-ready code
- Feature branches: `feature/description` or `copilot/description`
- Bug fixes: `fix/description`

### Pull Requests
- Include clear description of changes
- Test changes locally before creating PR
- Update README if adding new features or changing configuration
- Ensure no secrets are committed

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (e.g., "Add", "Fix", "Update", "Refactor")
- Keep first line under 50 characters
- Add detailed explanation in body if needed

## Common Tasks

### Adding a New Azure Service Integration
1. Add required SDK to `requirements.txt`
2. Add configuration variables to environment
3. Update `local.settings.json.example` with placeholder
4. Initialize client in `daily_milo_post()` function
5. Add error handling and logging
6. Update README with setup instructions

### Modifying the Scoring Algorithm
1. Locate `calculate_appeal_score()` function
2. Adjust scoring weights or add new criteria
3. Test with sample images
4. Document scoring changes in code comments
5. Update README if scoring logic significantly changes

### Changing the Schedule
1. Modify the `schedule` parameter in `@app.timer_trigger()` decorator
2. Format: `"sec min hour day month day-of-week"` (cron expression)
3. Test schedule locally with adjusted timer
4. Document schedule change in README

## Additional Notes

- This is a serverless application - it only runs on schedule or when manually triggered
- The function is idempotent - running it multiple times in a day won't cause issues (Postly handles duplicates)
- Blob storage should contain actual photos of Milo for best results
- AI fallback ensures posting continues even without uploaded photos
- Monitor Azure costs, especially for Computer Vision API calls and OpenAI usage
