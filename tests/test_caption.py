import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import AzureOpenAI
from function_app import generate_witty_caption, get_current_context

# Load environment variables from local.settings.json
def load_local_settings(path="local.settings.json"):
    with open(path, "r") as f:
        settings = json.load(f)
    values = settings.get("Values", {})
    for k, v in values.items():
        os.environ[k] = v

load_local_settings()

# Load environment variables for text model
TEXT_API_KEY = os.environ.get("OPENAI_TEXT_API_KEY")
TEXT_ENDPOINT = os.environ.get("OPENAI_TEXT_ENDPOINT")
TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")

# Initialize AzureOpenAI client for text
text_client = AzureOpenAI(
    api_key=TEXT_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=TEXT_ENDPOINT
)

# Get context for caption
context = get_current_context()

# Optionally provide an image description
image_description = "Milo is looking grumpy on a Monday morning."

# Generate witty caption
caption = generate_witty_caption(
    text_client,
    text_model=TEXT_MODEL,
    context=context,
    image_description=image_description
)

print("Generated Caption:", caption)
