"""
One-time script to analyze 3 Milo photos and generate a detailed description.
This description will be hardcoded into the AI generation function.
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from openai import AzureOpenAI


# Load environment variables from local.settings.json
def load_local_settings(path="../local.settings.json"):
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        settings = json.load(f)
    values = settings.get("Values", {})
    for k, v in values.items():
        os.environ[k] = v


load_local_settings()

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.environ.get("BLOB_CONTAINER_NAME", "milo-photos")
OPENAI_TEXT_API_KEY = os.environ.get("OPENAI_TEXT_API_KEY")
OPENAI_TEXT_ENDPOINT = os.environ.get("OPENAI_TEXT_ENDPOINT")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")


def get_milo_photos(container_client):
    """Get all image blobs that contain Milo (based on metadata)"""
    milo_blobs = []
    all_blobs = []

    for blob in container_client.list_blobs(include=["metadata"]):
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            all_blobs.append(blob)

            # Check if blob has milo_detected metadata set to true
            if blob.metadata and blob.metadata.get("milo_detected") == "true":
                milo_blobs.append(blob)
                print(f"  ✓ {blob.name} - Milo confirmed")

    print(f"\nFound {len(all_blobs)} total photos, {len(milo_blobs)} with Milo detected")

    # If no photos have milo_detected metadata, just use all photos
    if not milo_blobs and all_blobs:
        print("  No photos with milo_detected metadata, using all available photos")
        return all_blobs

    return milo_blobs


def analyze_milo_photos():
    """Analyze 3 Milo photos to get detailed description"""

    print("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

    print("Fetching photos with Milo...")
    milo_blobs = get_milo_photos(container_client)

    if not milo_blobs:
        print("No Milo photos found!")
        return None

    # Select specific photos by name
    target_blob_names = [
        "IMG20221104163035.jpg",  # Photo 3
        "IMG20250303180114.jpg",  # Photo 13
        "IMG20250325072532.jpg",  # Photo 14
    ]

    photos_to_analyze = [blob for blob in milo_blobs if blob.name in target_blob_names]

    if len(photos_to_analyze) != len(target_blob_names):
        print(f"Warning: Found {len(photos_to_analyze)} photos, expected {len(target_blob_names)}")

    print(f"\nAnalyzing {len(photos_to_analyze)} selected Milo photos:")

    image_urls = []
    for i, blob in enumerate(photos_to_analyze):
        print(f"  {i + 1}. {blob.name} ({blob.size / 1024:.1f} KB)")

        blob_client = container_client.get_blob_client(blob.name)

        # Generate SAS token for temporary read access (1 hour)
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=BLOB_CONTAINER_NAME,
            blob_name=blob.name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1),
        )
        blob_url = f"{blob_client.url}?{sas_token}"
        image_urls.append(blob_url)

    # Initialize OpenAI client
    print("\nConnecting to Azure OpenAI...")
    text_client = AzureOpenAI(
        api_key=OPENAI_TEXT_API_KEY,
        api_version="2024-02-01",
        azure_endpoint=OPENAI_TEXT_ENDPOINT,
    )

    # Build GPT-4 Vision message
    content = [
        {
            "type": "text",
            "text": (
                "You are analyzing photos of a cat named Milo. "
                "Please provide a comprehensive, detailed physical description of this cat "
                "that could be used to generate highly accurate similar images. "
                "\n\nInclude ALL of the following details:"
                "\n- Fur color and specific pattern (e.g., 'light gray tabby with darker charcoal stripes', 'orange and white bicolor', 'solid black')"
                "\n- Fur texture and length (short, medium, long, fluffy, sleek)"
                "\n- Eye color and shape (round, almond-shaped, bright green, amber)"
                "\n- Distinctive facial features (white chin, pink nose, facial markings)"
                "\n- Body type and size (stocky, slender, large, petite)"
                "\n- Any unique markings or features (chest patches, paw colors, ear tufts, tail patterns)"
                "\n\nProvide 4-6 detailed sentences capturing all visual aspects. "
                "Start with: 'Milo is a [description]...'"
            ),
        }
    ]

    # Add images
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    print(f"Calling GPT-4 Vision ({OPENAI_TEXT_MODEL}) to analyze photos...")

    # Call GPT-4 Vision
    response = text_client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing and describing cat appearances in precise detail for AI image generation.",
            },
            {"role": "user", "content": content},
        ],
        max_completion_tokens=500,
    )

    description = response.choices[0].message.content.strip()

    print("\n" + "=" * 80)
    print("MILO'S DETAILED DESCRIPTION:")
    print("=" * 80)
    print(description)
    print("=" * 80)
    print(f"\nLength: {len(description)} characters")
    print("\nCopy this description and use it in the generate_ai_image function!")

    return description


if __name__ == "__main__":
    analyze_milo_photos()
