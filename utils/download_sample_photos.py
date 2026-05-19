"""
Download the 3 Milo photos being analyzed
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from azure.storage.blob import BlobServiceClient


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


def download_sample_photos():
    """Download the first 3 Milo photos"""

    print("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

    # Get Milo photos
    milo_blobs = []
    for blob in container_client.list_blobs(include=["metadata"]):
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
            if blob.metadata and blob.metadata.get("milo_detected") == "true":
                milo_blobs.append(blob)

    # Download 10 photos at a time for review
    output_dir = os.path.join(os.path.dirname(__file__), "..", "sample_milo_photos")
    os.makedirs(output_dir, exist_ok=True)

    start_index = 10  # Start from photo 10
    num_photos = 10
    photo_indices = list(range(start_index, min(start_index + num_photos, len(milo_blobs))))

    print(
        f"Downloading {len(photo_indices)} photos (#{start_index + 1} to #{photo_indices[-1] + 1}):\n"
    )

    for idx in photo_indices:
        blob = milo_blobs[idx]
        print(f"Photo {idx + 1}: {blob.name}")
        blob_client = container_client.get_blob_client(blob.name)

        # Use original index in filename
        output_path = os.path.join(output_dir, f"milo_{idx + 1}_{blob.name}")

        with open(output_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

        print(f"  Saved to: {output_path}\n")

    print(f"\nPhotos saved to: {output_dir}")


if __name__ == "__main__":
    download_sample_photos()
