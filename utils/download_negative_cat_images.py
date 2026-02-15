"""
Download cat images from online sources to use as negative training examples
for Custom Vision (images that are NOT Milo or Emilio).

This script downloads cat images and optionally uploads them to Azure Blob Storage
with a "negative-examples" prefix for easier organization.
"""

import os
import sys
import requests
from pathlib import Path
import time
from typing import List
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def download_from_unsplash(count: int = 20, api_key: str = None) -> List[str]:
    """
    Download cat images from Unsplash API.

    Get free API key from: https://unsplash.com/developers
    """
    if not api_key:
        print("⚠ Unsplash API key not provided. Skipping Unsplash.")
        return []

    downloaded = []
    print(f"\n📥 Downloading {count} images from Unsplash...")

    for i in range(count):
        try:
            # Unsplash random cat photo endpoint
            url = "https://api.unsplash.com/photos/random"
            params = {"query": "cat", "orientation": "landscape", "client_id": api_key}

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            image_url = data["urls"]["regular"]  # ~1080px width
            image_id = data["id"]

            # Download the actual image
            img_response = requests.get(image_url)
            img_response.raise_for_status()

            filename = f"unsplash_{image_id}.jpg"
            downloaded.append((filename, img_response.content))

            print(f"  ✓ Downloaded {i + 1}/{count}: {filename}")
            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"  ✗ Error downloading image {i + 1}: {e}")

    return downloaded


def download_from_pexels(count: int = 20, api_key: str = None) -> List[str]:
    """
    Download cat images from Pexels API.

    Get free API key from: https://www.pexels.com/api/
    """
    if not api_key:
        print("⚠ Pexels API key not provided. Skipping Pexels.")
        return []

    downloaded = []
    print(f"\n📥 Downloading {count} images from Pexels...")

    try:
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        params = {
            "query": "cat",
            "per_page": min(count, 80),  # Pexels max 80 per page
            "orientation": "landscape",
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        photos = data.get("photos", [])

        for i, photo in enumerate(photos[:count]):
            try:
                image_url = photo["src"]["large"]  # ~1280px width
                image_id = photo["id"]

                # Download the actual image
                img_response = requests.get(image_url)
                img_response.raise_for_status()

                filename = f"pexels_{image_id}.jpg"
                downloaded.append((filename, img_response.content))

                print(f"  ✓ Downloaded {i + 1}/{count}: {filename}")
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"  ✗ Error downloading image {i + 1}: {e}")

    except Exception as e:
        print(f"  ✗ Error fetching from Pexels: {e}")

    return downloaded


def download_from_wikimedia_commons(count: int = 20) -> List[str]:
    """
    Download cat images from Wikimedia Commons (no API key needed).
    """
    downloaded = []
    print(f"\n📥 Downloading {count} images from Wikimedia Commons...")

    # Use Wikimedia Commons API to search for cat images
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": "cat",
            "gsrnamespace": "6",  # File namespace
            "gsrlimit": count * 2,  # Get more to filter
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 1024,
        }

        headers = {
            "User-Agent": "MiloPhotoPoster/1.0 (Custom Vision Training; ethan@example.com)"
        }

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        downloaded_count = 0
        for page_id, page in pages.items():
            if downloaded_count >= count:
                break

            try:
                imageinfo = page.get("imageinfo", [{}])[0]
                image_url = imageinfo.get("thumburl") or imageinfo.get("url")

                if not image_url or not image_url.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    continue

                # Download the image
                img_response = requests.get(image_url)
                img_response.raise_for_status()

                # Create filename from hash to avoid duplicates
                img_hash = hashlib.md5(img_response.content).hexdigest()[:12]
                ext = image_url.split(".")[-1].split("?")[0]
                filename = f"wikimedia_{img_hash}.{ext}"

                downloaded.append((filename, img_response.content))
                downloaded_count += 1

                print(f"  ✓ Downloaded {downloaded_count}/{count}: {filename}")
                time.sleep(0.5)

            except Exception as e:
                print(f"  ✗ Error downloading image: {e}")
                continue

    except Exception as e:
        print(f"  ✗ Error fetching from Wikimedia: {e}")

    return downloaded


def download_from_public_urls(count: int = 20) -> List[str]:
    """
    Download cat images from curated public URLs.
    These are freely available cat images from various sources.
    """
    downloaded = []
    print(f"\n📥 Downloading {count} images from public sources...")

    # Curated list of public cat image URLs
    public_cat_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/1200px-Kittyply_edit1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1200px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/1200px-Cat_poster_1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Six_weeks_old_cat_%28aka%29.jpg/1200px-Six_weeks_old_cat_%28aka%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Red_Kitten_01.jpg/1200px-Red_Kitten_01.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/RedCat_8727.jpg/1200px-RedCat_8727.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sheba1.JPG/1200px-Sheba1.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Lyra_Kedi.jpg/1200px-Lyra_Kedi.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Collage_of_Six_Cats-02.jpg/1200px-Collage_of_Six_Cats-02.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Patryk_Dziczek_Cat.jpg/1200px-Patryk_Dziczek_Cat.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Angora_cat_-_Sadie.jpg/1200px-Angora_cat_-_Sadie.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Cat_March_2010-1.jpg/1200px-Cat_March_2010-1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/June_odd-eyed-cat_cropped.jpg/1200px-June_odd-eyed-cat_cropped.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Youngkitten.JPG/1200px-Youngkitten.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/1200px-Siam_lilacpoint.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Gustav_chocolate.jpg/1200px-Gustav_chocolate.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Tord%C3%A9schirolais.jpg/1200px-Tord%C3%A9schirolais.jpg",
    ]

    for i, url in enumerate(public_cat_urls[:count]):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Create filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in ["jpg", "jpeg", "png"]:
                ext = "jpg"
            filename = f"public_{url_hash}.{ext}"

            downloaded.append((filename, response.content))
            print(f"  ✓ Downloaded {i + 1}/{count}: {filename}")
            time.sleep(0.3)

        except Exception as e:
            print(f"  ✗ Error downloading image {i + 1}: {e}")

    return downloaded


def save_images_locally(images: List[tuple], output_dir: str = "negative_cat_images"):
    """Save downloaded images to local directory."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n💾 Saving {len(images)} images to {output_path.absolute()}...")

    for filename, content in images:
        filepath = output_path / filename
        with open(filepath, "wb") as f:
            f.write(content)
        print(f"  ✓ Saved {filename}")

    print(f"\n✅ Saved {len(images)} images to {output_path.absolute()}")


def upload_to_blob_storage(
    images: List[tuple],
    connection_string: str,
    container_name: str,
    prefix: str = "negative-examples",
):
    """Upload images to Azure Blob Storage."""
    from azure.storage.blob import BlobServiceClient

    print(f"\n☁️  Uploading {len(images)} images to Azure Blob Storage...")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Ensure container exists
    try:
        container_client.create_container()
        print(f"  Created container: {container_name}")
    except Exception:
        pass  # Container already exists

    for filename, content in images:
        blob_name = f"{prefix}/{filename}"
        blob_client = container_client.get_blob_client(blob_name)

        try:
            blob_client.upload_blob(content, overwrite=True)
            print(f"  ✓ Uploaded {blob_name}")
        except Exception as e:
            print(f"  ✗ Error uploading {blob_name}: {e}")

    print(f"\n✅ Uploaded {len(images)} images to {container_name}/{prefix}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download cat images for negative training examples"
    )
    parser.add_argument(
        "--count", type=int, default=30, help="Total number of images to download"
    )
    parser.add_argument("--unsplash-key", help="Unsplash API key (optional)")
    parser.add_argument("--pexels-key", help="Pexels API key (optional)")
    parser.add_argument(
        "--output-dir", default="negative_cat_images", help="Local output directory"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to Azure Blob Storage"
    )
    parser.add_argument(
        "--connection-string",
        help="Azure Storage connection string (from env if not provided)",
    )
    parser.add_argument(
        "--container", default="milo-photos", help="Blob container name"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Cat Image Downloader for Custom Vision Negative Examples")
    print("=" * 80)

    all_images = []

    # Try Wikimedia Commons first (no API key needed)
    try:
        wikimedia_count = min(args.count // 3, 10)
        wikimedia_images = download_from_wikimedia_commons(wikimedia_count)
        all_images.extend(wikimedia_images)
    except Exception as e:
        print(f"  ⚠ Wikimedia download failed: {e}")

    # Download from public URLs
    remaining = args.count - len(all_images)
    if remaining > 0:
        public_images = download_from_public_urls(min(remaining, 20))
        all_images.extend(public_images)

    # Download from Pexels if API key provided
    if args.pexels_key:
        remaining = args.count - len(all_images)
        if remaining > 0:
            pexels_images = download_from_pexels(min(remaining, 20), args.pexels_key)
            all_images.extend(pexels_images)

    # Download from Unsplash if API key provided
    if args.unsplash_key:
        remaining = args.count - len(all_images)
        if remaining > 0:
            unsplash_images = download_from_unsplash(remaining, args.unsplash_key)
            all_images.extend(unsplash_images)

    if not all_images:
        print("\n❌ No images downloaded. Check your API keys or internet connection.")
        return

    # Save locally
    save_images_locally(all_images, args.output_dir)

    # Upload to blob storage if requested
    if args.upload:
        connection_string = args.connection_string or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        if not connection_string:
            print("\n⚠ Azure Storage connection string not provided. Skipping upload.")
            print(
                "  Set AZURE_STORAGE_CONNECTION_STRING environment variable or use --connection-string"
            )
        else:
            upload_to_blob_storage(all_images, connection_string, args.container)

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("\nNext steps:")
    print(f"  1. Review images in: {Path(args.output_dir).absolute()}")
    print("  2. Go to https://www.customvision.ai/")
    print("  3. Open your project: milo-description")
    print(
        "  4. Upload images and tag them as 'Negative' (or create a new tag like 'other-cats')"
    )
    print("  5. Train a new iteration")
    print("=" * 80)


if __name__ == "__main__":
    main()
