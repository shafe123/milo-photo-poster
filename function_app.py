"""
Milo Photo Poster - Azure Function
Automatically posts a daily photo of Milo the cat using the Postly API.
"""

import os
import logging
import io
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import calendar

import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobProperties, generate_blob_sas, BlobSasPermissions, ContentSettings
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI
import requests
from PIL import Image

app = func.FunctionApp()

# Configure logging - quiet Azure SDK verbose logging
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.environ.get("BLOB_CONTAINER_NAME", "milo-photos")
COMPUTER_VISION_ENDPOINT = os.environ.get("COMPUTER_VISION_ENDPOINT")
COMPUTER_VISION_KEY = os.environ.get("COMPUTER_VISION_KEY")

# Separate API keys and endpoints for image and text models
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "flux-2")
OPENAI_IMAGE_API_KEY = os.environ.get("OPENAI_IMAGE_API_KEY", None)
OPENAI_IMAGE_ENDPOINT = os.environ.get("OPENAI_IMAGE_ENDPOINT", None)

OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o")
OPENAI_TEXT_API_KEY = os.environ.get("OPENAI_TEXT_API_KEY", None)
OPENAI_TEXT_ENDPOINT = os.environ.get("OPENAI_TEXT_ENDPOINT", None)
POSTLY_API_KEY = os.environ.get("POSTLY_API_KEY")
POSTLY_WORKSPACE_ID = os.environ.get("POSTLY_WORKSPACE_ID")
POSTLY_TARGET_PLATFORMS = os.environ.get("POSTLY_TARGET_PLATFORMS", "all")  # Comma-separated account IDs
POSTLY_BLUESKY_ACCOUNT_ID = os.environ.get("POSTLY_BLUESKY_ACCOUNT_ID", None)  # Bluesky account ID (set to actual account ID from Postly)
POSTLY_INSTAGRAM_ACCOUNT_ID = os.environ.get("POSTLY_INSTAGRAM_ACCOUNT_ID", None)  # Instagram account ID (set to actual account ID from Postly)
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "7"))
MAX_PHOTOS_TO_ANALYZE = int(os.environ.get("MAX_PHOTOS_TO_ANALYZE", "10"))  # Limit to avoid rate limits
POSTED_HISTORY_DAYS = int(os.environ.get("POSTED_HISTORY_DAYS", "30"))  # Days to remember posted photos

# Constants
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
MIN_ACCEPTABLE_SCORE = 30  # Minimum photo appeal score to accept (0-100 scale)
POSTED_METADATA_KEY = "posted_date"  # Metadata key for tracking when a photo was posted
MAX_IMAGE_SIZE_BYTES = 1024 * 1024 * 0.90  # 1 MB - Safe size for all platforms, with buffer space
MAX_IMAGE_DIMENSION = 4096  # Maximum dimension to downsize images to

# Caption generation constants
CAPTION_PREFIX = "Daily Milo! 😾"  # Grumpy cat emoji to match Milo's personality
CAPTION_HASHTAGS = "#Milo #Cats #GrumpyCat"
CAPTION_MAX_TOKENS = 100  # Maximum tokens for GPT caption generation, current model does not support max tokens
CAPTION_TEMPERATURE = 1.0  # Temperature for caption creativity (0.0-1.0), current model only supports 1.0


def get_recent_blobs(container_client, days: int) -> List[BlobProperties]:
    """
    Get all blobs modified in the last N days from the container.
    
    Args:
        container_client: Azure Blob Container client
        days: Number of days to look back
        
    Returns:
        List of blob properties for recent blobs
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_blobs = []
    
    try:
        for blob in container_client.list_blobs():
            if blob.last_modified.replace(tzinfo=None) >= cutoff_date:
                # Only include image files
                if blob.name.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    recent_blobs.append(blob)
    except Exception as e:
        logging.error(f"Error listing blobs: {str(e)}")
        raise
    
    logging.info(f"Found {len(recent_blobs)} recent image(s) in blob storage")
    return recent_blobs


def is_blob_posted(blob_client, history_days: int) -> bool:
    """
    Check if a blob has been posted recently.
    
    Args:
        blob_client: Azure Blob client for the specific blob
        history_days: Number of days to consider a photo as "recently posted"
        
    Returns:
        True if blob was posted within history_days, False otherwise
    """
    try:
        properties = blob_client.get_blob_properties()
        metadata = properties.metadata
        
        if POSTED_METADATA_KEY not in metadata:
            return False
        
        posted_date_str = metadata[POSTED_METADATA_KEY]
        # Parse the ISO format date and ensure it's timezone-naive for comparison
        posted_date = datetime.fromisoformat(posted_date_str)
        if posted_date.tzinfo is not None:
            posted_date = posted_date.replace(tzinfo=None)
        
        cutoff_date = datetime.utcnow() - timedelta(days=history_days)
        
        return posted_date >= cutoff_date
    except Exception as e:
        logging.warning(f"Error checking posted status: {str(e)}")
        # If we can't determine, assume it hasn't been posted to be safe
        return False


def mark_blob_as_posted(blob_client) -> None:
    """
    Mark a blob as posted by setting metadata with the current timestamp.
    
    Args:
        blob_client: Azure Blob client for the specific blob
    """
    try:
        properties = blob_client.get_blob_properties()
        metadata = properties.metadata or {}
        metadata[POSTED_METADATA_KEY] = datetime.utcnow().isoformat()
        
        blob_client.set_blob_metadata(metadata)
        logging.info(f"Marked blob as posted: {blob_client.blob_name}")
    except Exception as e:
        logging.error(f"Error marking blob as posted: {str(e)}")


def downsize_image_if_needed(blob_client, max_size_bytes: int = MAX_IMAGE_SIZE_BYTES, 
                              max_dimension: int = MAX_IMAGE_DIMENSION) -> bool:
    """
    Check if an image is too large and downsize it if needed.
    Overwrites the original blob with the downsized version.
    
    Args:
        blob_client: Azure Blob Storage client for the image
        max_size_bytes: Maximum allowed file size in bytes
        max_dimension: Maximum width or height to resize to
        
    Returns:
        True if image was downsized, False otherwise
    """
    try:
        # Get blob properties to check size
        properties = blob_client.get_blob_properties()
        current_size = properties.size
        
        # If image is under the limit, no need to downsize
        if current_size <= max_size_bytes:
            return False
            
        logging.info(f"Image {blob_client.blob_name} is {current_size / 1024 / 1024:.2f} MB, downsizing...")
        
        # Download the image
        image_data = blob_client.download_blob().readall()
        image = Image.open(io.BytesIO(image_data))
        
        # Calculate new dimensions while maintaining aspect ratio
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            # Calculate scaling factor
            scale = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image using high-quality Lanczos filter
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logging.info(f"Resized from {width}x{height} to {new_width}x{new_height}")
        
        # Save to bytes buffer with progressive quality reduction until under size limit
        quality = 85  # Start with high quality
        while quality > 20:  # Don't go below quality 20
            output = io.BytesIO()
            
            # Save as JPEG (most efficient compression)
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert images with transparency to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
            
            image.save(output, format='JPEG', quality=quality, optimize=True)
            output_size = output.tell()
            
            if output_size <= max_size_bytes:
                # Size is acceptable, upload the downsized image
                output.seek(0)
                blob_client.upload_blob(output, overwrite=True, content_settings=ContentSettings(content_type='image/jpeg'))
                logging.info(f"Downsized image from {current_size / 1024 / 1024:.2f} MB to {output_size / 1024 / 1024:.2f} MB (quality={quality})")
                return True
            
            # Reduce quality for next iteration
            quality -= 5
        
        # If we get here, even at lowest quality the image is too large
        # This shouldn't happen in practice with reasonable max_dimension
        logging.warning(f"Could not downsize {blob_client.blob_name} to under {max_size_bytes} bytes")
        return False
        
    except Exception as e:
        logging.error(f"Error downsizing image {blob_client.blob_name}: {str(e)}")
        return False



def analyze_image_quality(cv_client: ComputerVisionClient, image_url: str) -> Dict[str, Any]:
    """
    Analyze image using Azure Computer Vision API.
    
    Args:
        cv_client: Computer Vision client
        image_url: URL of the image to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        features = [
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.adult,
            VisualFeatureTypes.color,
            VisualFeatureTypes.image_type
        ]
        
        analysis = cv_client.analyze_image(image_url, visual_features=features)
        
        return {
            'description': analysis.description.captions[0].text if analysis.description.captions else '',
            'confidence': analysis.description.captions[0].confidence if analysis.description.captions else 0,
            'tags': [(tag.name, tag.confidence) for tag in analysis.tags],
            'is_adult_content': analysis.adult.is_adult_content,
            'is_racy_content': analysis.adult.is_racy_content,
            'adult_score': analysis.adult.adult_score,
            'racy_score': analysis.adult.racy_score,
            'dominant_colors': analysis.color.dominant_colors,
            'is_bw': analysis.color.is_bw_img,
            'is_clip_art': analysis.image_type.clip_art_type,
            'is_line_drawing': analysis.image_type.line_drawing_type
        }
    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        raise


def calculate_appeal_score(analysis: Dict[str, Any]) -> float:
    """
    Calculate an appeal score for the image based on analysis results.
    
    Args:
        analysis: Analysis results from Computer Vision API
        
    Returns:
        Appeal score (0-100)
    """
    score = 0.0
    
    # Base score from description confidence
    score += analysis['confidence'] * 30
    
    # Bonus for cat-related tags
    cat_tags = ['cat', 'kitten', 'feline', 'pet', 'animal', 'mammal']
    for tag_name, tag_confidence in analysis['tags']:
        if any(cat_word in tag_name.lower() for cat_word in cat_tags):
            score += tag_confidence * 20
    
    # Penalty for adult/racy content
    if analysis['is_adult_content'] or analysis['is_racy_content']:
        score -= 50
    
    # Penalty for clip art or line drawings (we want real photos)
    if analysis['is_clip_art'] > 2:  # Scale is 0-3
        score -= 20
    if analysis['is_line_drawing'] > 0:  # Scale is 0-1
        score -= 15
    
    # Bonus for color images
    if not analysis['is_bw']:
        score += 10
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))


def select_best_photo(blob_service_client: BlobServiceClient, 
                     cv_client: ComputerVisionClient,
                     container_name: str,
                     days: int) -> Optional[Tuple[bytes, str, str]]:
    """
    Select the best photo from blob storage based on appeal score.
    
    Args:
        blob_service_client: Azure Blob Storage client
        cv_client: Computer Vision client
        container_name: Name of the blob container
        days: Number of days to look back
        
    Returns:
        Tuple of (image_bytes, blob_name, description) or None if no suitable photo found
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        recent_blobs = get_recent_blobs(container_client, days)
        
        if not recent_blobs:
            logging.info("No recent photos found in blob storage")
            return None
        
        # Filter out recently posted photos to avoid duplicates
        unposted_blobs = []
        for blob in recent_blobs:
            blob_client = container_client.get_blob_client(blob.name)
            if not is_blob_posted(blob_client, POSTED_HISTORY_DAYS):
                unposted_blobs.append(blob)
        
        logging.info(f"Found {len(unposted_blobs)} unposted photos out of {len(recent_blobs)} recent photos")
        
        if not unposted_blobs:
            logging.info("All recent photos have already been posted")
            return None
        
        # Randomly sample photos to analyze (to avoid rate limits)
        blobs_to_analyze = unposted_blobs
        if len(unposted_blobs) > MAX_PHOTOS_TO_ANALYZE:
            blobs_to_analyze = random.sample(unposted_blobs, MAX_PHOTOS_TO_ANALYZE)
            logging.info(f"Randomly selected {MAX_PHOTOS_TO_ANALYZE} photos from {len(unposted_blobs)} available photos")
        
        best_blob = None
        best_score = -1
        best_analysis = None
        
        for blob in blobs_to_analyze:
            try:
                # Get blob client and generate SAS URL for Computer Vision API access
                blob_client = container_client.get_blob_client(blob.name)
                
                # Downsize the image if it's too large for Computer Vision API
                downsize_image_if_needed(blob_client)
                
                # Generate SAS token for temporary read access (1 hour)
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_name,
                    blob_name=blob.name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)
                )
                blob_url = f"{blob_client.url}?{sas_token}"
                
                # Analyze the image
                logging.info(f"Analyzing blob: {blob.name}")
                analysis = analyze_image_quality(cv_client, blob_url)
                
                # Calculate appeal score
                score = calculate_appeal_score(analysis)
                logging.info(f"Blob {blob.name} scored: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_blob = blob
                    best_analysis = analysis
                    
            except Exception as e:
                logging.warning(f"Error processing blob {blob.name}: {str(e)}")
                continue
        
        if best_blob and best_score > MIN_ACCEPTABLE_SCORE:
            logging.info(f"Selected blob: {best_blob.name} with score {best_score:.2f}")
            blob_client = container_client.get_blob_client(best_blob.name)
            image_data = blob_client.download_blob().readall()
            description = best_analysis.get('description', '') if best_analysis else ''
            return (image_data, best_blob.name, description)
        else:
            logging.info(f"No photo met the minimum quality threshold ({MIN_ACCEPTABLE_SCORE})")
            return None
            
    except Exception as e:
        logging.error(f"Error selecting best photo: {str(e)}")
        return None


def extract_milo_characteristics(blob_service_client: BlobServiceClient,
                                text_client: AzureOpenAI,
                                text_model: str,
                                container_name: str) -> str:
    """
    Analyze existing Milo photos using GPT-4 Vision to extract detailed visual characteristics.
    
    Args:
        blob_service_client: Azure Blob Storage client
        openai_client: Azure OpenAI client (for GPT-4 Vision)
        gpt4v_deployment: Name of the GPT-4 Vision deployment
        container_name: Name of the blob container
        
    Returns:
        String description of Milo's visual characteristics from GPT-4 Vision
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        # Get recent photos (last 30 days for a good sample)
        recent_blobs = get_recent_blobs(container_client, days=30)
        
        if not recent_blobs:
            logging.info("No photos found to analyze Milo's characteristics, using default description")
            return "an adorable cat"
        
        # Analyze up to 3 recent photos with GPT-4 Vision
        max_to_analyze = min(3, len(recent_blobs))
        image_urls: list[str] = []
        
        for blob in recent_blobs[:max_to_analyze]:
            try:
                blob_client = container_client.get_blob_client(blob.name)
                
                # Generate SAS token for temporary read access (1 hour)
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_name,
                    blob_name=blob.name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)
                )
                blob_url = f"{blob_client.url}?{sas_token}"
                image_urls.append(blob_url)
                
            except Exception as e:
                logging.warning(f"Error preparing blob {blob.name} for GPT-4 Vision: {str(e)}")
                continue
        
        if not image_urls:
            logging.info("Could not prepare any photos for analysis, using default description")
            return "an adorable cat"
        
        # Build GPT-4 Vision message with multiple images
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "You are analyzing photos of a cat named Milo. "
                    "Please provide a detailed physical description of this cat that could be used "
                    "to generate similar images. Focus on: fur color and pattern (e.g., orange tabby with "
                    "dark stripes, calico, solid gray, etc.), fur length, eye color, distinctive markings, "
                    "body type, and any unique features. "
                    "Be specific and detailed, but concise (2-3 sentences max). "
                    "Format your response as: 'a [description] cat' (e.g., 'a fluffy orange tabby cat "
                    "with white paws and green eyes')."
                )
            }
        ]
        
        # Add up to 3 images to the request
        for url in image_urls[:3]:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        logging.info(f"Analyzing {len(image_urls)} photos with GPT-4 Vision to extract Milo's characteristics")
        
        # Call GPT-4 Vision
        response = text_client.chat.completions.create(
            model=text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and describing cat appearances for image generation."
                },
                {
                    "role": "user",
                    "content": content
                } # type: ignore
            ],
            max_tokens=300
        )
        
        description = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        # Clean up the description if needed
        if not description:
            description = "an adorable gray cat with a grumpy looking face"
        if not description.startswith("a ") and not description.startswith("an "):
            description = f"an adorable {description}"
        
        logging.info(f"GPT-4 Vision extracted Milo's characteristics: {description}")
        return description
        
    except Exception as e:
        logging.error(f"Error extracting Milo's characteristics with GPT-4 Vision: {str(e)}")
        return "an adorable cat"


def select_mood_and_prompt(milo_description: str = "an adorable cat") -> Tuple[str, str]:
    """
    Randomly select a mood and generate a corresponding prompt for Milo's AI image.
    
    Args:
        milo_description: Visual description of Milo's appearance (from photo analysis)
    
    Returns:
        Tuple of (mood, prompt) for image generation
    """
    moods = {
        "happy": (
            f"A high-quality, professional photo of Milo, {milo_description}, looking happy and content. "
            "Milo has a cheerful expression with bright eyes and relaxed posture. "
            "The photo captures Milo in a joyful moment, perhaps with a slight smile or playful demeanor. "
            "Natural lighting, sharp focus, photorealistic style."
        ),
        "playful": (
            f"A high-quality, professional photo of Milo, {milo_description}, in a playful mood. "
            "Milo is captured mid-play, showing energetic and spirited behavior. "
            "Perhaps Milo is batting at a toy, pouncing, or in a playful stance with alert, mischievous eyes. "
            "Natural lighting, action captured with sharp focus, photorealistic style."
        ),
        "sleepy": (
            f"A high-quality, professional photo of Milo, {milo_description}, looking sleepy and relaxed. "
            "Milo is resting peacefully, maybe with half-closed eyes or curled up in a cozy position. "
            "The photo captures a serene, drowsy moment showing Milo's calm and tranquil side. "
            "Soft, warm lighting, sharp focus, photorealistic style."
        ),
        "curious": (
            f"A high-quality, professional photo of Milo, {milo_description}, looking curious and inquisitive. "
            "Milo has wide, attentive eyes and alert ears, focused on something interesting. "
            "The photo captures Milo's natural curiosity and intelligence, with an engaged expression. "
            "Natural lighting, sharp focus, photorealistic style."
        ),
        "gloomy": (
            f"A high-quality, professional photo of Milo, {milo_description}, in a contemplative or gloomy mood. "
            "Milo has a slightly melancholic expression, perhaps gazing wistfully out a window or looking downcast. "
            "The photo captures a moody, pensive moment with softer, muted tones. "
            "Overcast or dim lighting, sharp focus, photorealistic style."
        ),
        "angry": (
            f"A high-quality, professional photo of Milo, {milo_description}, looking grumpy or mildly irritated. "
            "Milo has a stern expression with narrowed eyes or flattened ears, showing feline attitude. "
            "The photo captures Milo's feisty personality in a humorous, endearing way. "
            "Natural lighting, sharp focus, photorealistic style."
        ),
        "regal": (
            f"A high-quality, professional photo of Milo, {milo_description}, in a regal and majestic pose. "
            "Milo sits with perfect posture, looking dignified and noble like royalty. "
            "The photo captures Milo's elegant and sophisticated side with a commanding presence. "
            "Dramatic lighting, sharp focus, photorealistic style."
        ),
        "cozy": (
            f"A high-quality, professional photo of Milo, {milo_description}, in a cozy and comfortable setting. "
            "Milo is nestled in a warm spot, perhaps on a soft blanket or cushion, looking perfectly content. "
            "The photo captures a heartwarming moment of domestic bliss and comfort. "
            "Warm, inviting lighting, sharp focus, photorealistic style."
        )
    }
    
    mood = random.choice(list(moods.keys()))
    prompt = moods[mood]
    
    return mood, prompt


def generate_ai_image(client: AzureOpenAI, image_model: str,
                      text_client: Optional[AzureOpenAI] = None,
                      text_model: Optional[str] = None,
                      blob_service_client: Optional[BlobServiceClient] = None,
                      container_name: Optional[str] = None) -> Optional[bytes]:
    """
    Generate an AI image of Milo using Azure OpenAI DALL-E.
    Uses mood-based prompts to create varied and personalized images of Milo.
    If blob storage client and GPT-4 Vision deployment are provided, analyzes existing
    photos using GPT-4 Vision to extract Milo's visual characteristics for accurate generation.
    
    Args:
        client: Azure OpenAI client
        deployment_name: Name of the DALL-E deployment
        gpt4v_deployment: Optional GPT-4 Vision deployment name for analyzing photos
        blob_service_client: Optional Azure Blob Storage client for accessing existing photos
        container_name: Optional container name where Milo's photos are stored
        
    Returns:
        Image bytes or None if generation failed
    """
    try:
        # Extract Milo's characteristics from existing photos using GPT-4 Vision
        milo_description = "an adorable cat"
        if blob_service_client and text_client and text_model and container_name:
            milo_description = extract_milo_characteristics(
                blob_service_client, text_client, text_model, container_name
            )
        
        # Select a random mood and get corresponding prompt with Milo's characteristics
        mood, prompt = select_mood_and_prompt(milo_description)
        
        logging.info(f"Generating AI image with {image_model} using '{mood}' mood")
        logging.info(f"Milo's appearance: {milo_description}")
        response = client.images.generate(
            model=image_model,
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            style="natural"
        )
        
        # Get the image URL and download it
        if not response.data:
            logging.error("Failed to generate an image with DALL-E")
            return None
        
        image_url = response.data[0].url
        if not image_url:
            logging.error("Failed to generate image url")
            return None
        
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        logging.info(f"AI image generated successfully with '{mood}' mood")
        return image_response.content
        
    except Exception as e:
        logging.error(f"Error generating AI image: {str(e)}")
        return None


def get_current_context() -> Dict[str, Any]:
    """
    Get current temporal context including day of week, season, and notable dates.
    
    Returns:
        Dictionary containing contextual information
    """
    now = datetime.utcnow()
    
    # Day of week
    day_name = calendar.day_name[now.weekday()]
    
    # Season (Northern Hemisphere)
    month = now.month
    if month in [12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5]:
        season = "spring"
    elif month in [6, 7, 8]:
        season = "summer"
    else:
        season = "fall"
    
    # Check for notable holidays/dates
    holidays = []
    
    # New Year's
    if month == 1 and now.day == 1:
        holidays.append("New Year's Day")
    
    # Valentine's Day
    if month == 2 and now.day == 14:
        holidays.append("Valentine's Day")
    
    # St. Patrick's Day
    if month == 3 and now.day == 17:
        holidays.append("St. Patrick's Day")
    
    # April Fool's Day
    if month == 4 and now.day == 1:
        holidays.append("April Fool's Day")
    
    # Halloween
    if month == 10 and now.day == 31:
        holidays.append("Halloween")
    
    # Thanksgiving (4th Thursday of November)
    if month == 11:
        # Find the first day of November
        first_day_weekday = calendar.monthrange(now.year, 11)[0]  # 0 = Monday, 6 = Sunday
        
        # Thursday is weekday 3
        # Calculate the date of the first Thursday
        days_until_thursday = (3 - first_day_weekday) % 7
        first_thursday = 1 + days_until_thursday
        
        # 4th Thursday is 3 weeks (21 days) after the first Thursday
        fourth_thursday = first_thursday + 21
        
        if now.day == fourth_thursday:
            holidays.append("Thanksgiving")
    
    # Christmas
    if month == 12 and now.day == 25:
        holidays.append("Christmas")
    
    # New Year's Eve
    if month == 12 and now.day == 31:
        holidays.append("New Year's Eve")
    
    return {
        "day_of_week": day_name,
        "season": season,
        "holidays": holidays,
        "date": now.strftime("%B %d, %Y")
    }


def generate_witty_caption(openai_client: AzureOpenAI, 
                          text_model: str,
                          context: Dict[str, Any],
                          image_description: str = "") -> str:
    """
    Generate a witty, socially engaging caption using AI based on context and image content.
    
    Args:
        openai_client: Azure OpenAI client
        gpt_deployment: Name of the GPT deployment to use
        context: Contextual information (day of week, season, holidays)
        image_description: Optional description of the image content
        
    Returns:
        A witty caption string
    """
    try:
        # Build context string
        context_parts = [f"It's {context['day_of_week']}"]
        context_parts.append(f"in {context['season']}")
        
        if context['holidays']:
            context_parts.append(f"and it's {', '.join(context['holidays'])}")
        
        context_str = ", ".join(context_parts) + "."
        
        # Build prompt
        prompt = f"""You are a witty social media caption writer for Milo, a grumpy but lovable cat with a sassy personality.

Context: {context_str}

{"Image description: " + image_description if image_description else ""}

Generate a SHORT, witty, and engaging caption (maximum 15 words) that:
- Reflects Milo's grumpy yet endearing personality
- Occasionally references the day/season/holiday, but not every time
- Is funny and relatable to cat lovers
- Avoids hashtags (they'll be added separately)
- Uses a conversational tone that cats might use if they could talk

Return ONLY the caption text, nothing else."""

        logging.info(f"Generating witty caption with context: {context_str}")
        
        response = openai_client.chat.completions.create(
            model=text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative social media caption writer specializing in humorous cat content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=CAPTION_TEMPERATURE
        )
        
        if not response.choices[0].message.content:
            raise ValueError()
        caption = response.choices[0].message.content.strip()
        
        # Clean up the caption - remove quotes if present
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        if caption.startswith("'") and caption.endswith("'"):
            caption = caption[1:-1]
        
        logging.info(f"Generated witty caption: {caption}")
        return caption
        
    except Exception as e:
        logging.error(f"Error generating witty caption: {str(e)}")
        # Fallback to simple captions
        fallback_captions = [
            "Another day, another judgmental stare.",
            "I'm not grumpy, this is just my face.",
            "Existing is exhausting.",
            "Did someone say treats?",
            "Professional napper reporting for duty."
        ]
        return random.choice(fallback_captions)


def post_to_postly(api_key: str, workspace_id: str, 
                   image_data: bytes, caption: str, target_platforms: Optional[str] = None) -> bool:
    """
    Post image to Postly API.
    
    Args:
        api_key: Postly API key
        workspace_id: Postly workspace ID
        image_data: Image bytes to upload
        caption: Caption for the post
        target_platforms: Comma-separated list of platform account IDs (optional)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = {
            "X-API-KEY": api_key
        }
        
        upload_url = "https://openapi.postly.ai/v1/files"
        
        # Step 1: Upload the image
        # Reference: https://docs.postly.ai/upload-a-file-17449007e0
        logging.info(f"Uploading image to Postly ({len(image_data) / 1024:.1f}KB)")
        
        upload_headers = headers.copy()
        upload_headers["X-File-Size"] = str(len(image_data))
        
        files = {
            "file": ("milo.jpg", image_data, "image/jpeg")
        }
        
        upload_response = requests.post(upload_url, headers=upload_headers, files=files)
        upload_response.raise_for_status()
        
        upload_data = upload_response.json()
        image_url = upload_data.get("data", {}).get("url")
        
        if not image_url:
            logging.error(f"No URL returned from upload. Response: {upload_data}")
            return False
        
        logging.info(f"Image uploaded successfully. URL: {image_url}")
        
        # Step 2: Create a post with the uploaded media
        # Reference: https://docs.postly.ai/create-a-post-17486212e0
        post_url = "https://openapi.postly.ai/v1/posts"
        
        post_data = {
            "workspace": workspace_id,
            "text": caption,
            "media": [
                {
                    "url": image_url,
                    "type": "image/jpeg"
                }
            ]
        }
        
        # Add target platforms if provided
        if target_platforms:
            post_data["target_platforms"] = target_platforms
            logging.info(f"Targeting platforms: {target_platforms}")
        else:
            post_data["target_platforms"] = "all"
        
        logging.info("Creating post on Postly")
        post_response = requests.post(post_url, headers=headers, json=post_data)
        post_response.raise_for_status()
        
        post_result = post_response.json()
        logging.info(f"Successfully posted to Postly. Response: {post_result}")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error posting to Postly: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response status: {e.response.status_code}")
            logging.error(f"Response body: {e.response.text}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error posting to Postly: {str(e)}")
        return False


@app.timer_trigger(
    schedule="0 0 17 * * *",  # Cron: sec min hour day month day-of-week (10:00 AM UTC daily)
    arg_name="timer", 
    run_on_startup=False
)
def daily_milo_post(timer: func.TimerRequest) -> None:
    """
    Azure Function triggered daily at 10:00 AM UTC to post a Milo photo.
    
    Args:
        timer: Timer trigger information
    """
    logging.info("Daily Milo Photo Poster function started")
    
    # Validate configuration
    if not all([AZURE_STORAGE_CONNECTION_STRING, COMPUTER_VISION_ENDPOINT, 
                COMPUTER_VISION_KEY, OPENAI_TEXT_API_KEY, OPENAI_IMAGE_ENDPOINT,
                POSTLY_API_KEY, POSTLY_WORKSPACE_ID, OPENAI_TEXT_ENDPOINT]):
        logging.error("Missing required configuration. Please check environment variables.")
        return
    
    image_data = None
    image_source = None
    image_description = ""
    blob_name = None  # Track blob name for marking as posted
    
    try:
        # Step 1: Try to select best photo from blob storage
        logging.info("Attempting to select photo from blob storage")
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        
        cv_credentials = CognitiveServicesCredentials(COMPUTER_VISION_KEY)
        cv_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, cv_credentials)
        
        # Initialize separate OpenAI clients for image and text
        image_client = AzureOpenAI(
            api_key=OPENAI_IMAGE_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=OPENAI_IMAGE_ENDPOINT # type: ignore
        )
        text_client = AzureOpenAI(
            api_key=OPENAI_TEXT_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=OPENAI_TEXT_ENDPOINT # type: ignore
        )
        

        result = select_best_photo(
            blob_service_client, 
            cv_client,
            BLOB_CONTAINER_NAME,
            DAYS_TO_CHECK
        )

        if result:
            image_data, blob_name, image_description = result
            image_source = f"blob storage ({blob_name})"
        else:
            # Step 2: Fallback to AI generation
            logging.info("No suitable photo found, generating AI image")
            image_data = generate_ai_image(
                image_client,
                image_model=OPENAI_IMAGE_MODEL,
                text_client=text_client,
                text_model=OPENAI_TEXT_MODEL,
                blob_service_client=blob_service_client,
                container_name=BLOB_CONTAINER_NAME
            )
            image_source = "AI generated (OpenAI)"
            image_description = "AI-generated image of Milo"

        if not image_data:
            logging.error("Failed to obtain image (neither from storage nor AI)")
            return

        # Step 3: Generate witty caption
        context = get_current_context()
        witty_caption = generate_witty_caption(
            text_client,
            text_model=OPENAI_TEXT_MODEL,
            context=context,
            image_description=image_description
        )

        # Format caption: "Daily Milo! 😾" + witty caption + hashtags
        caption = f"{CAPTION_PREFIX} {witty_caption} {CAPTION_HASHTAGS}"

        logging.info(f"Final caption: {caption}")

        # Step 4: Post to Postly
        success = post_to_postly(
            POSTLY_API_KEY, # type: ignore
            POSTLY_WORKSPACE_ID, # type: ignore
            image_data,
            caption,
            POSTLY_TARGET_PLATFORMS
        )

        if success:
            logging.info(f"Successfully posted daily Milo photo from {image_source}")
            # Mark blob as posted to avoid duplicates
            if blob_name:
                try:
                    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
                    blob_client = container_client.get_blob_client(blob_name)
                    mark_blob_as_posted(blob_client)
                except Exception as e:
                    logging.warning(f"Could not mark blob as posted: {str(e)}")
        else:
            logging.error(f"Failed to post to Postly (image source: {image_source})")

    except Exception as e:
        logging.error(f"Error in daily_milo_post function: {str(e)}", exc_info=True)
