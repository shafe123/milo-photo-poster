"""
Milo Photo Poster - Azure Function
Automatically posts a daily photo of Milo the cat using the Postly API.
"""

import os
import logging
import io
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import base64

import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobProperties, generate_blob_sas, BlobSasPermissions
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI
import requests
from PIL import Image

app = func.FunctionApp()

# Configuration
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.environ.get("BLOB_CONTAINER_NAME", "milo-photos")
COMPUTER_VISION_ENDPOINT = os.environ.get("COMPUTER_VISION_ENDPOINT")
COMPUTER_VISION_KEY = os.environ.get("COMPUTER_VISION_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.environ.get("OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.environ.get("OPENAI_DEPLOYMENT_NAME", "dall-e-3")
POSTLY_API_KEY = os.environ.get("POSTLY_API_KEY")
POSTLY_WORKSPACE_ID = os.environ.get("POSTLY_WORKSPACE_ID")
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "7"))

# Constants
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
MIN_ACCEPTABLE_SCORE = 30  # Minimum photo appeal score to accept (0-100 scale)


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
                     days: int) -> Optional[Tuple[bytes, str]]:
    """
    Select the best photo from blob storage based on appeal score.
    
    Args:
        blob_service_client: Azure Blob Storage client
        cv_client: Computer Vision client
        container_name: Name of the blob container
        days: Number of days to look back
        
    Returns:
        Tuple of (image_bytes, blob_name) or None if no suitable photo found
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        recent_blobs = get_recent_blobs(container_client, days)
        
        if not recent_blobs:
            logging.info("No recent photos found in blob storage")
            return None
        
        best_blob = None
        best_score = -1
        
        for blob in recent_blobs:
            try:
                # Get blob client and generate SAS URL for Computer Vision API access
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
                
                # Analyze the image
                logging.info(f"Analyzing blob: {blob.name}")
                analysis = analyze_image_quality(cv_client, blob_url)
                
                # Calculate appeal score
                score = calculate_appeal_score(analysis)
                logging.info(f"Blob {blob.name} scored: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_blob = blob
                    
            except Exception as e:
                logging.warning(f"Error processing blob {blob.name}: {str(e)}")
                continue
        
        if best_blob and best_score > MIN_ACCEPTABLE_SCORE:
            logging.info(f"Selected blob: {best_blob.name} with score {best_score:.2f}")
            blob_client = container_client.get_blob_client(best_blob.name)
            image_data = blob_client.download_blob().readall()
            return (image_data, best_blob.name)
        else:
            logging.info(f"No photo met the minimum quality threshold ({MIN_ACCEPTABLE_SCORE})")
            return None
            
    except Exception as e:
        logging.error(f"Error selecting best photo: {str(e)}")
        return None


def generate_ai_image(client: AzureOpenAI, deployment_name: str) -> Optional[bytes]:
    """
    Generate an AI image of Milo using Azure OpenAI DALL-E.
    
    Args:
        client: Azure OpenAI client
        deployment_name: Name of the DALL-E deployment
        
    Returns:
        Image bytes or None if generation failed
    """
    try:
        prompt = (
            "A high-quality, adorable photo of a cat named Milo. "
            "The cat should look happy and photogenic. "
            "Professional photography style, good lighting, sharp focus."
        )
        
        logging.info("Generating AI image with DALL-E")
        response = client.images.generate(
            model=deployment_name,
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            style="natural"
        )
        
        # Get the image URL and download it
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        logging.info("AI image generated successfully")
        return image_response.content
        
    except Exception as e:
        logging.error(f"Error generating AI image: {str(e)}")
        return None


def post_to_postly(api_key: str, workspace_id: str, 
                   image_data: bytes, caption: str) -> bool:
    """
    Post image to Postly API.
    
    Args:
        api_key: Postly API key
        workspace_id: Postly workspace ID
        image_data: Image bytes to upload
        caption: Caption for the post
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Based on Postly API documentation
        # First, upload the image to get a media ID
        upload_url = "https://api.postly.ai/v1/media/upload"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        
        files = {
            "file": ("milo.jpg", image_data, "image/jpeg")
        }
        
        data = {
            "workspace_id": workspace_id
        }
        
        logging.info("Uploading image to Postly")
        upload_response = requests.post(upload_url, headers=headers, files=files, data=data)
        upload_response.raise_for_status()
        
        media_id = upload_response.json().get("media_id")
        
        if not media_id:
            logging.error("No media_id returned from upload")
            return False
        
        # Now create a post with the uploaded media
        post_url = "https://api.postly.ai/v1/posts"
        
        post_data = {
            "workspace_id": workspace_id,
            "caption": caption,
            "media_ids": [media_id],
            "publish_now": True
        }
        
        logging.info("Creating post on Postly")
        post_response = requests.post(post_url, headers=headers, json=post_data)
        post_response.raise_for_status()
        
        logging.info("Successfully posted to Postly")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error posting to Postly: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response: {e.response.text}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error posting to Postly: {str(e)}")
        return False


@app.timer_trigger(
    schedule="0 0 10 * * *",  # Cron: sec min hour day month day-of-week (10:00 AM UTC daily)
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
                COMPUTER_VISION_KEY, OPENAI_API_KEY, OPENAI_ENDPOINT,
                POSTLY_API_KEY, POSTLY_WORKSPACE_ID]):
        logging.error("Missing required configuration. Please check environment variables.")
        return
    
    image_data = None
    image_source = None
    
    try:
        # Step 1: Try to select best photo from blob storage
        logging.info("Attempting to select photo from blob storage")
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        
        cv_credentials = CognitiveServicesCredentials(COMPUTER_VISION_KEY)
        cv_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, cv_credentials)
        
        result = select_best_photo(
            blob_service_client, 
            cv_client,
            BLOB_CONTAINER_NAME,
            DAYS_TO_CHECK
        )
        
        if result:
            image_data, blob_name = result
            image_source = f"blob storage ({blob_name})"
        else:
            # Step 2: Fallback to AI generation
            logging.info("No suitable photo found, generating AI image")
            openai_client = AzureOpenAI(
                api_key=OPENAI_API_KEY,
                api_version="2024-02-01",
                azure_endpoint=OPENAI_ENDPOINT
            )
            
            image_data = generate_ai_image(openai_client, OPENAI_DEPLOYMENT_NAME)
            image_source = "AI generated (DALL-E)"
        
        if not image_data:
            logging.error("Failed to obtain image (neither from storage nor AI)")
            return
        
        # Step 3: Post to Postly
        today = datetime.utcnow().strftime("%Y-%m-%d")
        caption = f"Daily Milo! 🐱 #{today.replace('-', '')} #Milo #CatsOfInstagram"
        
        success = post_to_postly(
            POSTLY_API_KEY,
            POSTLY_WORKSPACE_ID,
            image_data,
            caption
        )
        
        if success:
            logging.info(f"Successfully posted daily Milo photo from {image_source}")
        else:
            logging.error(f"Failed to post to Postly (image source: {image_source})")
            
    except Exception as e:
        logging.error(f"Error in daily_milo_post function: {str(e)}", exc_info=True)
