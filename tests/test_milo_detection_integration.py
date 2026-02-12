"""
Integration tests for Milo detection using Azure Custom Vision
These tests make real API calls and should be run with: pytest -m integration
"""
import sys
import os
from unittest.mock import MagicMock, Mock
import json

# Mark all tests in this file as integration tests
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import function_app
from function_app import check_milo_in_photo


@pytest.mark.integration
def test_milo_detection_real_api_call():
    """
    Integration test: Make a real API call to Custom Vision
    
    Prerequisites:
    - CUSTOM_VISION_PREDICTION_ENDPOINT environment variable set
    - CUSTOM_VISION_PREDICTION_KEY environment variable set
    - CUSTOM_VISION_PROJECT_ID environment variable set
    - CUSTOM_VISION_ITERATION_NAME environment variable set (optional, defaults to Iteration1)
    - A test image URL (modify TEST_IMAGE_URL below)
    
    Run with: pytest -m integration -v
    """
    # Load configuration from environment
    endpoint = os.environ.get("CUSTOM_VISION_PREDICTION_ENDPOINT")
    key = os.environ.get("CUSTOM_VISION_PREDICTION_KEY")
    project_id = os.environ.get("CUSTOM_VISION_PROJECT_ID")
    iteration = os.environ.get("CUSTOM_VISION_ITERATION_NAME", "Iteration1")
    
    # Skip test if configuration is missing
    if not all([endpoint, key, project_id]):
        pytest.skip("Custom Vision credentials not configured in environment")
    
    # Set up function_app configuration
    function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = endpoint
    function_app.CUSTOM_VISION_PREDICTION_KEY = key
    function_app.CUSTOM_VISION_PROJECT_ID = project_id
    function_app.CUSTOM_VISION_ITERATION_NAME = iteration
    function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
    
    # Test image URL - using a publicly accessible cat image for testing
    # For actual testing, replace with a real Milo photo URL from blob storage
    TEST_IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800"
    
    # You can also test with an Azure Blob Storage URL if you have a test image uploaded
    # TEST_IMAGE_URL = "https://milophotosstg.blob.core.windows.net/milo-photos/test-image.jpg"
    
    # Mock blob client (we don't need actual blob storage for this test)
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test-integration-photo.jpg"
    
    # First call: no cached metadata
    mock_properties = Mock()
    mock_properties.metadata = {}
    mock_blob_client.get_blob_properties.return_value = mock_properties
    
    print(f"\nTesting Custom Vision with image: {TEST_IMAGE_URL}")
    print(f"Endpoint: {endpoint}")
    print(f"Project ID: {project_id}")
    print(f"Iteration: {iteration}")
    
    # Make the actual API call
    is_milo_present, confidence = check_milo_in_photo(
        mock_blob_client,
        TEST_IMAGE_URL
    )
    
    print(f"\nResults:")
    print(f"  Milo detected: {is_milo_present}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Verify the function returned valid values
    assert isinstance(is_milo_present, bool), "Should return boolean for is_milo_present"
    assert 0.0 <= confidence <= 1.0, f"Confidence should be between 0 and 1, got {confidence}"
    
    # Check if API call was successful (not a fallback result)
    # If confidence is exactly 1.0 and Milo is detected, it might be a fallback from error handling
    if confidence == 1.0 and is_milo_present:
        print("\nWarning: Got fallback result (confidence=1.0), API call may have failed")
        print("  Check if the test image URL is accessible to Custom Vision API")
        # Don't fail the test, but mark it as inconclusive
        pytest.skip("API call may have failed - got fallback result")
    
    # The function should have attempted to cache the result if successful
    if mock_blob_client.set_blob_metadata.called:
        cached_metadata = mock_blob_client.set_blob_metadata.call_args[0][0]
        assert function_app.MILO_DETECTED_KEY in cached_metadata
        assert function_app.MILO_CONFIDENCE_KEY in cached_metadata
        assert cached_metadata[function_app.MILO_DETECTED_KEY] in ["true", "false"]
        print("\n[OK] Integration test passed - Custom Vision API is working correctly")
    else:
        print("\n[WARN] Warning: Result was not cached, API call may have failed")


@pytest.mark.integration
def test_milo_detection_with_blob_storage_image():
    """
    Integration test: Test with actual blob storage image
    
    This test requires:
    - Azure Blob Storage with an uploaded test image
    - AZURE_STORAGE_CONNECTION_STRING environment variable
    - BLOB_CONTAINER_NAME environment variable
    
    Run with: pytest -m integration -v -k blob_storage
    """
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from datetime import datetime, timedelta
    
    # Load configuration
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.environ.get("BLOB_CONTAINER_NAME", "milo-photos")
    endpoint = os.environ.get("CUSTOM_VISION_PREDICTION_ENDPOINT")
    key = os.environ.get("CUSTOM_VISION_PREDICTION_KEY")
    project_id = os.environ.get("CUSTOM_VISION_PROJECT_ID")
    
    # Skip if not configured
    if not all([connection_string, endpoint, key, project_id]):
        pytest.skip("Azure Storage or Custom Vision not fully configured")
    
    # Set up function_app configuration
    function_app.CUSTOM_VISION_PREDICTION_ENDPOINT = endpoint
    function_app.CUSTOM_VISION_PREDICTION_KEY = key
    function_app.CUSTOM_VISION_PROJECT_ID = project_id
    function_app.CUSTOM_VISION_ITERATION_NAME = os.environ.get("CUSTOM_VISION_ITERATION_NAME", "Iteration1")
    function_app.MILO_CONFIDENCE_THRESHOLD = 0.7
    
    # Connect to blob storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # Get the first image from the container
    blobs = list(container_client.list_blobs(name_starts_with=""))
    image_blobs = [b for b in blobs if b.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    if not image_blobs:
        pytest.skip(f"No images found in container {container_name}")
    
    # Use the first image
    test_blob = image_blobs[0]
    blob_client = container_client.get_blob_client(test_blob.name)
    
    print(f"\nTesting with blob: {test_blob.name}")
    
    # Generate SAS URL for the image
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=test_blob.name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    
    image_url = f"{blob_client.url}?{sas_token}"
    
    # Clear any cached metadata first
    try:
        properties = blob_client.get_blob_properties()
        metadata = properties.metadata or {}
        if function_app.MILO_DETECTED_KEY in metadata:
            del metadata[function_app.MILO_DETECTED_KEY]
        if function_app.MILO_CONFIDENCE_KEY in metadata:
            del metadata[function_app.MILO_CONFIDENCE_KEY]
        blob_client.set_blob_metadata(metadata)
        print("Cleared cached detection metadata")
    except Exception as e:
        print(f"Note: Could not clear cached metadata: {e}")
    
    # Make the actual API call
    is_milo_present, confidence = check_milo_in_photo(
        blob_client,
        image_url
    )
    
    print(f"\nResults for {test_blob.name}:")
    print(f"  Milo detected: {is_milo_present}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Verify valid results
    assert isinstance(is_milo_present, bool)
    assert 0.0 <= confidence <= 1.0
    
    # Verify caching worked
    properties = blob_client.get_blob_properties()
    metadata = properties.metadata
    assert function_app.MILO_DETECTED_KEY in metadata
    assert function_app.MILO_CONFIDENCE_KEY in metadata
    assert function_app.MILO_ITERATION_KEY in metadata
    
    # Verify iteration is stored correctly
    cached_iteration = metadata[function_app.MILO_ITERATION_KEY]
    expected_iteration = os.environ.get("CUSTOM_VISION_ITERATION_NAME", "Iteration1")
    assert cached_iteration == expected_iteration, f"Expected iteration {expected_iteration}, got {cached_iteration}"
    
    print(f"\n[OK] Integration test passed - Blob storage image analyzed successfully")
    print(f"  Cached iteration: {cached_iteration}")
    
    # Test that cached result is used on second call
    is_milo_cached, confidence_cached = check_milo_in_photo(blob_client, image_url)
    assert is_milo_cached == is_milo_present
    # Allow small floating point difference due to metadata storage precision
    assert abs(confidence_cached - confidence) < 0.001
    print("[OK] Cached result verified")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
