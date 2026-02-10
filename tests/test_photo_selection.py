"""
Tests for photo selection and limiting logic
"""
import sys
import os
import json
import random
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import function_app
from function_app import select_best_photo, get_recent_blobs


def test_random_sampling_when_exceeds_limit():
    """Test that photos are randomly sampled when count exceeds MAX_PHOTOS_TO_ANALYZE"""
    # Set MAX_PHOTOS_TO_ANALYZE to 5 for testing
    with patch('function_app.MAX_PHOTOS_TO_ANALYZE', 5):
        # Create mock blob service client
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container_client
        
        # Create 10 mock blobs
        mock_blobs = []
        for i in range(10):
            mock_blob = Mock()
            mock_blob.name = f"photo_{i}.jpg"
            mock_blob.last_modified = datetime.utcnow()
            mock_blobs.append(mock_blob)
        
        # Mock get_recent_blobs to return 10 blobs
        with patch('function_app.get_recent_blobs', return_value=mock_blobs):
            # Mock Computer Vision client
            mock_cv_client = MagicMock()
            
            # Mock analyze_image_quality to return a low score so no photo is selected
            # This way we can just test the sampling logic
            with patch('function_app.analyze_image_quality', return_value={
                'description': {'captions': [{'text': 'test', 'confidence': 0.1}]},
                'tags': [],
                'adult': {'isAdultContent': False, 'isRacyContent': False},
                'color': {'isBWImg': False},
                'imageType': {'clipArtType': 0, 'lineDrawingType': 0}
            }) as mock_analyze:
                with patch('function_app.calculate_appeal_score', return_value=0):
                    with patch('function_app.generate_blob_sas', return_value="sas_token"):
                        mock_container_client.get_blob_client = MagicMock(side_effect=lambda name: Mock(url=f"http://test/{name}"))
                        
                        # Call select_best_photo
                        result = select_best_photo(
                            mock_blob_service,
                            mock_cv_client,
                            "test-container",
                            7
                        )
                        
                        # Verify that only 5 blobs were processed (analyzed)
                        assert mock_analyze.call_count == 5


def test_no_sampling_when_under_limit():
    """Test that all photos are analyzed when count is under MAX_PHOTOS_TO_ANALYZE"""
    # Set MAX_PHOTOS_TO_ANALYZE to 10 for testing
    with patch('function_app.MAX_PHOTOS_TO_ANALYZE', 10):
        # Create mock blob service client
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container_client
        
        # Create 5 mock blobs (less than limit)
        mock_blobs = []
        for i in range(5):
            mock_blob = Mock()
            mock_blob.name = f"photo_{i}.jpg"
            mock_blob.last_modified = datetime.utcnow()
            mock_blobs.append(mock_blob)
        
        # Mock get_recent_blobs to return 5 blobs
        with patch('function_app.get_recent_blobs', return_value=mock_blobs):
            # Mock Computer Vision client
            mock_cv_client = MagicMock()
            
            # Mock analyze_image_quality
            with patch('function_app.analyze_image_quality', return_value={
                'description': {'captions': [{'text': 'test', 'confidence': 0.1}]},
                'tags': [],
                'adult': {'isAdultContent': False, 'isRacyContent': False},
                'color': {'isBWImg': False},
                'imageType': {'clipArtType': 0, 'lineDrawingType': 0}
            }) as mock_analyze:
                with patch('function_app.calculate_appeal_score', return_value=0):
                    with patch('function_app.generate_blob_sas', return_value="sas_token"):
                        mock_container_client.get_blob_client = MagicMock(side_effect=lambda name: Mock(url=f"http://test/{name}"))
                        
                        # Call select_best_photo
                        result = select_best_photo(
                            mock_blob_service,
                            mock_cv_client,
                            "test-container",
                            7
                        )
                        
                        # Verify that all 5 blobs were processed
                        assert mock_analyze.call_count == 5


def test_random_sampling_variability():
    """Test that random sampling produces different results on multiple runs"""
    # Create 10 mock blobs
    mock_blobs = []
    for i in range(10):
        mock_blob = Mock()
        mock_blob.name = f"photo_{i}.jpg"
        mock_blob.last_modified = datetime.utcnow()
        mock_blobs.append(mock_blob)
    
    # Collect samples from multiple runs
    samples = []
    for run in range(10):
        # Use a different random seed for each run
        random.seed(run)
        
        # Sample blobs (simulating what the function does)
        max_photos = 3
        if len(mock_blobs) > max_photos:
            sampled = random.sample(mock_blobs, max_photos)
            samples.append(tuple(blob.name for blob in sampled))
    
    # Verify that we got different samples (not all the same)
    unique_samples = set(samples)
    assert len(unique_samples) > 1, "Random sampling should produce different results"


if __name__ == "__main__":
    print("Running test_random_sampling_when_exceeds_limit...")
    test_random_sampling_when_exceeds_limit()
    print("✓ test_random_sampling_when_exceeds_limit passed")
    
    print("\nRunning test_no_sampling_when_under_limit...")
    test_no_sampling_when_under_limit()
    print("✓ test_no_sampling_when_under_limit passed")
    
    print("\nRunning test_random_sampling_variability...")
    test_random_sampling_variability()
    print("✓ test_random_sampling_variability passed")
    
    print("\nAll tests passed!")
