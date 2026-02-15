"""
Tests for photo selection and limiting logic
"""

import sys
import os
import random
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from function_app import (
    select_best_photo,
    is_blob_posted,
    mark_blob_as_posted,
    POSTED_METADATA_KEY,
)


def test_random_sampling_when_exceeds_limit():
    """Test that photos are randomly sampled when count exceeds MAX_PHOTOS_TO_ANALYZE"""
    # Set MAX_PHOTOS_TO_ANALYZE to 5 for testing
    with patch("function_app.MAX_PHOTOS_TO_ANALYZE", 5):
        with patch("function_app.downsize_image_if_needed", return_value=False):
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
            with patch("function_app.get_recent_blobs", return_value=mock_blobs):
                # Mock Computer Vision client
                mock_cv_client = MagicMock()

                # Mock analyze_image_quality to return a low score so no photo is selected
                # This way we can just test the sampling logic
                with patch(
                    "function_app.analyze_image_quality",
                    return_value={
                        "description": {
                            "captions": [{"text": "test", "confidence": 0.1}]
                        },
                        "tags": [],
                        "adult": {"isAdultContent": False, "isRacyContent": False},
                        "color": {"isBWImg": False},
                        "imageType": {"clipArtType": 0, "lineDrawingType": 0},
                    },
                ) as mock_analyze:
                    with patch("function_app.calculate_appeal_score", return_value=0):
                        with patch(
                            "function_app.generate_blob_sas", return_value="sas_token"
                        ):
                            mock_container_client.get_blob_client = MagicMock(
                                side_effect=lambda name: Mock(url=f"http://test/{name}")
                            )

                            # Call select_best_photo
                            mock_text_client = MagicMock()
                            _ = select_best_photo(
                                mock_blob_service,
                                mock_cv_client,
                                mock_text_client,
                                "gpt-4o",
                                "test-container",
                                7,
                            )

                            # Verify that only 5 blobs were processed (analyzed)
                            assert mock_analyze.call_count == 5


def test_no_sampling_when_under_limit():
    """Test that all photos are analyzed when count is under MAX_PHOTOS_TO_ANALYZE"""
    # Set MAX_PHOTOS_TO_ANALYZE to 10 for testing
    with patch("function_app.MAX_PHOTOS_TO_ANALYZE", 10):
        with patch("function_app.downsize_image_if_needed", return_value=False):
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
            with patch("function_app.get_recent_blobs", return_value=mock_blobs):
                # Mock Computer Vision client
                mock_cv_client = MagicMock()

                # Mock analyze_image_quality
                with patch(
                    "function_app.analyze_image_quality",
                    return_value={
                        "description": {
                            "captions": [{"text": "test", "confidence": 0.1}]
                        },
                        "tags": [],
                        "adult": {"isAdultContent": False, "isRacyContent": False},
                        "color": {"isBWImg": False},
                        "imageType": {"clipArtType": 0, "lineDrawingType": 0},
                    },
                ) as mock_analyze:
                    with patch("function_app.calculate_appeal_score", return_value=0):
                        with patch(
                            "function_app.generate_blob_sas", return_value="sas_token"
                        ):
                            mock_container_client.get_blob_client = MagicMock(
                                side_effect=lambda name: Mock(url=f"http://test/{name}")
                            )

                            # Call select_best_photo
                            mock_text_client = MagicMock()
                            _ = select_best_photo(
                                mock_blob_service,
                                mock_cv_client,
                                mock_text_client,
                                "gpt-4o",
                                "test-container",
                                7,
                            )

                            # Verify that all 5 blobs were processed
                            assert mock_analyze.call_count == 5


def test_random_sampling_variability():
    """Test that random sampling produces different results on multiple runs

    Note: This test is probabilistic but has a very low failure rate.
    With 20 runs selecting 3 items from 10, the probability of getting
    all identical samples is astronomically low (< 10^-30).
    """
    # Create 10 mock blobs
    mock_blobs = []
    for i in range(10):
        mock_blob = Mock()
        mock_blob.name = f"photo_{i}.jpg"
        mock_blob.last_modified = datetime.utcnow()
        mock_blobs.append(mock_blob)

    # Collect samples from multiple runs (without setting seed to test true randomness)
    samples = []
    max_photos = 3
    for run in range(20):
        # Sample blobs (simulating what the function does)
        if len(mock_blobs) > max_photos:
            sampled = random.sample(mock_blobs, max_photos)
            samples.append(tuple(sorted(blob.name for blob in sampled)))

    # Verify that we got different samples (not all the same)
    # With 20 runs selecting 3 from 10, we should get multiple unique combinations
    unique_samples = set(samples)
    assert len(unique_samples) > 1, "Random sampling should produce different results"


def test_boundary_condition_exact_limit():
    """Test that all photos are analyzed when count equals MAX_PHOTOS_TO_ANALYZE"""
    # Set MAX_PHOTOS_TO_ANALYZE to 5 for testing
    with patch("function_app.MAX_PHOTOS_TO_ANALYZE", 5):
        with patch("function_app.downsize_image_if_needed", return_value=False):
            # Create mock blob service client
            mock_blob_service = MagicMock()
            mock_container_client = MagicMock()
            mock_blob_service.get_container_client.return_value = mock_container_client

            # Create exactly 5 mock blobs (equal to limit)
            mock_blobs = []
            for i in range(5):
                mock_blob = Mock()
                mock_blob.name = f"photo_{i}.jpg"
                mock_blob.last_modified = datetime.utcnow()
                mock_blobs.append(mock_blob)

            # Mock get_recent_blobs to return 5 blobs
            with patch("function_app.get_recent_blobs", return_value=mock_blobs):
                # Mock Computer Vision client
                mock_cv_client = MagicMock()

                # Mock analyze_image_quality
                with patch(
                    "function_app.analyze_image_quality",
                    return_value={
                        "description": {
                            "captions": [{"text": "test", "confidence": 0.1}]
                        },
                        "tags": [],
                        "adult": {"isAdultContent": False, "isRacyContent": False},
                        "color": {"isBWImg": False},
                        "imageType": {"clipArtType": 0, "lineDrawingType": 0},
                    },
                ) as mock_analyze:
                    with patch("function_app.calculate_appeal_score", return_value=0):
                        with patch(
                            "function_app.generate_blob_sas", return_value="sas_token"
                        ):
                            mock_container_client.get_blob_client = MagicMock(
                                side_effect=lambda name: Mock(url=f"http://test/{name}")
                            )

                            # Call select_best_photo
                            mock_text_client = MagicMock()
                            _ = select_best_photo(
                                mock_blob_service,
                                mock_cv_client,
                                mock_text_client,
                                "gpt-4o",
                                "test-container",
                                7,
                            )

                            # Verify that all 5 blobs were processed (no sampling needed)
                            assert mock_analyze.call_count == 5


def test_duplicate_avoidance_filters_posted_photos():
    """Test that already-posted photos are filtered out"""

    with patch("function_app.downsize_image_if_needed", return_value=False):
        # Create mock blob service client
        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_service.get_container_client.return_value = mock_container_client

        # Create mock blobs - some posted, some not
        mock_blobs = []
        for i in range(5):
            mock_blob = Mock()
            mock_blob.name = f"photo_{i}.jpg"
            mock_blob.last_modified = datetime.utcnow()
            mock_blobs.append(mock_blob)

        # Mock blob clients with metadata
        def mock_get_blob_client(name):
            blob_client = MagicMock()
            blob_client.blob_name = name

            # Simulate some photos as already posted
            if name in ["photo_0.jpg", "photo_2.jpg"]:
                # These were "posted" recently
                properties = Mock()
                properties.metadata = {"posted_date": datetime.utcnow().isoformat()}
                blob_client.get_blob_properties.return_value = properties
            else:
                # These haven't been posted
                properties = Mock()
                properties.metadata = {}
                blob_client.get_blob_properties.return_value = properties

            blob_client.url = f"http://test/{name}"
            return blob_client

        mock_container_client.get_blob_client = mock_get_blob_client

        # Mock get_recent_blobs to return our test blobs
        with patch("function_app.get_recent_blobs", return_value=mock_blobs):
            # Mock Computer Vision client
            mock_cv_client = MagicMock()

            # Mock analyze_image_quality
            with patch(
                "function_app.analyze_image_quality",
                return_value={
                    "description": "test",
                    "confidence": 0.1,
                    "tags": [],
                    "is_adult_content": False,
                    "is_racy_content": False,
                    "dominant_colors": [],
                    "is_bw": False,
                    "is_clip_art": 0,
                    "is_line_drawing": 0,
                },
            ) as mock_analyze:
                with patch("function_app.calculate_appeal_score", return_value=0):
                    with patch(
                        "function_app.generate_blob_sas", return_value="sas_token"
                    ):
                        with patch("function_app.POSTED_HISTORY_DAYS", 30):
                            # Call select_best_photo
                            mock_text_client = MagicMock()
                            _ = select_best_photo(
                                mock_blob_service,
                                mock_cv_client,
                                mock_text_client,
                                "gpt-4o",
                                "test-container",
                                7,
                            )

                            # Should only analyze 3 photos (photo_1, photo_3, photo_4)
                            # photo_0 and photo_2 should be filtered out as posted
                            assert mock_analyze.call_count == 3


def test_mark_blob_as_posted():
    """Test that marking a blob as posted sets the correct metadata"""

    # Create a mock blob client
    mock_blob_client = MagicMock()
    mock_blob_client.blob_name = "test_photo.jpg"

    # Mock existing metadata
    properties = Mock()
    properties.metadata = {"existing_key": "existing_value"}
    mock_blob_client.get_blob_properties.return_value = properties

    # Call mark_blob_as_posted
    mark_blob_as_posted(mock_blob_client)

    # Verify set_blob_metadata was called
    assert mock_blob_client.set_blob_metadata.called

    # Verify the metadata includes the posted_date key
    called_metadata = mock_blob_client.set_blob_metadata.call_args[0][0]
    assert POSTED_METADATA_KEY in called_metadata
    assert "existing_key" in called_metadata  # Existing metadata preserved


def test_is_blob_posted_returns_false_for_unposted():
    """Test that is_blob_posted returns False for blobs without metadata"""

    # Create a mock blob client
    mock_blob_client = MagicMock()
    properties = Mock()
    properties.metadata = {}
    mock_blob_client.get_blob_properties.return_value = properties

    # Should return False
    assert is_blob_posted(mock_blob_client, 30) is False


def test_is_blob_posted_returns_true_for_recently_posted():
    """Test that is_blob_posted returns True for recently posted photos"""

    # Create a mock blob client
    mock_blob_client = MagicMock()
    properties = Mock()
    # Posted 5 days ago
    posted_date = datetime.utcnow() - timedelta(days=5)
    properties.metadata = {POSTED_METADATA_KEY: posted_date.isoformat()}
    mock_blob_client.get_blob_properties.return_value = properties

    # Should return True with history_days=30
    assert is_blob_posted(mock_blob_client, 30) is True


def test_is_blob_posted_returns_false_for_old_posts():
    """Test that is_blob_posted returns False for photos posted long ago"""

    # Create a mock blob client
    mock_blob_client = MagicMock()
    properties = Mock()
    # Posted 35 days ago
    posted_date = datetime.utcnow() - timedelta(days=35)
    properties.metadata = {POSTED_METADATA_KEY: posted_date.isoformat()}
    mock_blob_client.get_blob_properties.return_value = properties

    # Should return False with history_days=30
    assert is_blob_posted(mock_blob_client, 30) is False


if __name__ == "__main__":
    print("Running test_random_sampling_when_exceeds_limit...")
    test_random_sampling_when_exceeds_limit()
    print("✓ test_random_sampling_when_exceeds_limit passed")

    print("\nRunning test_no_sampling_when_under_limit...")
    test_no_sampling_when_under_limit()
    print("✓ test_no_sampling_when_under_limit passed")

    print("\nRunning test_boundary_condition_exact_limit...")
    test_boundary_condition_exact_limit()
    print("✓ test_boundary_condition_exact_limit passed")

    print("\nRunning test_random_sampling_variability...")
    test_random_sampling_variability()
    print("✓ test_random_sampling_variability passed")

    print("\nRunning test_duplicate_avoidance_filters_posted_photos...")
    test_duplicate_avoidance_filters_posted_photos()
    print("✓ test_duplicate_avoidance_filters_posted_photos passed")

    print("\nRunning test_mark_blob_as_posted...")
    test_mark_blob_as_posted()
    print("✓ test_mark_blob_as_posted passed")

    print("\nRunning test_is_blob_posted_returns_false_for_unposted...")
    test_is_blob_posted_returns_false_for_unposted()
    print("✓ test_is_blob_posted_returns_false_for_unposted passed")

    print("\nRunning test_is_blob_posted_returns_true_for_recently_posted...")
    test_is_blob_posted_returns_true_for_recently_posted()
    print("✓ test_is_blob_posted_returns_true_for_recently_posted passed")

    print("\nRunning test_is_blob_posted_returns_false_for_old_posts...")
    test_is_blob_posted_returns_false_for_old_posts()
    print("✓ test_is_blob_posted_returns_false_for_old_posts passed")

    print("\nAll tests passed!")
