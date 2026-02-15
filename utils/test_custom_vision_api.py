"""
Diagnostic script to test Custom Vision API directly
"""

import os
import requests
import json

# Load credentials
endpoint = os.environ.get("CUSTOM_VISION_PREDICTION_ENDPOINT")
key = os.environ.get("CUSTOM_VISION_PREDICTION_KEY")
project_id = os.environ.get("CUSTOM_VISION_PROJECT_ID")
iteration = os.environ.get("CUSTOM_VISION_ITERATION_NAME")

print("=" * 80)
print("Custom Vision API Diagnostic Test")
print("=" * 80)
print("\nConfiguration:")
print(f"  Endpoint: {endpoint}")
print(f"  Project ID: {project_id}")
print(f"  Iteration: {iteration}")
print(f"  Key: {key[:10]}...{key[-10:]}")

# Strip trailing slash and construct URL
endpoint_clean = endpoint.rstrip("/")
prediction_url = f"{endpoint_clean}/customvision/v3.0/Prediction/{project_id}/classify/iterations/{iteration}/url"

print("\nConstructed URL:")
print(f"  {prediction_url}")

# Test image URL - using a simple accessible cat image
test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

headers = {"Prediction-Key": key, "Content-Type": "application/json"}

body = {"Url": test_image_url}

print(f"\nTest image URL: {test_image_url}")
print("\nMaking API call...")

try:
    response = requests.post(prediction_url, headers=headers, json=body)
    print(f"\nResponse Status: {response.status_code}")
    print("Response Headers:")
    for header, value in response.headers.items():
        print(f"  {header}: {value}")

    print("\nResponse Body:")
    try:
        response_json = response.json()
        print(json.dumps(response_json, indent=2))
    except Exception:
        print(response.text)

    if response.status_code == 200:
        print("\n✓ SUCCESS - API call worked!")
        predictions = response.json().get("predictions", [])
        print("\nPredictions:")
        for pred in predictions:
            print(f"  - {pred.get('tagName')}: {pred.get('probability'):.4f}")
    else:
        print(f"\n✗ FAILED - Status Code: {response.status_code}")
        print("\nPossible issues:")
        print("  1. Project ID might be incorrect (should be a GUID, not a name)")
        print("  2. Iteration name might be wrong")
        print("  3. Prediction key might be incorrect")
        print("  4. Endpoint URL might be malformed")

except Exception as e:
    print(f"\n✗ ERROR: {e}")

print("\n" + "=" * 80)
