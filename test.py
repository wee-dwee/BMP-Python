import requests
import json
import cv2
import random
import base64
import numpy as np

# Capture an image
response = requests.get("http://127.0.0.1:8000/capture_image")
print("Step-1 done")
capture_response = response.json()
print(capture_response)

def base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image



def show_image(captured_image):
    if captured_image is None:
        print("Error: No image to display. Capture an image first.")
        return
    
    # Display the captured image in a window
    cv2.imshow("Captured Image :", captured_image)
    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# If image captured successfully, calculate points
if "image" in capture_response:
    image_base64 = capture_response["image"]
    
    # Prepare the payload for calculating points
    payload = {
        "image": image_base64
    }

    image = base64_to_image(image_base64)
    show_image(image)
    
    # Call the calculate points API
    response = requests.post("http://127.0.0.1:8000/calculate_points", json=payload)
    print(response.json())
else:
    print("Image not in response")
