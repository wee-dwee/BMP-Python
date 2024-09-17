from flask import Flask, jsonify, request
import cv2
import random
import base64
import numpy as np
import time

app = Flask(__name__)

# Function to capture an image from the webcam
def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None, "Error: Could not open webcam."

    # Set a higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Add a short delay to allow the camera to adjust
    time.sleep(1)

    # Read a single frame from the webcam
    ret, frame = cap.read()

    # Release the webcam
    cap.release()

    if ret:
        return frame, "Image captured successfully."
    else:
        return None, "Error: Could not capture image."

# Function to convert an image to base64 for transmission
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Convert to JPEG format
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Encode to base64
    return image_base64

# Function to convert base64 to image
def base64_to_image(image_base64):
    image_data = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

# Function to generate random 3D points using the image
def calculate_points(image):
    # Example: Use the image to generate points (Here, we generate random points)
    # You could add logic to use image properties to influence these points
    height, width, _ = image.shape
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    z = random.uniform(0, 100)
    return (x, y, z)

# Flask route to capture an image
@app.route('/capture_image', methods=['GET'])
def api_capture_image():
    image, message = capture_image()
    if image is not None:
        image_base64 = image_to_base64(image)
        return jsonify({"message": message, "image": image_base64}), 200
    else:
        return jsonify({"error": message}), 500

# Flask route to calculate points based on the image
@app.route('/calculate_points', methods=['POST'])
def api_calculate_points():
    # Get the base64-encoded image from the request body
    data = request.get_json()
    image_base64 = data.get("image", None)
    
    if image_base64 is None:
        return jsonify({"error": "Image is required."}), 400

    # Convert the base64-encoded image back to an image format
    image = base64_to_image(image_base64)

    # Calculate points based on the image
    points = calculate_points(image)

    return jsonify({"points": points}), 200

if __name__ == "__main__":
    app.run(debug=True)
