from fastapi import FastAPI, HTTPException
import cv2
import random
import base64
import numpy as np
import time
from contextlib import asynccontextmanager

# Initialize the FastAPI app first
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_camera()  # Initialize the camera when the app starts
    yield  # The app runs here
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Release the camera when the app stops

app = FastAPI(lifespan=lifespan)

# Global variable for camera
cap = None

# Function to keep the camera open
def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Error: Could not open webcam.")

    # Set a higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Add a short delay to allow the camera to adjust
    time.sleep(1)

# Function to capture an image from the open camera
def capture_image():
    global cap
    if not cap.isOpened():
        return None, "Error: Camera is not opened."

    # Read a single frame from the webcam
    ret, frame = cap.read()

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
    height, width, _ = image.shape
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    z = random.uniform(0, 100)
    return (x, y, z)

@app.get("/capture_image")
def api_capture_image():
    image, message = capture_image()
    if image is not None:
        image_base64 = image_to_base64(image)
        return {"message": message, "image": image_base64}
    else:
        raise HTTPException(status_code=500, detail=message)

@app.post("/calculate_points")
def api_calculate_points(data: dict):
    # Get the base64-encoded image from the request body
    image_base64 = data.get("image", None)
    
    if image_base64 is None:
        raise HTTPException(status_code=400, detail="Image is required.")

    # Convert the base64-encoded image back to an image format
    image = base64_to_image(image_base64)

    # Calculate points based on the image
    points = calculate_points(image)

    return {"points": points}
