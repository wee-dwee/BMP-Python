from fastapi import FastAPI, HTTPException
import cv2
import random
import base64
import numpy as np
import time
from contextlib import asynccontextmanager
import json
from PIL import Image
import Trans3D
from SingleCameraAruCoLib_v3 import singleCameraAruco

# Initialize the FastAPI app first
def image_to_numpy(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)
    
    # Convert the image to RGB mode if it's not already in that mode
    img = img.convert("RGB")
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    return img_array

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

def read_json(file_name):
    with open(file_name) as fh:
        system_config = json.load(fh)
    return system_config

def calculate_points(image):
    # Reading system configuration
    system_config = read_json("Config_SingleCamUSB.json")
     # Initialize the single camera Aruco detector
    SingleCamAruco_obj = singleCameraAruco(aruco_ids_list=system_config['aruco_ids'], 
                                           aruco_size=system_config['aruco_size'],
                                           AruCo_DICT_KEY=system_config['aruco_dictionary'],
                                           camera_matrix=np.load(system_config['camera_matrix_file']),
                                           distortion_matrix=np.load(system_config['distortion_matrix_file']))
    frame = image
    if frame is None:
        print("Error: Could not read the image file.")
    else:
        # Processing the image for ArUco markers
        print("Processing image...")
        
        # Corner point list [array, array] for two aruco markers
        is_marker_detected, corner_points = SingleCamAruco_obj.get_world_coordinates(frame)
        print('Corner Points : ')
        print(corner_points)
        
        if all(is_marker_detected):
            print("Both markers detected.")
            # Use corner_points[0] as original points and corner_points[1] as additional points to transform
            points = corner_points[0]  # First marker's corner points
            coordinates = corner_points[1]  # Second marker's corner points
            
            # Target Points (desired points for the transformation)
            target_points = np.array([
                [-35, 35, 0],
                [35, 35, 0],
                [35, -35, 0],
                [-35, -35, 0]
            ]) / 1000.0  # Example target points in meters
            
            # Compute the affine transformation matrix
            transformation_matrix = Trans3D.affine_matrix_from_points(points.T, target_points.T, False, False, True)

            # Verify the transformation for the original points (corner_points[0])
            points_homogeneous = np.vstack((points.T, np.ones((1, points.shape[0]))))
            transformed_points = np.dot(transformation_matrix, points_homogeneous)
            transformed_points = transformed_points[:-1, :].T

            # Check if the transformation is correct
            is_correct = np.allclose(transformed_points, target_points, atol=1e-5)

            print("Affine Transformation Matrix:\n", transformation_matrix)
            print("Transformed Points:\n", transformed_points)
            print("Target Points:\n", target_points)
            print("Is the transformation correct?:", is_correct)

            # Transform the second set of corner points (corner_points[1]) using the same transformation matrix
            coordinates_homogeneous = np.vstack((coordinates.T, np.ones((1, coordinates.shape[0]))))
            transformed_coordinates = np.dot(transformation_matrix, coordinates_homogeneous)
            transformed_coordinates = transformed_coordinates[:-1, :].T

            print("Refined Transformed Additional Points:\n", transformed_coordinates)
            transformed_coordinates = np.array(transformed_coordinates)
            transformed_mean = np.mean(transformed_coordinates , axis = 0)
            return transformed_mean

@app.get("/capture_image")
def api_capture_image():
    image, message = capture_image()
    #image = image_to_numpy("image6.jpg")
    #message = "Image captured successfully."
    # Check if the image was captured
    print(image)
    if image is not None:
        # Replace the image with "image6.jpg" and update the message
        
        # Convert the new image to base64
        image_base64 = image_to_base64(image)
        return {"message": message, "image": image_base64}
    else:
        raise HTTPException(status_code=500, detail=message)


@app.post("/calculate_points")
def api_calculate_points(data: dict):
    try:
        image_base64 = data.get("image", None)
        if image_base64 is None:
            raise HTTPException(status_code=400, detail="Image is required.")

        # Convert base64 to image
        image = base64_to_image(image_base64)
        
        # Calculate points based on the image
        points = calculate_points(image)
        
        # If points is a NumPy array, convert it to a list
        if isinstance(points, np.ndarray):
            points = points.tolist()

        return {"points": points}
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

