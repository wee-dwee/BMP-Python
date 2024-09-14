# imagecap.py

import cv2

# Global variable to store the captured image
captured_image = None
points = []

def capture_image():
    global captured_image

    # Open the webcam (usually the first camera is at index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read a single frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Store the frame in the global variable
        captured_image = frame
        print("Image captured successfully.")
    else:
        print("Error: Could not capture image.")

    # Release the webcam
    cap.release()

def show_image():
    global captured_image

    if captured_image is None:
        print("Error: No image to display. Capture an image first.")
        return

    # Display the captured image in a window
    cv2.imshow("Captured Image", captured_image)

    # Wait for any key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mouse callback function to capture points on mouse click
def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append point (x, y) to points list
        points.append((x, y))
        print(f"Point captured: ({x}, {y})")

        # Display the point on the image (optional)
        cv2.circle(captured_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Captured Image", captured_image)

def extract_2d_points():
    global captured_image, points

    if captured_image is None:
        print("Error: No image to extract points from. Capture an image first.")
        return

    points.clear()  # Clear any previously stored points

    # Display the image and set the mouse callback to capture clicks
    cv2.imshow("Captured Image", captured_image)
    cv2.setMouseCallback("Captured Image", click_event)

    # Wait until the user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

if __name__ == "__main__":
    capture_image()
    extract_2d_points()
    print("Extracted 2D points:", points)
