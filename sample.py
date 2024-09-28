import cv2
import numpy as np

# Define camera calibration parameters (replace these with your calibration values)
camera_matrix = np.array([[1406.08415449821, 0, 0],
                          [2.20679787308599, 1417.99930662800, 0],
                          [1014.13643417416, 566.347754321696, 1]])
dist_coeffs = np.array([-0.297535464236523, 0.127671761468177, 
                        0.000919612740592, -0.001056550874274, 
                        -0.060517931893662])

# Size of the marker in your scene (in meters)
marker_length = 0.1  # example: 10 cm marker

# Load the dictionary and detector parameters
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Load your image
image = cv2.imread('image1.jpg')  # Replace with your image path

# Detect markers in the image
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

# Draw detected markers on the image
image_with_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

# Estimate the pose of the marker
if len(corners) > 0:
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

    # Draw the pose of each marker on the image
    for rvec, tvec in zip(rvecs, tvecs):
        cv2.aruco.drawAxis(image_with_markers, camera_matrix, dist_coeffs, rvec, tvec, marker_length)
    
    # Display image with detected markers and pose
    cv2.imshow('Detected Markers and Pose', image_with_markers)
    cv2.waitKey(0)

    # Example transformation of a point from marker to camera coordinates
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
    marker_point = np.array([[0], [0], [0]])  # Origin of the marker's coordinate system

    # Transform point to camera coordinates
    camera_point = np.dot(rotation_matrix, marker_point) + tvecs[0].reshape(3, 1)
    print("Camera Coordinates of Marker Center:", camera_point.ravel())

    # Example transformation of a point from camera to marker coordinates
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    example_point_camera = np.array([[0.1], [0.2], [0.5]])  # Example point in camera coordinates

    # Transform from camera to marker coordinates
    marker_point_from_camera = np.dot(rotation_matrix_inv, example_point_camera - tvecs[0].reshape(3, 1))
    print("Marker Coordinates of Example Point:", marker_point_from_camera.ravel())

else:
    print("No markers detected.")

cv2.destroyAllWindows()
