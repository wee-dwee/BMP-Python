import numpy as np
import json
import cv2
import Trans3D
from scipy.spatial import procrustes
from SingleCameraAruCoLib_v3 import singleCameraAruco

def read_json(file_name):
    with open(file_name) as fh:
        system_config = json.load(fh)
    return system_config

# Function to perform Procrustes transformation
def procrustes_transform(points, target_points):
    # Procrustes analysis to minimize the difference between two sets of points
    mtx1, mtx2, disparity = procrustes(target_points, points)
    return mtx1, disparity

if __name__ == '__main__':
    
    # Reading system configuration
    system_config = read_json("Config_SingleCamUSB.json")
    
    print('System Initializing...')
    
    # Path to the image file
    image_file_path = r'image2.jpg'
    
    # Initialize the single camera Aruco detector
    SingleCamAruco_obj = singleCameraAruco(aruco_ids_list=system_config['aruco_ids'], 
                                           aruco_size=system_config['aruco_size'],
                                           AruCo_DICT_KEY=system_config['aruco_dictionary'],
                                           camera_matrix=np.load(system_config['camera_matrix_file']),
                                           distortion_matrix=np.load(system_config['distortion_matrix_file']))

    # Load the image
    frame = cv2.imread(image_file_path)

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

            # Now use Procrustes to refine the transformation
            refined_transformed_coordinates, disparity = procrustes_transform(transformed_coordinates, target_points)

            # Output the transformed and rescaled coordinates
            print("Refined Transformed Additional Points:\n", refined_transformed_coordinates)
            print("Disparity (error measure):", disparity)
        
        else:
            print("Not all markers are detected.")
        
        # Show the processed image with markers
        cv2.imshow('aruco_Tracking', cv2.resize(frame, (640, 480)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('Processing completed.')
