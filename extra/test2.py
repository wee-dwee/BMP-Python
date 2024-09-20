import numpy as np
import json
import math
import cv2
from SingleCameraAruCoLib_v3 import singleCameraAruco
import time

def calculate_distance(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    z1 = point1[2]
    z2 = point2[2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return d

def angle_between_two_vector(v1,v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = np.rad2deg(angle_rad)
    return angle_degree

def angle_between_two_aruco(d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)

    n1 = np.cross(d1[0,:]-d1[2,:],d1[1,:]-d1[3,:])    
    n2 = np.cross(d2[0,:]-d2[2,:],d2[1, :]-d2[3,:])   
    v1_u = n1 / np.linalg.norm(n1)
    v2_u = n2 / np.linalg.norm(n2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = np.rad2deg(angle_rad)
    return angle_degree

# use svd for plane carefully
def angle_between_two_aruco_svd(d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)

    u, s, vh = np.linalg.svd(d1 - np.mean(d1,axis=0))
    v1_u = vh[2, :]
    u, s, vh = np.linalg.svd(d1 - np.mean(d2,axis=0))
    v2_u = vh[2, :]
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = np.rad2deg(angle_rad)
    if angle_degree > 90:
        angle_degree = 180 - angle_degree
    return angle_degree

def read_json(file_name):
    with open(file_name) as fh:
        system_config = json.load(fh)
    return system_config

if __name__ == '__main__':
    
    # Reading system configuration
    system_config = read_json("Config_SingleCamUSB.json")
    
    print('System Initializing...')
    
    # Path to the image file
    image_file_path = r'image3.jpg'
    
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
        
        if all(is_marker_detected):
            print("Both markers detected.")
            
            # Distance between two Aruco markers
            d = np.linalg.norm(np.mean(corner_points[0], axis=0) - np.mean(corner_points[1], axis=0))
            angle_degree = angle_between_two_aruco(corner_points[0], corner_points[1])
            print("Distance between markers:", d)
            print("Angle between two Aruco markers:", angle_degree)
        
        else:
            print("Not all markers are detected.")
        
        # Show the processed image with markers
        cv2.imshow('aruco_Tracking', cv2.resize(frame, (640, 480)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('Processing completed.')
