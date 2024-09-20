

import numpy as np
import json
import math
import cv2
from SingleCameraAruCoLib_v3 import singleCameraAruco, SingleCamCapture_USB
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

def anlge_between_two_vector(v1,v2):
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


# use svd for plane carefully.......
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

    '''
    system_config = {
    "cam_id" : 0,
    "cal_parameter_file" : "elp_stereo_cal_1080x1920_A1.pickle",
    "cal_checkboard_size" : 76.7, 
    "image_size_H" : 1080,
    "image_size_W" : 1920,
    "aruco_ids" : [5,9],
    "aruco_dictionary" : "DICT_4X4_50"
    }
    '''
    
    system_config = read_json("Config_SingleCamUSB.json")
    
    #print(system_config)
    

    print('System Initializing...')
    
    
    video_file_name = 'WIN_20240828_23_08_35_Pro.mp4'
    file_path = r'C:\Users\JEET\Pictures\Camera Roll'
    
    system_config['cam_id'] = file_path + '\\' + video_file_name
    
    # for aruco marker corner detection and tringulation using streoObj
    SingleCamAruco_obj = singleCameraAruco(aruco_ids_list = system_config['aruco_ids'], 
                                           aruco_size = system_config['aruco_size'],
                                           AruCo_DICT_KEY = system_config['aruco_dictionary'],
                                           camera_matrix = np.load(system_config['camera_matrix_file']),
                                           distortion_matrix = np.load(system_config['distortion_matrix_file']))
    
    # for caputre stereo frame
    SingleCamFrameCap_Obj = SingleCamCapture_USB(cam_id = system_config['cam_id'], 
                                                 img_H = system_config['image_size_H'], 
                                                 img_W = system_config['image_size_W'])
    
    print('System Initializing..done')
    
    np.set_printoptions(precision=4)
    count = 0
    d_vect = []
    angle_vect = []
    while True:
        ret, frame = SingleCamFrameCap_Obj.get_FRAME()
        
        if ret:
            count = count + 1
        else:
            break
        
        # corner_point list [array, array] for two aruco markers
        is_marker_detected, corner_points = SingleCamAruco_obj.get_world_coordinates(frame)
        
        corner_points = corner_points # meter to mm
        
        #print((corner_points[0]))


        #if( (corner_points[0] is not None) and (corner_points[1] is not None)):
        if(all(is_marker_detected) ):
            print(count, "  ----------------------------------------------------------") 
        
            #distance between two aruco markers

            
            d = np.linalg.norm(np.mean(corner_points[0],axis=0) - np.mean(corner_points[1],axis=0))
            angle_degree = angle_between_two_aruco(corner_points[0],corner_points[1])
            print("Distance between Marker: " , d) 
            print("angle between two aruco 1:", angle_degree)
            
            d_vect.append(d)
            angle_vect.append(angle_degree)
              
            '''
            #cross corner to corner distance of individual aruco marker            
            d0_02 = calculate_distance(point1=corner_points[0][0] , point2 = corner_points[0][2]) 
            d0_13 = calculate_distance(point1=corner_points[0][1] , point2 = corner_points[0][3]) 
            d1_02 = calculate_distance(point1=corner_points[1][0] , point2 = corner_points[1][2]) 
            d1_13 = calculate_distance(point1=corner_points[1][1] , point2 = corner_points[1][3]) 
            print("dist_cross_0:", d0_02, d0_13)
            print("dist_cross_1:", d1_02, d1_13)
            
            # angle between two aruco markers (planes)
            a = angle_between_two_aruco_svd(corner_points[0] , corner_points[1]) #angle between two aruco
            print("angle : " , a)  
            
            #distance of aruco marker from camera
            dwc0 = np.linalg.norm(np.mean(corner_points[0],axis=0))
            dwc1 = np.linalg.norm(np.mean(corner_points[1],axis=0))
            
            print('dist from camera 0:', dwc0)
            print('dist from camera 1:', dwc1)
            '''
        else:
            print("all marker are not detected...")
            
        cv2.imshow('aruco_Tracking',cv2.resize(frame,(640,480)))
        #time.sleep(0.1)
        #break while loop when keyboad interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        
    
    

    