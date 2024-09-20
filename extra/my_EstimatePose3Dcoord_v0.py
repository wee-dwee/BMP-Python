import cv2
from vidgear.gears import CamGear
import numpy as np
import math
from scipy.spatial.distance import pdist, cdist
import time
import matplotlib.pyplot as plt

def findDictionary(markersize=4, totalMarkers=50):
    key = getattr(
        cv2.aruco, f'DICT_{markersize}X{markersize}_{totalMarkers}')
    return key


def match_and_index_ids(target_ids, detected_ids):
    
    target_ids = np.asarray(target_ids,dtype=int)
    #detected_ids = np.asarray(detected_ids)
    if detected_ids is None: 
       detected_ids = np.asarray([None]) 
    elif len(detected_ids)==1 and len(detected_ids[0])==1: #[[10]]
        detected_ids = np.asarray([detected_ids[0][0]], dtype=int) 
    else:
        detected_ids = np.asarray(detected_ids,dtype=int)
        detected_ids = np.squeeze(detected_ids)
    
    
    
    print("target_ids:", target_ids)
    print("detected_ids:", detected_ids)
    
    if set(target_ids).issubset(set(detected_ids)):
        ret = True
        idx = np.zeros((target_ids.size),dtype=int)
        for i in range(target_ids.size):
            idx[i] = np.squeeze(np.argwhere(detected_ids==target_ids[i])[0])    
            
        idx = np.asarray(idx,dtype=np.int64())
    else:
        ret = False
        idx = None        
    return ret, idx



class CustomEstimatePose:
    
    def __init__(self, objectPoints=None, cameraMatrix=None, distCoeffs=None):
        self.objectPoints = objectPoints   #objectPoints Nx3 numpy array
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
    def estimatePose(self, imagePoints):
        _ ,rvec, tvec =  cv2.solvePnP(self.objectPoints, imagePoints, self.cameraMatrix, self.distCoeffs)
        return rvec, tvec
        
    def detectMarkers(self, gray):
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)       
        return np.asarray(corners), ids, rejected_img_points
    
    def get_world_coordinates(self, imagePoints): #imagePoints Nx2 numpy array
        
        _ ,rvec, tvec =  cv2.solvePnP(self.objectPoints, imagePoints, self.cameraMatrix, self.distCoeffs,flags=cv2.SOLVEPNP_IPPE)
        mrv, jacobian = cv2.Rodrigues(rvec)
        wp = self.objectPoints @ mrv.T + tvec.T
        return wp
    




def reshape_corners(corners):
    reshaped_corners = np.squeeze(corners[0])
    for i in range(1,corners.shape[0]):
        reshaped_corners = np.concatenate((reshaped_corners, np.squeeze(corners[i])), axis=0)
    return reshaped_corners
        
        
def generate_object_point_FourAruCo(aruco_size, gap_size):
    ac = np.asarray([[-aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0, -aruco_size/2.0],
                     [-aruco_size/2.0, -aruco_size/2.0]])
    
    temp = ac - np.asarray([ (aruco_size+gap_size)/2.0,  -(aruco_size+gap_size)/2.0])  # UL
    objectPoints = temp
    temp = ac - np.asarray([-(aruco_size+gap_size)/2.0,  -(aruco_size+gap_size)/2.0])  # UR
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    temp = ac - np.asarray([-(aruco_size+gap_size)/2.0,   (aruco_size+gap_size)/2.0])  # LR
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    temp = ac - np.asarray([ (aruco_size+gap_size)/2.0,   (aruco_size+gap_size)/2.0])  # LL
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    
    objectPoints = np.concatenate((objectPoints, np.zeros((16,1))),axis=1)
    return objectPoints.astype('float32')



def generate_object_point_ThreeAruCo(aruco_size, gap_size):
    ac = np.asarray([[-aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0, -aruco_size/2.0],
                     [-aruco_size/2.0, -aruco_size/2.0]])
    
    temp = ac - np.asarray([ (aruco_size+gap_size)/2.0,  -(aruco_size+gap_size)/2.0])  # UL
    objectPoints = temp
    temp = ac - np.asarray([-(aruco_size+gap_size)/2.0,  -(aruco_size+gap_size)/2.0])  # UR
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    temp = ac - np.asarray([-(aruco_size+gap_size)/2.0,   (aruco_size+gap_size)/2.0])  # LR
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    
    objectPoints = np.concatenate((objectPoints, np.zeros((12,1))),axis=1)
    return objectPoints.astype('float32')



def generate_object_point_TwoAruCo(aruco_size, gap_size):
    ac = np.asarray([[-aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0, -aruco_size/2.0],
                     [-aruco_size/2.0, -aruco_size/2.0]])
    
    temp = ac - np.asarray([0,   -(aruco_size+gap_size)/2.0])  # U
    objectPoints = temp
    temp = ac - np.asarray([0,    (aruco_size+gap_size)/2.0])  # L
    objectPoints = np.concatenate((objectPoints, temp),axis=0)
    objectPoints = np.concatenate((objectPoints, np.zeros((8,1))),axis=1)
    return objectPoints.astype('float32')

def generate_object_point_SingleAruCo(aruco_size):
    objectPoints = np.asarray([[-aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0,  aruco_size/2.0],
                     [ aruco_size/2.0, -aruco_size/2.0],
                     [-aruco_size/2.0, -aruco_size/2.0]])
    
    objectPoints = np.concatenate((objectPoints, np.zeros((4,1))),axis=1)
    
    return objectPoints.astype('float32')


