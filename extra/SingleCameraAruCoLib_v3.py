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


def rotate_marker_corners(rvec, markersize, tvec=None):
    # resource : https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner
    # -coordinates?rq=1

    mhalf = markersize / 2.0
    # convert rot vector to rot matrix both do: markerworld -> cam-world
    mrv, jacobian = cv2.Rodrigues(rvec)

    # in markerworld the corners are all in the xy-plane so z is zero at first
    X = mhalf * mrv[:, 0]  # rotate the x = mhalf
    Y = mhalf * mrv[:, 1]  # rotate the y = mhalf
    minusX = X * (-1)
    minusY = Y * (-1)

    # calculate 4 corners of the marker in cam-world. corners are enumerated clockwise
    markercorners = []
    # was upper left in markerworld
    markercorners.append(np.add(minusX, Y))
    markercorners.append(np.add(X, Y))  # was upper right in markerworld
    # was lower right in markerworld
    markercorners.append(np.add(X, minusY))
    # was lower left in markerworld
    markercorners.append(np.add(minusX, minusY))
    # if tvec given, move all by tvec
    if tvec is not None:
        C = tvec  # center of marker in cam-world
        for i, mc in enumerate(markercorners):
            markercorners[i] = np.add(C, mc)  # add tvec to each corner
    # print('Vec X, Y, C, dot(X,Y)', X,Y,C, np.dot(X,Y)) # just for debug
    # type needed when used as input to cv2
    markercorners = np.array(markercorners, dtype=np.float32)
    markercorners = np.squeeze(markercorners)  # 4x3
    return markercorners, mrv


class singleCameraAruco:

    def __init__(self, aruco_ids_list,  aruco_size, AruCo_DICT_KEY, camera_matrix, distortion_matrix):
        self.world_points = None
        self.mid_points = None
        self.aruco_ids = np.asarray(aruco_ids_list)
        self.camera_matrix = camera_matrix
        self.distortion_matrix = distortion_matrix
        #self.aruco_size = aruco_size
        self.aruco_size = aruco_size        
        self.AruCo_DICT_KEY = AruCo_DICT_KEY
        self.tool_vector = aruco_ids_list

    def set_id(self, m_id):
        if type(m_id) == list:
            self.aruco_ids = np.array(m_id)

        else:
            raise Exception('Aruco ID must be in form of list')

    # not used

    def get_world_coordinates(self, image, draw_corners=False, draw_axis=False):

        self.mid_points = []
        self.world_points = []
        self.is_detected = []
        #key = findDictionary(4, 50)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key = getattr(cv2.aruco, f'{self.AruCo_DICT_KEY}')
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(key) # Changed danger
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
        
        #print('2D CORNER SHAPE:',np.asarray(corners).shape)
        #print('ID        SHAPE:',ids)
                                                              
        # cv2.aruco.drawDetectedMarkers(image, corners)
        
        if len(corners) == 0:
            self.world_points = [None]*self.aruco_ids.shape[0]
            self.is_detected = [False]*self.aruco_ids.shape[0]

        if len(corners) > 0:
            ids = np.squeeze(ids)

            for i in range(self.aruco_ids.shape[0]):
                index_arr = np.where(ids == self.aruco_ids[i])
                index_arr = np.array(index_arr)

                if index_arr.shape[1] == 0:
                    self.mid_points.append(None)
                    self.world_points.append(None)
                    self.is_detected.append(False)
                else:
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[index_arr[0][0]],
                                                                                   self.aruco_size, self.camera_matrix,
                                                                                   self.distortion_matrix)
                    
                    #print('INPUT to estimatePose:', corners[index_arr[0][0]].shape)

                    markercorners, mrv = rotate_marker_corners(
                        rvec, self.aruco_size, tvec)
                    self.mid_points.append(tvec)
                    self.world_points.append(markercorners)
                    self.is_detected.append(True)

                    if draw_corners:
                        cv2.aruco.drawDetectedMarkers(image, corners)

                    if draw_axis:
                        cv2.aruco.drawAxis(image, self.camera_matrix,
                                           self.distortion_matrix, rvec, tvec, 0.01)
        cv2.aruco.drawDetectedMarkers(image, corners)                
        if draw_corners or draw_axis:
            return self.world_points, self.mid_points, image, corners, ids
        else:
            return self.is_detected, self.world_points


def drawCornersAndIds(corners, ids, image, dist):
    if corners.shape[0] == 2 and ids.shape[0] == 2:
        if ids is not None and corners is not None:
            for mis, mcs in zip(ids, corners):
                # corners = corners.reshape((4, 2))
                cors = mcs

                if cors.shape[1] == 2:
                    topleft = cors[3]
                    topright = cors[2]
                    bottomleft = cors[0]
                    bottomright = cors[1]
                    topleft = np.squeeze(topleft)
                    topright = np.squeeze(topright)
                    bottomleft = np.squeeze(bottomleft)
                    bottomright = np.squeeze(bottomright)

                    mid = (topleft + bottomright) // 2
                    topright = (int(topright[0]), int(topright[1]))
                    bottomright = (int(bottomright[0]), int(bottomright[1]))
                    bottomleft = (int(bottomleft[0]), int(bottomleft[1]))
                    topleft = (int(topleft[0]), int(topleft[1]))
                    middle = (int(mid[0]), int(mid[1]))

                    cv2.circle(image, (topright[0], topright[1]), 8, (0, 255, 255), -1)
                    cv2.circle(image, (bottomright[0], bottomright[1]), 8, (255, 0, 255), -1)
                    cv2.circle(image, (bottomleft[0], bottomleft[1]), 8, (0, 69, 255), -1) 
                    cv2.circle(image, (topleft[0], topleft[1]), 8, (255, 255, 0), -1)

                    cX = int((topleft[0] + bottomright[0]) / 2.0)
                    cY = int((topleft[1] + bottomright[1]) / 2.0)
                    cv2.circle(image, (cX, cY), 4, (255, 0, 0), -1)
                    cv2.putText(image, str(mis),
                                (bottomleft[0], bottomleft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 3)
                    cv2.line(image, topleft, middle, (0, 255, 0), 1)
                    cv2.line(image, topright, middle, (0, 255, 0), 1)
                    cv2.line(image, bottomleft, middle, (0, 255, 0), 1)
                    cv2.line(image, bottomright, middle, (0, 255, 0), 1)

            crs = np.squeeze(corners)
            if np.squeeze(corners).shape[1] == 4:
                tl1 = crs[0][3]
                br1 = crs[0][1]

                tl2 = crs[1][3]
                br2 = crs[1][1]

                tl1 = np.squeeze(tl1)
                tl2 = np.squeeze(tl2)

                br1 = np.squeeze(br1)
                br2 = np.squeeze(br2)

                m1 = (tl1 + br1) // 2
                m2 = (tl2 + br2) // 2
                midMid = (m1 + m2) // 2

                cv2.line(image, (int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1])), (255, 255, 0), 3)

                cv2.putText(image, str(round(dist * 100, 4)) + " cms", (int(midMid[0]), int(midMid[1] - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return image



class SingleCamCapture_USB():
    
    def __init__(self, cam_id, img_H, img_W):
        
       
        self.cam_id = cam_id
        self.img_H = img_H
        self.img_W = img_W
        
        options = {"CAP_PROP_FRAME_WIDTH": img_W, 
                   "CAP_PROP_FRAME_HEIGHT": img_H
                  }
        
        self.cap = CamGear(source=cam_id, **options).start()
        
 
        ramp_frames = 5# initial images to discard to adjust the camera
        for i in range(ramp_frames):
            frame = self.cap.read()            
            time.sleep(0.01)
            print("camera resoultion:", frame.shape)
                                    
    def get_FRAME(self):
        frame = self.cap.read()
        if frame is not None:
            ret = True
        else:
            ret = False           
        #print(frame.shape)
        #frame = cv2.flip(frame, 0) # 0 for vertical flip        
        return ret, frame 
    
    def get_FRAME_GRAY(self):
        ret, frame = self.get_FRAME()    
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        return ret, frame
    
    def check_capture(self):
        ret, frame = self.get_stereo_FRAME_SET()
        plt.imshow(frame[::2,::2,:])
        plt.show()

def calculate_distance(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    z1 = point1[2]
    z2 = point2[2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return d


def loadMats(camPath,matPath):
    with open(camPath, 'rb') as f:
        CameraMatrix = np.load(f)
    with open(matPath, 'rb') as f:
        distCoeffs = np.load(f)

    return CameraMatrix, distCoeffs


def setEnv(camMat, dist):
    obj = singleCameraAruco(camMat, dist, 0.06)
    ids = [5, 9]
    obj.set_id(ids)

    return obj


def getDist(obj, img):
    world_points, mid_points, img, corners, mids = obj.get_world_coordinates(
        img, True, True)
    
# --------------------------------------Testing--------------------------------------
    # for i in range(len(world_points)):                  # printing the distance between the markers
        # print("world_points: ", world_points[i].shape)
        # print("marker: ",pdist(world_points[i] )*1000)
    marker1_mid_point = (world_points[0][0]+world_points[0][2])/2
    marker2_mid_point = (world_points[1][0]+world_points[1][2])/2
    print("Distance Mid: ", calculate_distance(marker1_mid_point, marker2_mid_point)*1000)
    print("marker_pair: \n",cdist(world_points[0], world_points[1] )*1000)
    print("\n")    
# ------------------------------------------------------------------------------------

    points = []
    if len(obj.aruco_ids) == 2:
        if len(mid_points) == 2:
            if mid_points[0] is not None:
                if (mid_points[0] is not None) & (mid_points[1] is not None):
                    points.append(np.squeeze(mid_points[0]))
                    points.append(np.squeeze(mid_points[1]))

                    if (points[0] is not None) & (points[1] is not None) & (np.squeeze(corners).shape[0] == 2):
                        image = drawCornersAndIds(np.squeeze(corners), mids,
                                                  img, calculate_distance(points[0], points[1]))

                        return image

    return None
