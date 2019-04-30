import cv2
import numpy as np
import glob
import pickle

sift = cv2.xfeatures2d.SIFT_create()


def featureDes(img):
    
    kp1, des1 = sift.detectAndCompute(img, None)
    i=0
    arr=[]
    for point in kp1:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        ++i
        arr.append(temp)
    return arr;
def unpickle(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

image = cv2.imread("jfe.jpg")
k = featureDes(image)
f = open("keypoints.txt", "w")
f.write(pickle.dumps(k))
f.close()