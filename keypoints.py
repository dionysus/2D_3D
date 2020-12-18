'''
Get Keypoints using various algorithms
- SIFT
'''

import cv2 as cv
from helpers import plot_img
from enum import Enum

class KD(Enum):
    SIFT = 1

def getKD(method, img):
    '''
    Algorithm selector
    '''
    if method == KD.SIFT:
        return getSIFT(img)
    else:
        return getSIFT(img)

def getSIFT(img):
    '''
    Returns SIFT keypoints
    From: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    '''
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    return kp, des
