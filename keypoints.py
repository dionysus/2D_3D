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
    # gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    
    return kp, des


if __name__ == "__main__":
    # test and plot keypoints of an image of the RoM building
    img = cv.imread('imgs/rom.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp, des = getKD(KD.SIFT, gray)
    
    imgA=cv.drawKeypoints(gray,kp,img)
    plot_img(img)
    
    imgB=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plot_img(img)