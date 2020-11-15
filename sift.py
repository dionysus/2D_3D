import cv2 as cv
from helpers import plot_img

def getSIFT(img):
    '''
    Returns SIFT keypoints
    
    From: https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
    '''
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    
    return kp


if __name__ == "__main__":
    # test and plot keypoints of an image of the RoM building
    img = cv.imread('rom.jpg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp = getSIFT(img)
    
    imgA=cv.drawKeypoints(gray,kp,img)
    plot_img(img)
    
    imgB=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plot_img(img)