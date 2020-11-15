import cv2 as cv
from helpers import plot_img
from keypoints import getKD, KD
from match import getMatches

# open images
img1 = cv.imread('imgs/rom3.jpg')
img2 = cv.imread('imgs/rom4.jpg')

# get keypoint and descriptors
kp1, des1 = getKD(KD.SIFT, img1)
kp2, des2 = getKD(KD.SIFT, img2)

# get matches
matches, matchesMask = getMatches(kp1, des1, kp2, des2)

# draw matches
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plot_img(img3)
