'''
Matching keypoints using FLANN
- Fast Library for Approximate Nearest Neighbors

URL: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
* just testing out the implementations from the OPENCV docs 
'''

import cv2
from helpers import plot_img

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

def getMatches(kp1, des1, kp2, des2):
    '''
    Returns matches given two sets of keypoints and descriptors
    '''
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    return matches, matchesMask

def plot_matches(img1,kp1,img2,kp2,matches,matchesMask):
  draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matchesMask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)

  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
  plot_img(img3)

if __name__ == "__main__":
    
    from keypoints import getKD, KD
    
    img1 = cv2.imread('imgs/rom3.jpg')
    img2 = cv2.imread('imgs/rom4.jpg')

    kp1, des1 = getKD(KD.SIFT, img1)
    kp2, des2 = getKD(KD.SIFT, img2)

    matches, matchesMask = getMatches(kp1, des1, kp2, des2)

    # draw matches
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plot_img(img3)
