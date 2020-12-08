'''
Matching keypoints 

using Brute Force or FLANN
- Brute Force (from class/tutorial)
- Fast Library for Approximate Nearest Neighbors
    URL: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
'''

import cv2
from helpers import plot_img

#! ----------------------------------------------------------------------------- Match Algs
def get_matches_BF(des1, des2):
    bf = cv2.BFMatcher()
    all_matches = bf.knnMatch(des1, des2, k=2) 
    return all_matches

def get_matches_FLANN(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    all_matches = flann.knnMatch(des1,des2,k=2)
    return all_matches

#! ----------------------------------------------------------------------------- Matching
def getMatches(kp1, des1, kp2, des2, list=True):
    '''
    Returns matches given two sets of keypoints and descriptors
    From Dionysus.A3
    '''

    # Get Matches via
    all_matches = get_matches_BF(des1, des2)

    # Apply ratio test
    good_matches = []
    good_matches_flat = []
    matches_mask = [[0,0] for i in range(len(all_matches))]

    for i,(m,n) in enumerate(all_matches):
        if m.distance < 0.75*n.distance: # only accept matchs that are considerably better than the 2nd best match
            good_matches.append([m])
            good_matches_flat.append(m) # this is to simplify finding a homography later
            matches_mask[i]=[1,0]

    if list:
        return all_matches, good_matches, matches_mask
    else:
        return all_matches, good_matches_flat, matches_mask

#! ----------------------------------------------------------------------------- Helpers

def plot_matches(img1,kp1,img2,kp2,all_matches,matches_mask):
  draw_params = dict(matchColor = (0,255,0),
                     singlePointColor = (255,0,0),
                     matchesMask = matches_mask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)

  img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,all_matches,None,**draw_params)
  plot_img(img3)


#! ----------------------------------------------------------------------------- Testing
if __name__ == "__main__":
    
    from keypoints import getKD, KD
    
    img1 = cv2.imread('imgs/img_clip01.jpg')
    img2 = cv2.imread('imgs/img_clip02.jpg')
    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    kp1, des1 = getKD(KD.SIFT, img1_gray)
    kp2, des2 = getKD(KD.SIFT, img2_gray)

    all_matches, good_matches, matches_mask = getMatches(kp1, des1, kp2, des2, True)
    plot_matches(img1,kp1,img2,kp2,all_matches,matches_mask)
