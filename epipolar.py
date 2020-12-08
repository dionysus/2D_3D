'''
Epipolar functions
for determining 3D points given matches

https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
'''
import cv2
import numpy as np

def get_fundamental_matrix(kp1, kp2, good_matches):
  '''
  Return the Fundamental Matrix given a list of good matches.

  With reference to:
  https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
  '''
  src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
  dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

  F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)

  src_pts = src_pts[mask.ravel()==1]
  dst_pts = dst_pts[mask.ravel()==1]

  return F, src_pts, dst_pts

