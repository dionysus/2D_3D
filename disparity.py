'''
from:
https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
'''

import numpy as np
import cv2
from helpers import *

def get_disparity_map(img1, img2):
  win_size = 5
  min_disp = -1
  max_disp = 63 #min_disp * 9
  num_disp = max_disp - min_disp # Needs to be divisible by 16

  #Create Block matching object. 
  stereo = cv2.StereoSGBM_create(
    minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    uniquenessRatio = 5,
    speckleWindowSize = 5,
    speckleRange = 5,
    disp12MaxDiff = 1,
    P1 = 8*3*win_size**2,#8*3*win_size**2,
    P2 = 32*3*win_size**2) #32*3*win_size**2)

  #Compute disparity map
  print ("\nComputing the disparity map...")
  disparity_map = stereo.compute(img1, img2)
  return disparity_map

if __name__ == "__main__":

  img1 = cv2.imread("imgs/img01.jpg")
  img2 = cv2.imread("imgs/img02.jpg")

  img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

  img1_undistorted = undistort_img(img1_gray)
  img2_undistorted = undistort_img(img2_gray)

  plot_img(img1_undistorted)

  scale = 0.5
  dim = (int(img1_undistorted.shape[1] * scale), int(img_1_undistorted.shape[0] * scale))
  img1_small = cv2.resize(img1_undistorted, dim)
  img2_small = cv2.resize(img2_undistorted, dim)

  disparity_map = get_disparity_map(img1_small,img2_small)

  #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
  plt.imshow(disparity_map,'gray')
  plt.show()