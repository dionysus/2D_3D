import numpy as np
import cv2

from camera import load_K, load_camera_dist, load_camera_ret

def undistort_img(img):
  '''
  Return an undistorted image given previous calibrated parameters 
  References from OpenCV docs
  '''
  ret   = load_camera_ret()
  K     = load_K()
  dist  = load_camera_dist()
  h,w   = img.shape[:2]

  new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
  img_undistorted = cv2.undistort(img, K, dist, None, new_camera_matrix)

  return img_undistorted
