import numpy as np
import cv2

def undistort_img(img):
  '''
  Return an undistorted image given previous calibrated parameters 
  '''
  ret   = np.load('./calibrated_params/ret.npy')
  K     = np.load('./calibrated_params/K.npy')
  dist  = np.load('./calibrated_params/dist.npy')
  h,w   = img.shape[:2]

  new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
  img_undistorted = cv2.undistort(img, K, dist, None, new_camera_matrix)

  return img_undistorted