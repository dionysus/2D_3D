'''
Calibrate with chessboard from source camera

Adapted from openCV docs
https://docs.opencv.org/master/d9/dab/tutorial_homography.html
With further explanations gained from
https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-ii-77754b58bfe0
'''

import numpy as np
import cv2
import glob
import PIL.ExifTags
import PIL.Image
from helpers import *

def calibrate_camera():
  '''
  Calibrate and save settings for source camera
  '''
  # chessboard dimensions

  board_h = 9
  board_w = 6
  board_dims = (board_h,board_w)

  # termination criteria for cornerSubPix
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # prepare object points
  objp = np.zeros((board_h * board_w,3), np.float32)
  objp[:,:2] = np.mgrid[0:board_h,0:board_w].T.reshape(-1,2)

  obj_points = [] # 3d point in real world space
  img_points = [] # 2d points in image plane.

  calibration_images = glob.glob("calibrate/*")
  print(calibration_images)
  # images to calibrate from
  for image in calibration_images:
      # process calibration image with chessboard corners
      img = cv2.imread(image)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      ret, corners = cv2.findChessboardCorners(gray, board_dims, None)

      if not ret:
        print("no checkerboard found for: {}".format(image))
      else:
          print("checkerboard found for: {}".format(image))
          # refine corner detection
          sub_corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
          img_points.append(sub_corners)
          obj_points.append(objp)

          # Draw and display the corners
          display_img = cv2.drawChessboardCorners(img, board_dims, sub_corners, ret)
          # plot_img(display_img)

  # run calibrate camera with processed images
  ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                              obj_points, img_points,gray.shape[::-1], None, None)

  # Get Focal Length of source camera
  exif_img = PIL.Image.open(calibration_images[0])
  exif_data = {
  PIL.ExifTags.TAGS[k]:v
  for k, v in exif_img._getexif().items()
  if k in PIL.ExifTags.TAGS}
  focal_length = exif_data['FocalLength']
  print("focal length: {}".format(focal_length))

  # save calibration settings to file
  np.save("./calibrated_params/ret", ret)
  np.save("./calibrated_params/K", K)
  np.save("./calibrated_params/dist", dist)
  np.save("./calibrated_params/rvecs", rvecs)
  np.save("./calibrated_params/tvecs", tvecs)
  np.save("./calibrated_params/FocalLength", focal_length)

def load_K():
  K = np.load('./calibrated_params/K.npy')
  return K