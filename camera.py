'''
Camera calibration functions

Adapted from openCV docs
https://docs.opencv.org/master/d9/dab/tutorial_homography.html

With further explanations of chessboard gained from
https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-ii-77754b58bfe0
'''

import numpy as np
import cv2
import glob
import math
import PIL.ExifTags
import PIL.Image
from helpers import *

from cloud import plot_point_cloud_colorless

def calibrate_camera():
  '''
  Calibrate and save settings for source camera

  Referencing OpenCV docs 
    for use of chessboard calibration
  '''
  # set chessboard dimensions
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
  # print(calibration_images)

  # images to calibrate from
  for image in calibration_images:
      # process calibration image with chessboard corners
      img = cv2.imread(image, 0)
      ret, corners = cv2.findChessboardCorners(img, board_dims, None)

      if not ret:
        print("no checkerboard found for: {}".format(image))
      else:
          print("checkerboard found for: {}".format(image))
          # refine corner detection
          sub_corners = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
          img_points.append(sub_corners)
          obj_points.append(objp)

          # Draw and display the corners
          display_img = cv2.drawChessboardCorners(img, board_dims, sub_corners, ret)
          # plot_img(display_img)

  # run calibrate camera with processed images
  ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                              obj_points, img_points,img.shape[::-1], None, None)

  # Get Focal Length of source camera from image metadata
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
  np.save("./calibrated_params/FocalLength", focal_length)


def save_camera_props(num_cameras, degree, distance, height):
  '''
  Return a list of camera coordinates, given Turntable properties. 
  '''
  camera_props = []

  # first camera 
  curr_coord = [0, height, distance]
  curr_angle = 0

  # (x, y, z, rotation)
  curr_camera_prop = curr_coord + [curr_angle]
  camera_props.append(curr_camera_prop)

  for i in range(1, num_cameras):
    curr_angle += degree
    curr_coord = _rotate_coord(curr_coord, degree)
    curr_camera_prop = curr_coord + [curr_angle]
    camera_props.append(curr_camera_prop)
  
  print(camera_props)
  camera_props = np.array(camera_props)

  np.save("./calibrated_params/camera_props", camera_props)

def _rotate_coord(coord, degree):
  '''
  Returns the new coord given a degree rotation.
  '''
  x, y, z = coord[0], coord[1], coord[2]
  rot = np.radians(degree)

  x_new = x * np.cos(rot) + z * np.sin(rot)
  z_new = -x * np.sin(rot) + z * np.cos(rot)

  return [x_new, y, z_new]

#! ------------------------------------- load previously saved camera properties

def load_camera_props():
  camera_props = np.load('./calibrated_params/camera_props.npy')
  return camera_props

def load_K():
  K = np.load('./calibrated_params/K.npy')
  return K

def load_camera_ret():
  ret   = np.load('./calibrated_params/ret.npy')
  return ret

def load_camera_dist():
  dist  = np.load('./calibrated_params/dist.npy')
  return dist

if __name__ == "__main__":
  num_cameras = 18
  degree = 20
  distance = 175
  height = 0

  # save_camera_props(num_cameras,degree,distance,height)
  camera_props = load_camera_props()

  # check that it makes a full circle
  camera_coords = camera_props[:, :3]
  print(camera_props)
  plot_point_cloud_colorless(camera_coords)