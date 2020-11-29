import cv2
from helpers import plot_img
from keypoints import getKD, KD
from match import getMatches, plot_matches
from calibrate import calibrate_camera
from undistort import undistort_img
from disparity import get_disparity_map
from cloud import get_point_cloud

if __name__ == "__main__":

  #! 1) calibrate camera with chessboard
  # note: take a few pictures of the OpenCV chessboard with source camera
  calibrate_folder = 'calibrate'
  calibrate_camera(calibrate_folder)

  # open target images
  img1 = cv2.imread('imgs/img_clip01.jpg', 0)
  img2 = cv2.imread('imgs/img_clip02.jpg', 0)

  #! 2) undistort images
  img1_undistorted = undistort_img(img1)
  img2_undistorted = undistort_img(img2)
  # plot_img(img1_undistorted)
  # plot_img(img2_undistorted)

  # shrink
  scale = 0.5
  dim = (int(img1_undistorted.shape[1] * scale), int(img1_undistorted.shape[0] * scale))
  img1_small = cv2.resize(img1_undistorted, dim)
  img2_small = cv2.resize(img2_undistorted, dim)

  # STEREO METHOD
  # https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
  # get disparity_map - this is not working
  disparity_map = get_disparity_map(img1_small,img2_small)
  # plot_img(disparity_map)
  get_point_cloud(img1_small, disparity_map)
  # I then can open this in a 3D modeling software

  # OTHER METHOD
  # get keypoint and descriptors
  kp1, des1 = getKD(KD.SIFT, img1_small)
  kp2, des2 = getKD(KD.SIFT, img2_small)

  # get matches
  matches, matchesMask = getMatches(kp1, des1, kp2, des2)
  plot_matches(img1_small,kp1,img2_small,kp2,matches,matchesMask)
  
  #TODO: get Depth Map
  depth_map = get_depthmap()
  # camera stuff
  # Homograpy

  get_point_cloud(img1_small, depth_map)
