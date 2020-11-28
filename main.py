import cv2
from helpers import plot_img
from keypoints import getKD, KD
from match import getMatches, plot_matches
from calibrate import calibrate_camera
from undistort import undistort_img

if __name__ == "__main__":

  # calibrate camera with chessboard
  # note: take a few pictures of the OpenCV chessboard with source camera
  calibrate_folder = 'calibrate'
  calibrate_camera(calibrate_folder)

  # open target images
  img1 = cv2.imread('imgs/img_shell01.jpg', 0)
  img2 = cv2.imread('imgs/img_shell02.jpg', 0)

  # undistort images
  img1_undistorted = undistort_img(img1)
  img2_undistorted = undistort_img(img2)
  plot_img(img1_undistorted)
  plot_img(img2_undistorted)

  # shrink
  scale = 0.5
  dim = (int(img1_undistorted.shape[1] * scale), int(img1_undistorted.shape[0] * scale))
  img1_small = cv2.resize(img1_undistorted, dim)
  img2_small = cv2.resize(img2_undistorted, dim)

  # get keypoint and descriptors
  kp1, des1 = getKD(KD.SIFT, img1_small)
  kp2, des2 = getKD(KD.SIFT, img2_small)

  # get matches
  matches, matchesMask = getMatches(kp1, des1, kp2, des2)

  # draw matches
  plot_matches(img1_small,kp1,img2_small,kp2,matches,matchesMask)
