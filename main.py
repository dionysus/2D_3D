import cv2

from undistort import undistort_img
from calibrate import calibrate_camera, load_K
from keypoints import getKD, KD
from match import getMatches, plot_matches
from epipolar import get_fundamental_matrix
# from cloud import get_point_cloud
from helpers import plot_img

if __name__ == "__main__":

  print("-" * 60)
  #! 1. calibrate camera with chessboard
  # note: take a few pictures of the OpenCV chessboard with source camera
  # don't need to do this again since they have been saved for the sample images
  # calibrate_camera()
  K = load_K()
  print("K (Intrinsic) Matrix:")
  print(K)
  print()

  # open target images
  img1 = cv2.imread('imgs/000.jpg', 0)
  img2 = cv2.imread('imgs/020.jpg', 0)

  #! 2. undistort images
  img1_undistorted = undistort_img(img1)
  img2_undistorted = undistort_img(img2)
  # plot_img(img1_undistorted)

  #! 3. Get Keypoint and Descriptors
  kp1, des1 = getKD(KD.SIFT, img1_undistorted)
  kp2, des2 = getKD(KD.SIFT, img2_undistorted)

  #! 4. Get Keypoint Matches
  all_matches, good_matches, matches_mask = getMatches(kp1, des1, kp2, des2, False)
  # plot_matches(img1_undistorted,kp1,img2_undistorted,kp2,all_matches,matches_mask)
  print("Good Matches: {}".format(len(good_matches)))

  #! 5. Fundamental Matrix
  F, pts1, pts2 = get_fundamental_matrix(kp1, kp2, good_matches)
  print("Fundamental Matrix:")
  print(F)
  print()

  #! 6. ???
