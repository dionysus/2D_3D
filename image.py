import cv2
import glob

from undistort import undistort_img
from camera import calibrate_camera, load_K
from keypoints import getKD, KD
from match import getMatches, plot_matches
from projection import *
from epipolar import get_fundamental_matrix
# from cloud import get_point_cloud
from helpers import plot_img

#! TODO
def process_matches(good_matches):
  
  pair_pts, pair_rgb = []

  return pair_pts, pair_rgb

def process_img_pair(img1, img2):
  '''
  Process a pair of sequential images
  Return a list of points and a list of colors for triangulated point cloud
  '''
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
  # print("Good Matches: {}".format(len(good_matches)))

  pair_pts, pair_rgb = process_matches(good_matches, ??????) --------------------------#! TODO


def process_img_folder(folder, loop):
  '''
  Process a sequence of images.
  Return a list of points and a list of colors for a point cloud
  '''
  #! THE FOLLOWING WILL BE REFACTORED INTO OWN FILE
  # gather images
  images = sorted(glob.glob(folder + "/*.jpg"))
  print("Images Discovered")
  print(images)

  # loop allows for full 360 sequence of images - compare last and first
  iters = num_cameras if loop else num_cameras - 1

  cloud_pts = []
  cloud_rgb = []

  # for i in range(iters):
  for i in range(1): #! this is for test purposes, compare first two

    #! 0. open target images
    img1 = cv2.imread(images[i], 0)

    if i < num_cameras - 1:
      img2_index = i+1
    elif loop:
      img2_index = 0
    else:
      break # error here, shouldn't reach
    img2 = cv2.imread(images[img2_index], 0)
    print("comparing images: {} and {}".format(images[i], images[img2_index]))

    pair_pts, pair_rgb = process_img_pair(img1, img2, i, )
    cloud_pts.append(pair_pts)
    cloud_rgb.append(pair_rgb)
  
  return cloud_pts, cloud_rgb

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

  # calculate and store camera locations
  num_cameras = 18
  degree = 20
  distance = 175
  height = 0
  #TODO cameras = camera_positions(num_cameras, degree, distance, height)

  # process a folder of images
  process_img_folder(folder="imgs", loop=True)
