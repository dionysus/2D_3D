import cv2
import glob
import numpy as np

from undistort import undistort_img
from camera import calibrate_camera, load_K, load_camera_props
from keypoints import getKD, KD
from match import getMatches, plot_matches
from projection import distance_vector, projected_point
from focal import get_focal_mm, mm_pixel, principal_coordinates, point_location
from helpers import plot_img

def process_matches(img1_pts, img2_pts, img1_index, img2_index, img1_image):
  '''
  Process set of matches from a pair of images
  Return a list of points and a list of colors for triangulated point cloud
  '''
  pair_pts = []
  pair_rgb = []
  num_matches = img1_pts.shape[0]
  K = load_K()
  focal = K[0][0]
  img1_p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
  img2_p = np.array([int(K[0][2]//1), int(K[1][2]//1)])
  camera_matrix = load_camera_props()

  for i in range(num_matches):
    # coordinate
    img1_pt = img1_pts[i]
    img2_pt = img2_pts[i]
    w, h = img1_image.shape[1], img1_image.shape[0]

    # camera properties
    sensor_dim = 4.8, 3.5 # in mm's from the manufacturer's site
    sensor_width = sensor_dim[1]
    focal_mm = get_focal_mm(focal, sensor_width, w)
    mmp = mm_pixel(focal_mm, focal)

    img1_distance, img1_location = get_img_dist_loc(camera_matrix, img1_index, focal_mm, img1_p, img1_pts, i, mmp)
    img2_distance, img2_location = get_img_dist_loc(camera_matrix, img2_index, focal_mm, img2_p, img2_pts, i, mmp)

    # calculate projected points
    projected = projected_point(img1_distance, img1_location, img2_distance, img2_location)
    pair_pts.append(projected)

    # retrieve image rgb values
    rgb = img1_image[int(img1_pt[1])][int(img1_pt[0])]
    pair_rgb.append(rgb)

  return pair_pts, pair_rgb

def get_img_dist_loc(camera_matrix, img_index, focal_mm, img_p, img_pts, i, mmp):
  '''
  Returns the image distance and location given camera properties
  '''
  img_camera_coord = np.array([
    camera_matrix[img_index][0],
    camera_matrix[img_index][1],
    camera_matrix[img_index][2]])

  img_angle = np.radians(camera_matrix[img_index][3])
  img_principal = principal_coordinates(focal_mm, img_camera_coord, img_angle)
  img_location = point_location(img_principal, img_p, img_pts[i], mmp, img_angle)
  img_distance = distance_vector(img_camera_coord, img_location)

  return img_distance, img_location

def process_img(img):
  '''
  Processes an image for matching
  Returns Keypoints, Descriptors, and undistorted color image.
  '''
  img_undistorted = undistort_img(img)
  img_undistorted_gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)
  kp, des = getKD(KD.SIFT, img_undistorted_gray)
  return kp, des, img_undistorted

def process_img_pair(img1, img2, img1_index, img2_index):
  '''
  Process a pair of sequential images.
  Return a list of points and a list of colors for triangulated point cloud
  '''

  # get keypoints and descriptors
  kp1, des1, img1_undistorted = process_img(img1)
  kp2, des2, img2_undistorted = process_img(img2)

  # get matches
  all_matches, good_matches, matches_mask = getMatches(kp1, des1, kp2, des2, False)
  # plot_matches(img1_undistorted,kp1,img2_undistorted,kp2,all_matches,matches_mask)
  # print("Good Matches: {}".format(len(good_matches)))

  # break matches into lists of coordinates
  img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
  img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])

  # proceses matches
  pair_pts, pair_rgb = process_matches(img1_pts, img2_pts, img1_index, img2_index, img1_undistorted)

  return pair_pts, pair_rgb

def process_img_folder(folder, loop):
  '''
  Process a sequence of images.
  Return a list of points and a list of colors for a point cloud
  '''

  # gather images
  images = sorted(glob.glob(folder + "/*.jpg"))
  print("Images Discovered")
  print(images)

  # loop allows for full 360 sequence of images - compare last and first
  iters = len(images) if loop else len(images) - 1

  cloud_pts = []
  cloud_rgb = []

  for i in range(iters):
  # for i in range(1): #!--------------------------------- this is for test purposes, compare first two
    # open pair of images
    img1 = cv2.imread(images[i], 1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if i < len(images) - 1:
      img2_index = i+1
    elif loop:
      img2_index = 0
    else:
      break # error here, shouldn't reach
    img2 = cv2.imread(images[img2_index], 1)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print("comparing images: {} and {}".format(images[i], images[img2_index]))

    # process pair of images
    pair_pts, pair_rgb = process_img_pair(img1, img2, i, img2_index)

    cloud_pts.extend(pair_pts)
    cloud_rgb.extend(pair_rgb)

  cloud_pts = np.array(cloud_pts)
  cloud_rgb = np.array(cloud_rgb)

  return cloud_pts, cloud_rgb
