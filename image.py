import cv2
import glob

from undistort import undistort_img
from camera import calibrate_camera, load_K
from keypoints import getKD, KD
from match import getMatches, plot_matches
from projection import get_focal_mm
from epipolar import get_fundamental_matrix
# from cloud import get_point_cloud
from helpers import plot_img

#! TODO
def process_matches(img1_pts, img2_pts, img1_index, img2_index, img1_image):
  pair_pts, pair_rgb = []
  num_matches = img1_pts.shape[0]
  K = load_K()
  focal = K[0][0]
  img1_p = np.array([K[0][2], K[1][2]])
  img2_p = np.array([K[0][2], K[1][2]])
  camera_matrix = load_camera_props()

  for i in range(num_matches):
    # coordinate
    img1_pt = img1_pts[i]
    img2_pt = img2_pts[i]
    w, h = img1_image.shape[1], img1_image.shape[0]
    
    sensor_dim = 4.8, 3.5 # in mm's from the manufacturer's site
    sensor_width = sensor_dim[1]
    focal_mm = get_focal_mm(focal, sensor_width, w)
    mmp = mm_pixel(focal_mm, focal)
    # Grab camera coord and angle
     # (x, y, z, rotation)
    img1_camera_coord = np.array([camera_matrix[img1_index][0],
     camera_matrix[img1_index][1], camera_matrix[img1_index][2]])
    img2_camera_coord = np.array([camera_matrix[img2_index][0],
     camera_matrix[img2_index][1], camera_matrix[img2_index][2]])
    img1_angle = camera_matrix[img1_index][3]
    img2_angle = camera_matrix[img2_index][3]

    img1_principal = principal_coordinates(focal_mm, img1_camera_coord, img1_angle)
    img2_principal = principal_coordinates(focal_mm, img2_camera_coord, img2_angle)
    
    img1_location = point_location(img1_principal, img1_p, img1_pts[i], mmp, img1_angle)

    img2_location = point_location(img2_principal, img2_p, img2_pts[i], mmp, img2_angle)
    
    img1_distance = distance_vector(img1_camera_coord, img1_location)
    img2_distance = distance_vector(img2_camera_coord, img2_location)

    projected = projected_point(img1_distance, img1_location, img2_distance, img2_location)

    pair_pts.append(projected)
    # color
    rgb = img1_image.getpixel(img1_pt)
    pair_rgb.append(rgb)

  return pair_pts, pair_rgb

def process_img_pair(img1, img2, img1_index, img2_index):
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
  
  # break matches into lists of coordinates
  img1_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
  img2_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
  pair_pts, pair_rgb = process_matches(img1_pts, img2_pts, img1_index, img2_index, img1_undistorted) #! -------------------------- TODO


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

    pair_pts, pair_rgb = process_img_pair(img1, img2, i, img2_index)
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
