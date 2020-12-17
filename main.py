import cv2
import glob

from camera import calibrate_camera, load_K, save_camera_props
from image import process_img_folder
from helpers import plot_img
from cloud import save_point_cloud, plot_point_cloud, plot_point_cloud_colorless

#! 1. calibrate and store camera parameters
print("-" * 60)
# note: take a few pictures of the OpenCV chessboard with source camera
# this only has to be done once, as the parameters will be saved
# calibrate_camera()
# K = load_K()
# print("K (Intrinsic) Matrix:")
# print(K)

#! 2. calculate and store camera positions from Turntable properties
print("\n" + "-" * 60)
print("Calculating and Storing Camera Properties")

num_cameras = 18  # number of sequential images
degree = 20       # degree of rotation between images
distance = 175    # based on the turntable setup
height = 0        # based on the turntable setup
# save_camera_props(num_cameras, degree, distance, height)

#! 3. process a folder of images
print("\n" + "-" * 60)
print("Processing Images into Point Cloud")

folder = "imgs" # where the source images are stored
loop = True     # True iff images are taken in a loop 
cloud_pts, cloud_rgb = process_img_folder(folder, loop)
# print("FINAL POINT CLOUD")
# print(cloud_pts)
print(cloud_rgb)

#! 4. save and store the results
print("\n" + "-" * 60)
print("Plotting Point Cloud")
# save_point_cloud(cloud_pts, cloud_rgb, "prinplup")
# plot_point_cloud_colorless(cloud_pts)
plot_point_cloud(cloud_pts, cloud_rgb)
