# 2D_3D Reconstruction

**Project Webpage:** [https://dionysus.works/project/2d_3d/](https://dionysus.works/project/2d_3d/)

**Team:** Dionysus Cho + Raag Kashyap

## Process
- preprocess images
  - calibrate camera parameters
  - calculate camera positions
  - open fold of images
  - undistort images
- keypoint detection
- feature matching between images
- triangulation & color
- point cloud generation

## How to Use:
Simply run `main.py`.

The important lines are:

34> `cloud_pts, cloud_rgb = process_img_folder(folder, loop)`

which calls `process_img_folder` on a set of images (the 360 sequence of the
Toy from the Turntable) saved in the `imgs` folder.

39> `save_point_cloud(cloud_pts, cloud_rgb, "prinplup")`

which saves the processed images from line 34 to the `output` folder.

40> `plot_point_cloud(cloud_pts, cloud_rgb)`

which plots the point cloud from line 34 in the browser
