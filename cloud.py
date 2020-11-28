'''
Generate Point Cloud

https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
will need to install in anaconda environment:
https://github.com/daavoo/pyntcloud
'''

import numpy as np
import cv2
import pandas as pd
from pyntcloud import PyntCloud

def get_point_cloud(img, disparity_map):
    '''
    Given a disparity map, output a point cloud
    '''
    print ("\nGenerating the 3D map...")
    h,w = disparity_map.shape[0], disparity_map.shape[1]
    Q_w = -w/2.0
    Q_h = h/2.0
    #Load focal length. 
    # focal_length = np.load('calibrated_params/FocalLength.npy')
    focal_length = 3.99

    #Perspective transformation matrix
    #This transformation matrix is from the openCV documentation, didn't seem to work for me. 
    # Q = np.float32([[1,0,0,Q_w],[0,-1,0,Q_h],[0,0,0,-focal_length],[0,0,1,0]])

    #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
    #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q = np.float32([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
        [0,0,0,1]])

    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    #Get color points
    colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()
    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]
    #Generate point cloud 
    print ("\n Creating the output file... \n")
    save_point_cloud(output_points, output_colors)

def save_point_cloud(output_points, output_colors):
    '''
    Output point cloud to .PLY file
    '''
    #Define name for output file
    output_file = 'output/reconstructed.ply'
    d = {'x': output_points[:,0], 'y': output_points[:,1], 'z': output_points[:,2],
    'red': output_colors[:,0], 'green': output_colors[:,1], 'blue':output_colors[:,2]}
    cloud = PyntCloud(pd.DataFrame(data=d))
    cloud.to_file(output_file)
