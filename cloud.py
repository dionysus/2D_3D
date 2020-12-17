'''
Point Cloud

For pointcloud export:
	using PyntCloud library
	will need to install in anaconda environment:
	https://github.com/daavoo/pyntcloud
'''

import numpy as np
import cv2
import pandas as pd
from pyntcloud import PyntCloud
import plotly.graph_objects as go

def save_point_cloud(output_points, output_colors):
	'''
	Output point cloud to .PLY file
	'''
	#Define name for output file
	output_file = 'output/reconstruction.ply'
	d = {
		'x': output_points[:,0], 
		'y': output_points[:,1], 
		'z': output_points[:,2],
		'red': output_colors[:,0], 
		'green': output_colors[:,1], 
		'blue':output_colors[:,2]
		}
	cloud = PyntCloud(pd.DataFrame(data=d))
	cloud.to_file(output_file)

def plot_point_cloud_colorless(output_points):
	'''
	plots the Nx6 point cloud pc in 3D
	assumes (1,0,0), (0,1,0), (0,0,-1) as basis

	adapted from A4Q3 Starter Code
	'''
	fig = go.Figure(data=[go.Scatter3d(
		x = output_points[:, 0],
		y = output_points[:, 1],
		z = -output_points[:, 2],
		mode='markers',
		marker=dict(
			size=2,
			color=[255,0,0],
			opacity=0.8
		)
	)])
	fig.show()

def plot_point_cloud(output_points, output_colors):
	'''
	plots the Nx6 point cloud pc in 3D
	assumes (1,0,0), (0,1,0), (0,0,-1) as basis

	adapted from A4Q3 Starter Code
	'''
	fig = go.Figure(data=[go.Scatter3d(
		x = output_points[:, 0],
		y = output_points[:, 1],
		z = -output_points[:, 2],
		mode='markers',
		marker=dict(
			size=2,
			color=output_colors[..., ::-1],
			opacity=0.8
		)
	)])
	fig.show()
