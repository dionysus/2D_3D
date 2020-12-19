'''
Used to calculate the location of the matched point in the 3D space
'''

import numpy as np


def get_focal_mm(focal, sensor_width, image_width):
    '''
    Use the focal length in pixels, the sensor width in millimeters and the
    image width in pixels to derive the focal length in millimeters
    '''
    return focal*sensor_width/image_width


def mm_pixel(focal_mm, focal_pixel):
    '''
    Find the ratio millimeters to pixels
    '''
    return focal_mm/focal_pixel


def principal_coordinates(focal, camera, angle):
    '''
    Find the principal coordinates in real world values
    '''
    # Use -1 * focal since we are moving in the opposite directions
    c1 = camera[0] + (-focal * np.sin(angle))
    c2 = camera[1]
    c3 = camera[2] + (-focal * np.cos(angle))
    point = np.array([c1, c2, c3])
    return point

def point_location(principal, p_plane, image_point, mmp, angle):
    '''
    Find the location of the point on the image plane in 3D space
    '''
    # Get x coordinate on image plane
    x_pix = image_point[0]-p_plane[0]
    # Get y coordinate on image plane
    y_pix = image_point[1] - p_plane[1]
    # In real coordinate system
    y_mm = y_pix * mmp
    x_mm = x_pix * mmp
    # Find angle between image plane and Z-axis
    image_angle = angle + np.radians(90)
    # Use to figure out point in 3D space
    c1 = principal[0] + (x_mm * np.sin(image_angle))
    c2 = y_mm
    c3 = principal[2] + (x_mm * np.cos(image_angle))
    point = np.array([c1, c2, c3])
    return point

