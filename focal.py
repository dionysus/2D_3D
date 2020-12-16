import numpy as np
import cv2


def get_focal_mm(focal, sensor_width, image_width):
    # Focal is the focal length in pixels
    # sensor_width is the actual width of the sensor in mm
    # image_width is the width of the image in pixels
    return focal*sensor_width/image_width


def mm_pixel(focal_mm, focal_pixel):
    return focal_mm/focal_pixel


def principal_coordinates(focal, camera, angle):
    # Want to find principal coordinates in real world values
    # We know the focal length, camera coordinates and angle with respect
    # to Z-axis
    c1 = camera[0] + (focal * np.cos(angle))
    c2 = camera[1]
    c3 = camera[2] + (focal * np.sin(angle))
    point = np.array([c1, c2, c3])
    return point

def point_location(principal, p_plane, image_point, mmp, angle):
    # Find the location of the point on the image plane in 3D space
    # Get x coordinate on image plane
    x_pix = image_point[0]-p_plane[0]
    # Get y coordinate on image plane
    y_pix = image_point[1] - p_plane[1]
    # In real coordinate system
    y_mm = y_pix*mmp
    x_mm = x_pix * mmp
    # Find angle between image plane and Z-axis
    image_angle = angle + 90
    # Use to figure out point in 3D space
    c1 = principal[0] + (x_mm * np.cos(image_angle))
    c2 = y_mm
    c3 = principal[2] + (x_mm * np.sin(image_angle))
    point = np.array([c1, c2, c3])
    return point

