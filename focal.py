import numpy as np
import cv2


def get_focal_mm(focal, sensor_width, image_width):
    return focal*sensor_width/image_width


def mm_pixel(focal_mm, focal_pixel):
    return focal_mm/focal_pixel


def principal_coordinates(focal, camera, angle):
    c1 = camera[0] + (focal * np.cos(angle))
    c2 = camera[1] + (focal * np.sin(angle))
    point = np.array([c1, c2])
    return point

def point_location(focal, principal, p_plane, image_point, mmp):
    x_pix = image_point[0]-p_plane[0]
    y_pix = image_point[0] - p_plane[0]
    y_mm = y_pix*mmp
    x_mm = x_pix * mmp
    point_dist = np.sqrt((np.square(focal)+np.square(x_mm)))
