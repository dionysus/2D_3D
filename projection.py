import numpy as np


def norm(vector):
    '''
    Returns the norm of a vector
    '''
    square = np.square(vector)
    sum_vect = np.sum(square)
    return np.sqrt(sum_vect)


def distance_vector(camera, image_plane):
    '''
    Returns the corresponding unit distance vector for the camera coordinates
    and the the point on the image plane
    '''
    distance = image_plane-camera
    value = norm(distance)
    return distance / value


def cross_product(distance1, distance2):
    '''
    Calculate the cross product of the distance vectors in order to find
    a vector that is perpendicular to both of them
    '''
    c1 = distance1[1]*distance2[2]-(distance2[1]*distance1[2])
    c2 = distance1[0]*distance2[2]-(distance2[0]*distance1[2])
    c3 = distance1[0]*distance2[1]-(distance2[0]*distance1[1])
    normal = np.array([c1, -c2, c3])
    value = norm(normal)
    return normal / value


def projected_point(distance1, point1, distance2, point2):
    '''
    Find the coordinates of the point on the image plane projected
    back onto the 3D space
    '''
    # Find the unit normal vector to both distance vectors
    normal = cross_product(distance1, distance2)
    matrix = np.zeros((3, 3))
    # Create a matrix of the two distance vectors and their normal
    matrix[0] = distance1
    matrix[1] = -distance2
    matrix[2] = normal
    matrix = matrix.T
    # Solve the linear system of equations to find closest points
    parameters = np.linalg.solve(matrix, point2-point1)
    # Find the point on each line that's closest to each other
    first_projected = point1 + parameters[0] * distance1
    second_projected = point2 + parameters[1] * distance2
    # Find their midpoint in order to get our projected point
    mid_x = (first_projected[0] + second_projected[0]) / 2
    mid_y = (first_projected[1] + second_projected[1]) / 2
    mid_z = (first_projected[2] + second_projected[2]) / 2
    projected = np.array([mid_x, mid_y, mid_z])
    return projected
