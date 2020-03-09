import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import open3d as o3
import json
import pprint
import glob
import cv2

EPSILON = 1.0e-10 # norm should not be small

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()

    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)

        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0

    pcd.colors = o3.utility.Vector3dVector(colours)

    return pcd

def get_origin_of_a_view(view):
    return view['origin']


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))

    return transform


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global

# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)

def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    lidar['points'] = points_trans[:, 0:3]

    return lidar


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)

    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)

    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                                                np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
            (1. - pixel_opacity) * \
            np.multiply(image[pixel_rows, pixel_cols, :], \
                        colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

with open ('/media/jessica/FE20FA1220F9D21F/Uni/Dissertation/datasets/a2d2/cams_lidars.json', 'r') as f:
    config = json.load(f)

src_view_front_center = config['cameras']['front_center']['view']
vehicle_view = target_view = config['vehicle']['view']

lidar_path = '/media/jessica/FE20FA1220F9D21F/Uni/Dissertation/datasets/audi/camera_lidar_semantic/20180810_142822/lidar/cam_front_center'
result_path = '/media/jessica/FE20FA1220F9D21F/Uni/Dissertation/datasets/a2d2/training/lidar/'

image_path = glob.glob(lidar_path + '/*.npz')
image_path.sort()

for i in range(0,len(image_path)):
    lidar_front_center = np.load(image_path[i])
    filename = image_path[i].split('/')[-1].split('.')[0]
    img = np.zeros([1208, 1920, 3])
    result = map_lidar_points_onto_image(img, lidar_front_center)
    # plt.fig = plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    print(filename)
    cv2.imwrite(result_path + filename + '.png', result)
# plt.imshow(lidar_front_center['points'])
# plt.show()

# lidar_front_center = project_lidar_from_to(lidar_front_center, src_view_front_center, vehicle_view)
# pcd_front_center = create_open3d_pc(lidar_front_center)
# o3.visualization.draw_geometries([pcd_front_center])

img = np.zeros([1208,1920,3])
image = map_lidar_points_onto_image(img, lidar_front_center)
plt.fig = plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()