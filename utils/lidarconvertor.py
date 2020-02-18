import numpy as np
import matplotlib.pyplot as plt
from kitti_foundation import Kitti, Kitti_util
import cv2

velo_path = '/home/jessica/Downloads/data_road/testing/velodyne'

velo = Kitti_util(frame=65, velo_path=velo_path)
frame = velo.velo_file

print(frame.shape)

x_range, y_range, z_range, scale = (-20, 20), (-20, 20), (-2, 2), 10
topview_img = velo.velo_2_topview_frame(x_range=x_range, y_range=y_range, z_range=z_range)

# Plot result
plt.subplots(1,1, figsize = (5,5))
plt.imshow(topview_img)
plt.axis('off')
plt.show()


def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    image = image * 0

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

""" save one frame about projecting velodyne points into camera image """
image_type = 'gray'  # 'gray' or 'color' image
mode = '00' if image_type == 'gray' else '02'  # image_00 = 'graye image' , image_02 = 'color image'

image_path = '/home/jessica/Downloads/data_road/testing/image'

v_fov, h_fov = (-24.9, 2.0), (-90, 90)

v2c_filepath = 'calib_velo_to_cam.txt'
c2c_filepath = 'calib_cam_to_cam.txt'
lidar_path = '/home/jessica/Downloads/data_road/testing/lidar_2d/'
for i in range(82):

    res = Kitti_util(frame=i, camera_path=image_path, velo_path=velo_path, \
                    v2c_path=v2c_filepath, c2c_path=c2c_filepath)


    img, pnt, c_ = res.velo_projection_frame(v_fov=v_fov, h_fov=h_fov)
    filename = res.image_path[i].split('/')[-1]
    print(filename)

    result = print_projection_plt(pnt, c_, img)
    cv2.imwrite(lidar_path + filename, result)

    # display result image
    # plt.subplots(1,1, figsize = (13,3) )
    # plt.title("Velodyne points to camera image Result")
    # plt.imshow(result)
    # plt.show()
