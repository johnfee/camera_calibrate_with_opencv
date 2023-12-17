import os
import numpy as np
import yaml
import cv2

local_path = os.getcwd()
dir_board = os.path.join(local_path, 'boards')


def load_images():
    images_list = np.array(
        [os.path.join(dir_board, f) for f in os.listdir(dir_board) if f.endswith(".jpg") and f.startswith('img')])
    print(images_list)
    return images_list

def load_config():
    dir_config = os.path.join(local_path, 'config\\camera_calibration')
    with open(dir_config) as file:
        cam_calib_dist = yaml.load(file, Loader=yaml.Loader)
        print('load yaml')
        print(cam_calib_dist.keys())
        mtx = cam_calib_dist['mtx']
        dist = cam_calib_dist['dist']
        print(mtx)
        print(dist)


load_config()
