import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from distutils.dir_util import copy_tree

train_path = "input/stage1_train/"
file_paths = glob.glob(train_path + "*/images/*.png")

type_names = {0: "staining", 1: "fluorescent", 2: "brightfield"}

def get_violet_num(img):
    violet_num = 0
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            if img[y, x][0] > img[y, x][1] and img[y, x][2] > img[y, x][1]:
                violet_num += 1

    return violet_num

def get_microscopy_type(img):
    violet_num = get_violet_num(img)
    if violet_num > 0:
        return 0
    mean_int = img.mean()
    # randomly picked threshold for distinquishing fluorescent and brightfield
    if mean_int > 100:
        return 2
    return 1

def save_img_type(file_paths):
    for file_path in file_paths:
        img = cv2.imread(file_path)
        img_type = type_names[get_microscopy_type(img)]
        dir_path = os.path.dirname(file_path)
        dir_path = dir_path.split('image')
        dir_path = dir_path[0]
        if img_type == 'fluorescent':
            path = 'input/clusters/fluorescent/{}'.format(os.path.basename(file_path).split('.')[0])
            if not os.path.exists(os.path.join(path, 'images')):
                os.makedirs(os.path.join(path, 'images'))
                os.makedirs(os.path.join(path, 'masks'))
            copy_tree(dir_path, path)
        elif img_type == 'brightfield':
            path = 'input/clusters/brightfield/{}'.format(os.path.basename(file_path).split('.')[0])
            if not os.path.exists(os.path.join(path, 'images')):
                os.makedirs(os.path.join(path, 'images'))
                os.makedirs(os.path.join(path, 'masks'))
            copy_tree(dir_path, path)
        else:
            path = 'input/clusters/staining/{}'.format(os.path.basename(file_path).split('.')[0])
            if not os.path.exists(os.path.join(path, 'images')):
                os.makedirs(os.path.join(path, 'images'))
                os.makedirs(os.path.join(path, 'masks'))
            copy_tree(dir_path, path)

        #shutil.copy(file_path, os.path.join(clusters_path, img_type))
        #shutil.copy(file_path, clusters_path + "/" + str(img_type))


save_img_type(file_paths)