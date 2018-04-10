import skimage.io
import numpy as np
import random
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from skimage import transform
import cntk
from PIL import Image
import nltk


def read_image_labels(image_id):
    # most of the content in this function is taken from 'Example Metric Implementation' kernel
    # by 'William Cukierski'
    image_file = "input/stage1_train/{}/images/{}.png".format(image_id,image_id)
    mask_file = "input/stage1_train/{}/masks/*.png".format(image_id)
    image = skimage.io.imread(image_file)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + 1
    return image, labels

def data_aug(image,label,angel):
    image = transform.rotate(image,angel)
    label = transform.rotate(label,angel)
    return image, label


def make_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        image, labels = read_image_labels(image_id)
        if not os.path.exists("input/stage1_train/{}/augs/".format(image_id)):
            os.makedirs("input/stage1_train/{}/augs/".format(image_id))
        if not os.path.exists("input/stage1_train/{}/augs_masks/".format(image_id)):
            os.makedirs("input/stage1_train/{}/augs_masks/".format(image_id))
        if not os.path.exists("input/stage1_train/{}/merge_masks/".format(image_id)):
            os.makedirs("input/stage1_train/{}/merge_masks/".format(image_id))
        # also save the original image in augmented file
        plt.imsave(fname="input/stage1_train/{}/augs/{}.png".format(image_id, image_id), arr=image)
        plt.imsave(fname="input/stage1_train/{}/augs_masks/{}.png".format(image_id, image_id), arr=labels)
        plt.imsave(fname="input/stage1_train/{}/merge_masks/{}.png".format(image_id, image_id), arr=labels)

        for i in range(3):
            new_image, new_labels = data_aug(image, labels, 90 * (i + 1))
            aug_img_dir = "input/stage1_train/{}/augs/{}_{}.png".format(image_id, image_id, i)
            aug_mask_dir = "input/stage1_train/{}/augs_masks/{}_{}.png".format(image_id, image_id, i)
            plt.imsave(fname=aug_img_dir, arr=new_image)
            plt.imsave(fname=aug_mask_dir, arr=new_labels)

        image_flip, labels_flip = np.fliplr(image), np.fliplr(labels)

        for i in range(4):
            new_image, new_labels = data_aug(image_flip, labels_flip, 90 * (i + 1))
            aug_img_dir = "input/stage1_train/{}/augs/{}_{}.png".format(image_id, image_id, i + 3)
            aug_mask_dir = "input/stage1_train/{}/augs_masks/{}_{}.png".format(image_id, image_id, i + 3)
            plt.imsave(fname=aug_img_dir, arr=new_image)
            plt.imsave(fname=aug_mask_dir, arr=new_labels)


def clean_data_augmentation(image_ids):
    for ax_index, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        if os.path.exists("input/stage1_train/{}/augs/".format(image_id)):
            shutil.rmtree("input/stage1_train/{}/augs/".format(image_id))
        if os.path.exists("input/stage1_train/{}/augs_masks/".format(image_id)):
            shutil.rmtree("input/stage1_train/{}/augs_masks/".format(image_id))
        if os.path.exists("input/stage1_train/{}/merge_masks/".format(image_id)):
            shutil.rmtree("input/stage1_train/{}/merge_masks/".format(image_id))

def get_width_height(image_ids):
    temp_hw = []
    for image_id in image_ids:
        image_file = "input/stage1_train/{}/images/{}.png".format(image_id, image_id)
        image = skimage.io.imread(image_file)
        height, width, _ = image.shape
        temp_hw.append(str(height) + ',' + str(width))
    fd = nltk.FreqDist(temp_hw)
    return fd

