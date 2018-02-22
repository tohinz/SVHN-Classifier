import numpy as np
import os
import argparse
import scipy.io as sio
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path to the cropped SVHN files (train_32x32.mat and test_32x32.mat).", type=str)
FLAGS = parser.parse_args()


##normalize SVHN
# img_array = sio.loadmat('/informatik2/wtm/home/hinz/Datasets/SVHN_cropped/train_32x32.mat')['X']
#
# rows = img_array.shape[0]
# cols = img_array.shape[1]
# chans = img_array.shape[2]
# num_imgs = img_array.shape[3]
# scalar = 1 / 255.
# # Note: not the most efficent way but can monitor what is happening
# new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
# for x in range(0, num_imgs):
#     # TODO reuse normalize_img here
#     chans = img_array[:, :, :, x]
#     # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
#     norm_vec = (255-chans)*1.0/255.0
#     # Mean Subtraction
#     # norm_vec -= np.mean(norm_vec, axis=0)
#     new_array[x] = norm_vec
# # print(new_array[0])
# # exit()
# np.save("/informatik2/wtm/home/hinz/Datasets/SVHN_cropped/train_32x32_normalized.npy", new_array)
# exit()

def load_images(path):
    train_images = sio.loadmat(path+'/train_32x32.mat')
    test_images = sio.loadmat(path+'/test_32x32.mat')

    return train_images, test_images


def normalize_images(images):
    rows = images.shape[0]
    cols = images.shape[1]
    chans = images.shape[2]
    num_imgs = images.shape[3]
    scalar = 1 / 255.
    images = images * scalar

    return images
    # Note: not the most efficent way but can monitor what is happening
    # new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    # for x in range(0, num_imgs):
    #     # TODO reuse normalize_img here
    #     chans = img_array[:, :, :, x]
    #     # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
    #     norm_vec = (255 - chans) * 1.0 / 255.0
    #     # Mean Subtraction
    #     # norm_vec -= np.mean(norm_vec, axis=0)
    #     new_array[x] = norm_vec


def save_data():
    pass

train_images, test_images = load_images(FLAGS.data)
images = normalize_images()
store(images)