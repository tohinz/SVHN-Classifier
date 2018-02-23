import numpy as np
import os
import argparse
import scipy.io as sio
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Path to the cropped SVHN files (train_32x32.mat and test_32x32.mat).", type=str)
parser.add_argument("--save_to", help="Path where the normalized images are stored.", type=str, default="")
FLAGS = parser.parse_args()

assert os.path.exists(FLAGS.data+'/train_32x32.mat'), "There exists no file \"train_32x32.mat\" in {}".format(FLAGS.data)
assert os.path.exists(FLAGS.data+'/test_32x32.mat'), "There exists no file \"test_32x32.mat\" in {}".format(FLAGS.data)

if FLAGS.save_to == "":
    FLAGS.save_to = FLAGS.data
else:
    assert os.path.exists(FLAGS.save_to), "The specified directory {} to save the data does not exist".\
        format(FLAGS.save_to)


def load_images(path):
    train_images = sio.loadmat(path+'/train_32x32.mat')
    test_images = sio.loadmat(path+'/test_32x32.mat')

    return train_images, test_images


def normalize_images(images):
    imgs = images["X"]
    imgs = np.transpose(imgs, (3, 0, 1, 2))

    labels = images["y"]
    # replace label "10" with label "0"
    labels[labels == 10] = 0

    # normalize images so pixel values are in range [0,1]
    scalar = 1 / 255.
    imgs = imgs * scalar

    return imgs, labels


def save_data(images, labels, name):
    with h5py.File(name+".hdf5", "w") as f:
        f.create_dataset("X", data=images, shape=images.shape, dtype='float32', compression="gzip")
        f.create_dataset("Y", data=labels, shape=labels.shape, dtype='int32', compression="gzip")


train_images, test_images = load_images(FLAGS.data)

train_images_normalized, train_labels = normalize_images(train_images)
save_data(train_images_normalized, train_labels, "SVHN_train")

test_images_normalized, test_labels = normalize_images(test_images)
save_data(test_images_normalized, test_labels, "SVHN_test")