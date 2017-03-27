from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
from  skimage.color import rgb2gray
from skimage.transform import resize

def get_log_dir(log_base, suffix):
    """
    For each run create a direction
    :param log_base:
    :param suffix:
    :return:
    """
    run = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '%s/%s_%s' % (log_base, run, suffix)
    os.mkdir(log_dir)
    return log_dir


def preprocess_cartpole(frame):
    """
    add dimension...
    :param frame:
    :return:
    """
    return frame[np.newaxis,:,np.newaxis]


def preprocess_atari(frame):
    """
    Preprocess the image as suggested on kharpathys blog
    :param state:
    :return:
    """
    frame = frame[35:195,:]
    frame = rgb2gray(frame)*255.
    frame = resize(frame,(84,84))
    return frame[:,:,np.newaxis].astype(np.uint8)

def preprocess_atari_no_crop(frame):
    """
    Preprocess the image as suggested on kharpathys blog
    :param state:
    :return:
    """
    frame = rgb2gray(frame)*255.
    frame = resize(frame,(84,84))
    return frame[:,:,np.newaxis].astype(np.uint8)