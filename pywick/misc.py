from enum import Enum, auto
import errno
import random
import os
import numpy as np
import torch

class ExecType(Enum):
    TRAIN = auto()
    VAL = auto()

# from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def trun_n_d(n,d):
    '''
    Truncate float (n) to (d) decimal places
    :param n: float to truncate
    :param d: how many decimal places to truncate to
    :return:
    '''
    return int(n*10**d)/10**d

def is_iterable(x):
    return isinstance(x, (tuple, list))
def is_tuple_or_list(x):
    return isinstance(x, (tuple, list))


def time_left_str(seconds):
    '''
    Produces a human-readable string in hh:mm:ss format
    :param seconds:

    :return:
    '''
    # seconds = 370000.0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        thetime = "Projected time remaining  |  {:d}d:{:d}h:{:02d}m".format(d, h, m)
    elif h > 0:
        thetime = "Projected time remaining:  |  {:d}h:{:02d}m".format(h, m)
    elif m > 0:
        thetime = "Projected time remaining:  |  {:02d}m:{:02d}s".format(m, s)
    else:
        thetime = "Projected time remaining:  |  {:02d}s".format(seconds)
    return thetime

# Source: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
def initialize_random(seed, init_cuda=True):
    '''
    Initializes random seed for all aspects of training: python, numpy, torch, cuda

    :param seed:
    :param init_cuda: whether to init cuda seed
    :return:
    '''
    ## Initialize random seed for repeatability ##
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if init_cuda:
        torch.cuda.manual_seed_all(seed)
    ## END ##


def check_mkdir(dir_name):
    """
    Delegates to mkdir_p and is kept around for legacy support
    :param dir_name:
    :return:
    """
    mkdir_p(dir_name)


def mkdir_p(path):
    """Equivalent of a 'mkdir -p' linux command which creates directories if they don't exist. This also correctly resolves paths with ~ in them."""
    path1 = os.path.expanduser(path)
    try:
        os.makedirs(path1)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path1):
            pass
        else:
            raise


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical