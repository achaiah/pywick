import fnmatch
import os
import os.path
import random
import warnings

import numpy as np

try:
    from PIL import Image
except:
    warnings.warn('Cant import PIL.. Cant load PIL images')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path, color_space=''):
    try:
        if color_space.lower() == 'rgb':
            return Image.open(path).convert('RGB')
        if color_space.lower() == 'rgba':
            return Image.open(path).convert('RGBA')
        elif color_space.lower() == 'l':
            return Image.open(path).convert('L')
        elif color_space.lower() == '1' or color_space.lower() == 'binary':
            return Image.open(path).convert('1')
        else:
            return Image.open(path)
    except OSError:
        print("!!!  Could not read path: " + path)
        exit(2)


def pil_loader_rgb(path):
    with open(path, 'rb', 0) as f:
        return Image.open(f).convert('RGB')


def pil_loader_bw(path):
    with open(path, 'rb', 0) as f:
        return Image.open(f).convert('L')


def npy_loader(path, color_space=None):     # color space is unused here
    return np.load(path)


def _process_array_argument(x):
    if not is_tuple_or_list(x):
        x = [x]
    return x


def default_file_reader(x):
    if isinstance(x, str):
        if x.endswith('.npy'):
            x = npy_loader(x)
        else:
            try:
                x = pil_loader(x, color_space='RGB')
            except:
                raise ValueError('File Format is not supported')
    #else:
        #raise ValueError('x should be string, but got %s' % type(x))
    return x

def is_tuple_or_list(x):
    return isinstance(x, (tuple,list))

def _process_transform_argument(tform, num_inputs):
    tform = tform if tform is not None else _pass_through
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else _pass_through for t in tform]
    else:
        tform = [tform] * num_inputs
    return tform

def _process_co_transform_argument(tform, num_inputs, num_targets):
    tform = tform if tform is not None else _multi_arg_pass_through
    if is_tuple_or_list(tform):
        if len(tform) != num_inputs:
            raise Exception('If transform is list, must provide one transform for each input')
        tform = [t if t is not None else _multi_arg_pass_through for t in tform]
    else:
        tform = [tform] * min(num_inputs, num_targets)
    return tform


def _return_first_element_of_list(x):
    return x[0]

def _pass_through(x):
    return x

def _multi_arg_pass_through(*x):
    return x

def _find_classes(dir):
    dir = os.path.expanduser(dir)
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _finds_inputs_and_targets(root, class_mode, class_to_idx=None, input_regex='*',
                              rel_target_root='', target_prefix='', target_postfix='', target_extension='png',
                              splitRatio=1.0, random_seed=None, exclusion_file=None):
    """
    Map a dataset from a root folder. Optionally, split the dataset randomly into two partitions (e.g. train and val)

    :param root: string\n
        root dir to scan
    :param class_mode: string in `{'label', 'image', 'path'}`\n
        whether to return a label, an image or a path (of the input) as target
    :param class_to_idx: list\n
        classes to map to indices
    :param input_regex: string (default: *)\n
        regex to apply to scanned input entries
    :param rel_target_root: string\n
        relative target root to scan (if any)
    :param target_prefix: string\n
        prefix to use (if any) when trying to locate the matching target
    :param target_postfix: string\n
        postfix to use (if any) when trying to locate the matching target
    :param splitRatio: float\n
        if set to 0.0 < splitRatio < 1.0 the function will return two datasets
    :param random_seed: int (default: None)\n
        you can control replicability of the split by explicitly setting the random seed
    :param exclusion_file: string (default: None)\n
        list of files (one per line) to exclude when enumerating all files\n
        The list must contain paths relative to the root parameter\n
        each line may include the filename and additional comma-separated metadata, in which case the first item will be considered the path itself and the rest will be ignored

    :return: partition1 (list of (input, target)), partition2 (list of (input, target))
    """
    if class_mode is not 'image' and class_mode is not 'label' and class_mode is not 'path':
        raise ValueError('class_mode must be one of: {label, image, path}')

    if class_mode == 'image' and rel_target_root == '' and target_prefix == '' and target_postfix == '':
            raise ValueError('must provide either rel_target_root or a value for target prefix/postfix when class_mode is set to: image')

    ## Handle exclusion list, if any
    exclusion_list = set()
    if exclusion_file:
        with open(exclusion_file, 'r') as exclfile:
            for line in exclfile:
                exclusion_list.add(line.split(',')[0])

    trainlist_inputs = []
    trainlist_targets = []
    vallist_inputs = []
    vallist_targets = []
    icount = 0
    root = os.path.expanduser(root)
    for subdir in sorted(os.listdir(root)):
        d = os.path.join(root, subdir)
        if not os.path.isdir(d):
            continue

        for rootz, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if _is_image_file(fname):
                    if fnmatch.fnmatch(fname, input_regex):
                        icount = icount + 1

                        # enforce random split
                        if random.random() < splitRatio:
                            inputs = trainlist_inputs
                            targets = trainlist_targets
                        else:
                            inputs = vallist_inputs
                            targets = vallist_targets

                        if not os.path.join(subdir,fname) in exclusion_list:        # exclude any undesired files
                            path = os.path.join(rootz, fname)
                            inputs.append(path)
                            if class_mode == 'path':
                                targets.append(path)
                            elif class_mode == 'label':
                                target_name = class_to_idx.get(subdir)
                                if target_name is None:
                                    print("WARN WARN: !!! Label " + subdir + " does NOT have a valid mapping to ID!  Ignoring...")
                                    inputs.pop()   # Also remove last entry from inputs
                                else:
                                    targets.append(target_name)
                            elif class_mode == 'image':
                                name_vs_ext = fname.rsplit('.', 1)
                                target_fname = os.path.join(root, rel_target_root, subdir, target_prefix + name_vs_ext[0] + target_postfix + '.' + target_extension)
                                if os.path.exists(target_fname):
                                    targets.append(target_fname)
                                else:
                                    raise ValueError('Could not locate file: ' + target_fname + ' corresponding to input: ' + path)
    if class_mode is None:
        return trainlist_inputs, vallist_inputs
    else:
        assert len(trainlist_inputs) == len(trainlist_targets) and len(vallist_inputs) == len(vallist_targets)
        print("Total processed: %i    Train-list: %i items   Val-list: %i items    Exclusion-list: %i items" % (icount, len(trainlist_inputs), len(vallist_inputs), len(exclusion_list)))
        return list(zip(trainlist_inputs, trainlist_targets)), list(zip(vallist_inputs, vallist_targets))
