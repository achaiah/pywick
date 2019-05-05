import itertools
import os

from PIL import Image
from .FolderDataset import FolderDataset, npy_loader, pil_loader, rgb_image_loader, rgba_image_loader, _find_classes, _finds_inputs_and_targets


class MultiFolderDataset(FolderDataset):
    """
    This class extends the FolderDataset with abilty to supply multiple root directories. The ``rel_target_root`` must exist
    relative to each root directory. For complete description of functionality see ``FolderDataset``

    :param roots: (list):
        list of root directories to traverse\n

    :param class_mode: (string in `{'label', 'image', 'path'}):`
        type of target sample to look for and return\n
        `label` = return class folder as target\n
        `image` = return another image as target (determined by optional target_prefix/postfix)\n
            NOTE: if class_mode == 'image', in addition to input, you must also provide rel_target_root,
            target_prefix or target_postfix (in any combination).
        `path` = determines paths for inputs and targets and applies the respective loaders to the path

    :param class_to_idx: (dict):
        If specified, the given class_to_idx map will be used. Otherwise one will be derived from the directory structure.

    :param input_regex: (string `(default is any valid image file)`):
        regular expression to find input images\n
        e.g. if all your inputs have the word 'input',
        you'd enter something like input_regex='*input*'

    :param rel_target_root: (string `(default is Nothing)`):
        root of directory where to look for target images RELATIVE to the root dir (first arg)

    :param target_prefix: (string `(default is Nothing)`):
        prefix to use (if any) when trying to locate the matching target

    :param target_postfix: (string):
        postfix to use (if any) when trying to locate the matching target

    :param transform: (torch transform):
        transform to apply to input sample individually

    :param target_transform: (torch transform):
        transform to apply to target sample individually

    :param co_transform: (torch transform):
        transform to apply to both the input and the target

    :param apply_co_transform_first: (bool):
        whether to apply the co-transform before or after individual transforms (default: True = before)

    :param default_loader: (string in `{'npy', 'pil'}` or function  `(default: pil)`):
        defines how to load samples from file. Will be applied to both input and target unless a separate target_loader is defined.\n
        if a function is provided, it should take in a file path as input and return the loaded sample.

    :param target_loader: (string in `{'npy', 'pil'}` or function  `(default: pil)`):
        defines how to load target samples from file\n
        if a function is provided, it should take in a file path as input and return the loaded sample.

    :param exclusion_file: (string):
        list of files to exclude when enumerating all files.
        The list must be a full path relative to the root parameter

    :param target_index_map: (dict `(defaults to binary mask: {255:1})):
        a dictionary that maps pixel values in the image to classes to be recognized.\n
        Used in conjunction with 'image' class_mode to produce a label for semantic segmentation
        For semantic segmentation this is required so the default is a binary mask. However, if you want to turn off
        this feature then specify target_index_map=None
    """
def __init__(self,
                 roots,
                 class_mode='label',
                 class_to_idx=None,
                 input_regex='*',
                 rel_target_root='',
                 target_prefix='',
                 target_postfix='',
                 target_extension='png',
                 transform=None,
                 target_transform=None,
                 co_transform=None,
                 apply_co_transform_first=True,
                 default_loader='pil',
                 target_loader=None,
                 exclusion_file=None,
                 target_index_map=None):

        # call the super constructor first, then set our own parameters
        # super().__init__()
        self.num_inputs = 1  # these are hardcoded for the fit module to work
        self.num_targets = 1  # these are hardcoded for the fit module to work

        if default_loader == 'npy':
            default_loader = npy_loader
        elif default_loader == 'pil':
            default_loader = pil_loader
        self.default_loader = default_loader

        # separate loading for targets (e.g. for black/white masks)
        self.target_loader = target_loader

        if class_to_idx:
            self.classes = class_to_idx.keys()
            self.class_to_idx = class_to_idx
        else:
            self.classes, self.class_to_idx = _find_classes(roots)

        data_list = list()
        for root in roots:
            datai, _ = _finds_inputs_and_targets(root, class_mode=class_mode, class_to_idx=self.class_to_idx, input_regex=input_regex,
                                                 rel_target_root=rel_target_root, target_prefix=target_prefix, target_postfix=target_postfix,
                                                 target_extension=target_extension, exclusion_file=exclusion_file)
            data_list.append(datai)

        self.data = list(itertools.chain.from_iterable(data_list))

        if len(self.data) == 0:
            raise (RuntimeError('Found 0 data items in subfolders of: {}'.format(roots)))
        else:
            print('Found %i data items' % len(self.data))

        self.roots = [os.path.expanduser(x) for x in roots]
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.apply_co_transform_first = apply_co_transform_first
        self.target_index_map = target_index_map

        self.class_mode = class_mode
