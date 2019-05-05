import numpy as np
import os

from PIL import Image
from .UsefulDataset import UsefulDataset
from .data_utils import npy_loader, pil_loader, _find_classes, _finds_inputs_and_targets

# convenience loaders one can use (in order not to reinvent the wheel)
rgb_image_loader = lambda path: Image.open(path).convert('RGB')   # a loader for images that require RGB color space
rgba_image_loader = lambda path: Image.open(path).convert('RGBA')   # a loader for images that require RGBA color space
bw_image_loader = lambda path: Image.open(path).convert('L')      # a loader for images that require B/W color space
identity_x = lambda x: x


class FolderDataset(UsefulDataset):
    """
    An incredibly versatile dataset class for loading out-of-memory data.\n
    First, the relevant directory structures are traversed to find all necessary files.\n
    Then provided loader(s) is/(are) invoked on inputs and targets.\n
    Finally provided transforms are applied with optional ability to specify the order of individual and co-transforms.\n

    The rel_target_root parameter is used for image segmentation cases
        Typically the structure will look like the following:\n
        |- root (aka training images)\n
        |  - dir1\n
        |  - dir2\n
        |- masks (aka label images)\n
        |  - dir1\n
        |  - dir2\n

    :param root: (string):
        path to main directory
    :param class_mode: (string in `{'label', 'image', 'path'}`):
        type of target sample to look for and return\n
        `label` = return class folder as target\n
        `image` = return another image as target (determined by optional target_prefix/postfix).
        NOTE: if class_mode == 'image', in addition to input, you must also provide ``rel_target_root``,
        ``target_prefix`` or ``target_postfix`` (in any combination).\n
        `path` = determines paths for inputs and targets and applies the respective loaders to the path
    :param class_to_idx: (dict):
        If specified, the given class_to_idx map will be used. Otherwise one will be derived from the directory structure.
    :param input_regex: (string `(default is any valid image file)`):
        regular expression to find input images.
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
    :param default_loader: (string in `{'npy', 'pil'}` or function `(default: pil)`):
        defines how to load samples from file. Will be applied to both input and target unless a separate target_loader is defined.
        if a function is provided, it should take in a file path as input and return the loaded sample.
    :param target_loader: (string in `{'npy', 'pil'}` or function `(default: pil)`):
        defines how to load target samples from file.
        If a function is provided, it should take in a file path as input and return the loaded sample.
    :param exclusion_file: (string):
        list of files to exclude when enumerating all files.
        The list must be a full path relative to the root parameter
    :param target_index_map: (dict `(defaults to binary mask: {255:1})`):
        a dictionary that maps pixel values in the image to classes to be recognized.\n
        Used in conjunction with 'image' class_mode to produce a label for semantic segmentation
        For semantic segmentation this is required so the default is a binary mask. However, if you want to turn off
        this feature then specify target_index_map=None
    """

    def __init__(self,
                 root,
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
        super().__init__()

        if default_loader == 'npy':
            default_loader = npy_loader
        elif default_loader == 'pil':
            default_loader = pil_loader
        self.default_loader = default_loader

        # separate loading for targets (e.g. for black/white masks)
        self.target_loader = target_loader

        root = os.path.expanduser(root)

        if class_to_idx:
            self.classes = class_to_idx.keys()
            self.class_to_idx = class_to_idx
        else:
            self.classes, self.class_to_idx = _find_classes([root])
        data, _ = _finds_inputs_and_targets(root, class_mode=class_mode, class_to_idx=self.class_to_idx, input_regex=input_regex,
                                            rel_target_root=rel_target_root, target_prefix=target_prefix, target_postfix=target_postfix,
                                            target_extension=target_extension, exclusion_file=exclusion_file)

        if len(data) == 0:
            raise (RuntimeError('Found 0 data items in subfolders of: %s' % root))
        else:
            print('Found %i data items' % len(data))

        self.root = os.path.expanduser(root)
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.apply_co_transform_first = apply_co_transform_first
        self.target_index_map = target_index_map

        self.class_mode = class_mode

    def __getitem__(self, index):
        # get paths
        input_sample, target_sample = self.data[index]

        in_base = input_sample
        out_base = target_sample

        try:
            if self.target_loader is not None:
                target_sample = self.target_loader(target_sample)

            ## DELETEME
            # if len(self.classes) == 1 and self.class_mode == 'image':  # this is a binary segmentation map
            #     target_sample = self.default_loader(target_sample, color_space='L')
            # else:
            #     if self.class_mode == 'image':
            #         target_sample = self.default_loader(target_sample)
            ## END DELETEME

            # load samples into memory
            input_sample = self.default_loader(input_sample)
            if self.class_mode == 'image' and self.target_index_map is not None:   # if we're dealing with image masks, we need to change the underlying pixels
                target_sample = np.array(target_sample)  # convert to np
                for k, v in self.target_index_map.items():
                    target_sample[target_sample == k] = v  # replace pixels with class values
                target_sample = Image.fromarray(target_sample.astype(np.float32))  # convert back to image

            # apply transforms
            if self.apply_co_transform_first and self.co_transform is not None:
                input_sample, target_sample = self.co_transform(input_sample, target_sample)
            if self.transform is not None:
                # input_sample = self.transform(image=input_sample)     # needed for albumentations to work (but currently albumentations dies with multiple workers)
                input_sample = self.transform(input_sample)
            if self.target_transform is not None:
                target_sample = self.target_transform(target_sample)
            if not self.apply_co_transform_first and self.co_transform is not None:
                input_sample, target_sample = self.co_transform(input_sample, target_sample)

            return input_sample, target_sample
        except Exception as e:
            print('########## ERROR ########')
            print(str(e))
            print('=========================')
            print("ERROR: Exception occurred while processing dataset with input {} and output {}".format(str(in_base), str(out_base)))

    def __len__(self):
        return len(self.data)

    def getdata(self):
        return self.data

    def getmeta_data(self):
        meta = {'num_inputs': self.num_inputs,  # these are hardcoded for the fit module to work
                'num_targets': self.num_targets,
                'transform': self.transform,
                'target_transform': self.target_transform,
                'co_transform': self.co_transform,
                'class_to_idx': self.class_to_idx,
                'class_mode': self.class_mode,
                'classes': self.classes,
                'default_loader': self.default_loader,
                'target_loader': self.target_loader,
                'apply_co_transform_first': self.apply_co_transform_first,
                'target_index_map': self.target_index_map
                }
        return meta