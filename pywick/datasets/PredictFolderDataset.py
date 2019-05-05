from .FolderDataset import FolderDataset, identity_x

class PredictFolderDataset(FolderDataset):
    """
    Convenience class for loading out-of-memory data that is more geared toward prediction data loading (where ground truth is not available). \n
    If not transformed in any way (either via one of the loaders or transforms) the inputs and targets will be identical (paths to the discovered files)\n
    Instead, the intended use is that the input path is loaded into some kind of binary representation (usually an image), while the target is either
    left as a path or is post-processed to accommodate some special need.

    Arguments
    ---------
    :param root: (string):
        path to main directory

    :param input_regex: (string `(default is any valid image file)`):
        regular expression to find inputs.
        e.g. if all your inputs have the word 'input',
        you'd enter something like input_regex='*input*'

    :param input_transform: (torch transform):
        transform to apply to each input before returning

    :param input_loader: (callable `(default: identity)`):
        defines how to load input samples from file.
        If a function is provided, it should take in a file path as input and return the loaded sample. Identity simply returns the input.

    :param target_loader: (callable `(default: None)`):
        defines how to load target samples from file (which, in our case, are the same as inputs)
        If a function is provided, it should take in a file path as input and return the loaded sample.

    :param exclusion_file: (string):
        list of files to exclude when enumerating all files.
        The list must be a full path relative to the root parameter
    """
    def __init__(self, root, input_regex='*', input_transform=None, input_loader=identity_x, target_loader=None,  exclusion_file=None):

        super().__init__(root=root, class_mode='path', input_regex=input_regex, target_extension=None, transform=input_transform,
                         default_loader=input_loader, target_loader=target_loader, exclusion_file=exclusion_file, target_index_map=None)

