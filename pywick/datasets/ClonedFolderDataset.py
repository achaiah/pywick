import random
from .FolderDataset import FolderDataset


class ClonedFolderDataset(FolderDataset):
    """
    Dataset that can be initialized with a dictionary of internal parameters (useful when trying to clone a FolderDataset)

    :param data: (list):
        list of data on which the dataset operates

    :param meta_data: (dict):
        parameters that correspond to the target dataset's attributes

    :param kwargs: (args):
        variable set of key-value pairs to set as attributes for the dataset
    """
    def __init__(self, data, meta_data, **kwargs):

        if len(data) == 0:
            raise (RuntimeError('No data provided'))
        else:
            print('Initializing with %i data items' % len(data))

        self.data = data

        # Source: https://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        # generic way of initializing the object
        for key in meta_data:
            setattr(self, key, meta_data[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def random_split_dataset(orig_dataset, splitRatio=0.8, random_seed=None):
    '''
    Randomly split the given dataset into two datasets based on the provided ratio

    :param orig_dataset: (UsefulDataset):
        dataset to split (of type pywick.datasets.UsefulDataset)

    :param splitRatio: (float):
        ratio to use when splitting the data

    :param random_seed: (int):
        random seed for replicability of results

    :return: tuple of split ClonedFolderDatasets
    '''
    random.seed(a=random_seed)

    # not cloning the dictionary at this point... maybe it should be?
    orig_dict = orig_dataset.getmeta_data()
    part1 = []
    part2 = []

    for i, item in enumerate(orig_dataset.getdata()):
        if random.random() < splitRatio:
            part1.append(item)
        else:
            part2.append(item)

    return ClonedFolderDataset(part1, orig_dict), ClonedFolderDataset(part2, orig_dict)