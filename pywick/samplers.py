"""
Samplers are used during the training phase and are especially useful when your training data is not uniformly distributed among
all of your classes.
"""

import math
from .utils import th_random_choice
import torch.utils.data
import torchvision

from .datasets import FolderDataset, MultiFolderDataset, PredictFolderDataset, ClonedFolderDataset
from torch.utils.data.sampler import Sampler


class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch

    :param class_vector: (torch tensor):
            a vector of class labels
    :param batch_size: (int):
            size of the batch
    """
    def __init__(self, class_vector, batch_size):
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.

    This allows the number of samples per epoch to be larger than the number
    of samples itself, which can be useful when training on 2D slices taken
    from 3D images, for instance.
    """
    def __init__(self, nb_samples, desired_samples, shuffle=False):
        """Initialize MultiSampler

        :param data_source: (dataset):
            the dataset to sample from
        :param desired_samples: (int):
            number of samples per batch you want.
            Whatever the difference is between an even division will
            be randomly selected from the samples.
            e.g. if len(data_source) = 3 and desired_samples = 4, then
            all 3 samples will be included and the last sample will be
            randomly chosen from the 3 original samples.
        :param shuffle: (bool):
            whether to shuffle the indices or not
        
        Example:
            >>> m = MultiSampler(2, 6)
            >>> x = m.gen_sample_array()
            >>> print(x) # [0,1,0,1,0,1]
        """
        self.data_samples = nb_samples
        self.desired_samples = desired_samples
        self.shuffle = shuffle

    def gen_sample_array(self):
        n_repeats = self.desired_samples / self.data_samples
        cat_list = []
        for i in range(math.floor(n_repeats)):
            cat_list.append(torch.arange(0,self.data_samples))
        # add the left over samples
        left_over = self.desired_samples % self.data_samples
        if left_over > 0:
            cat_list.append(th_random_choice(self.data_samples, left_over))
        self.sample_idx_array = torch.cat(cat_list).long()
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.desired_samples


# Source: https://github.com/ufoym/imbalanced-dataset-sampler (MIT license)
class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    :param indices: (list, optional): a list of indices
    :param num_samples: (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is FolderDataset or dataset_type is MultiFolderDataset or dataset_type is PredictFolderDataset or dataset_type is ClonedFolderDataset:
            return dataset.getdata()[idx][1]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
