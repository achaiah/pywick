import numpy as np
import torch as th
from torchvision import transforms
from .data_utils import is_tuple_or_list


class BaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __len__(self):
        return len(self.inputs) if not isinstance(self.inputs, (tuple,list)) else len(self.inputs[0])

    def add_input_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_inputs))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.input_transform[i] = transforms.Compose([transform, self.input_transform[i]])
        else:
            for i in idx:
                self.input_transform[i] = transforms.Compose([self.input_transform[i], transform])

    def add_target_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.num_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.target_transform[i] = transforms.Compose([transform, self.target_transform[i]])
        else:
            for i in idx:
                self.target_transform[i] = transforms.Compose([self.target_transform[i], transform])

    def add_co_transform(self, transform, add_to_front=True, idx=None):
        if idx is None:
            idx = np.arange(len(self.min_inputs_or_targets))
        elif not is_tuple_or_list(idx):
            idx = [idx]

        if add_to_front:
            for i in idx:
                self.co_transform[i] = transforms.Compose([transform, self.co_transform[i]])
        else:
            for i in idx:
                self.co_transform[i] = transforms.Compose([self.co_transform[i], transform])

    def load(self, num_samples=None, load_range=None):
        """
        Load all data or a subset of the data into actual memory.
        For instance, if the inputs are paths to image files, then this
        function will actually load those images.

        :param num_samples: (int (optional)):
            number of samples to load. if None, will load all
        :param load_range: (numpy array of integers (optional)):
            the index range of images to load
            e.g. np.arange(4) loads the first 4 inputs+targets
        """
        def _parse_shape(x):
            if isinstance(x, (list,tuple)):
                return (len(x),)
            elif isinstance(x, th.Tensor):
                return x.size()
            else:
                return (1,)

        if num_samples is None and load_range is None:
            num_samples = len(self)
            load_range = np.arange(num_samples)
        elif num_samples is None and load_range is not None:
            num_samples = len(load_range)
        elif num_samples is not None and load_range is None:
            load_range = np.arange(num_samples)


        if self.has_target:
            for enum_idx, sample_idx in enumerate(load_range):
                input_sample, target_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        _shape = [len(load_range)] + list(_parse_shape(input_sample))
                        inputs = np.empty(_shape)
                    else:
                        inputs = []
                        for i in range(self.num_inputs):
                            _shape = [len(load_range)] + list(_parse_shape(input_sample[i]))
                            inputs.append(np.empty(_shape))
                        #inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))) for i in range(self.num_inputs)]

                    if self.num_targets == 1:
                        _shape = [len(load_range)] + list(_parse_shape(target_sample))
                        targets = np.empty(_shape)
                        #targets = np.empty((len(load_range), *_parse_shape(target_sample)))
                    else:
                        targets = []
                        for i in range(self.num_targets):
                            _shape = [len(load_range)] + list(_parse_shape(target_sample[i]))
                            targets.append(np.empty(_shape))
                        #targets = [np.empty((len(load_range), *_parse_shape(target_sample[i]))) for i in range(self.num_targets)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

                if self.num_targets == 1:
                    targets[enum_idx] = target_sample
                else:
                    for i in range(self.num_targets):
                        targets[i][enum_idx] = target_sample[i]

            return inputs, targets
        else:
            for enum_idx, sample_idx in enumerate(load_range):
                input_sample = self.__getitem__(sample_idx)

                if enum_idx == 0:
                    if self.num_inputs == 1:
                        _shape = [len(load_range)] + list(_parse_shape(input_sample))
                        inputs = np.empty(_shape)
                        #inputs = np.empty((len(load_range), *_parse_shape(input_sample)))
                    else:
                        inputs = []
                        for i in range(self.num_inputs):
                            _shape = [len(load_range)] + list(_parse_shape(input_sample[i]))
                            inputs.append(np.empty(_shape))
                        #inputs = [np.empty((len(load_range), *_parse_shape(input_sample[i]))) for i in range(self.num_inputs)]

                if self.num_inputs == 1:
                    inputs[enum_idx] = input_sample
                else:
                    for i in range(self.num_inputs):
                        inputs[i][enum_idx] = input_sample[i]

            return inputs

    def fit_transforms(self):
        """
        Make a single pass through the entire dataset in order to fit
        any parameters of the transforms which require the entire dataset.
        e.g. StandardScaler() requires mean and std for the entire dataset.

        If you dont call this fit function, then transforms which require properties
        of the entire dataset will just work at the batch level.
        e.g. StandardScaler() will normalize each batch by the specific batch mean/std
        """
        it_fit = hasattr(self.input_transform, 'update_fit')
        tt_fit = hasattr(self.target_transform, 'update_fit')
        ct_fit = hasattr(self.co_transform, 'update_fit')
        if it_fit or tt_fit or ct_fit:
            for sample_idx in range(len(self)):
                if hasattr(self, 'input_loader'):
                    x = self.input_loader(self.inputs[sample_idx])
                else:
                    x = self.inputs[sample_idx]
                if it_fit:
                    self.input_transform.update_fit(x)
                if self.has_target:
                    if hasattr(self, 'target_loader'):
                        y = self.target_loader(self.targets[sample_idx])
                    else:
                        y = self.targets[sample_idx]
                if tt_fit:
                    self.target_transform.update_fit(y)
                if ct_fit:
                    self.co_transform.update_fit(x,y)