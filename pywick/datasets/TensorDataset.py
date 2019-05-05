from .BaseDataset import BaseDataset
from .data_utils import _process_array_argument, _return_first_element_of_list, _process_transform_argument, _process_co_transform_argument, _pass_through

class TensorDataset(BaseDataset):

    """
    Dataset class for loading in-memory data.

    :param inputs: (numpy array)

    :param targets: (numpy array)

    :param input_transform: (transform):
        transform to apply to input sample individually

    :param target_transform: (transform):
        transform to apply to target sample individually

    :param co_transform: (transform):
        transform to apply to both input and target sample simultaneously

    """
    def __init__(self,
                 inputs,
                 targets=None,
                 input_transform=None,
                 target_transform=None,
                 co_transform=None):
        self.inputs = _process_array_argument(inputs)
        self.num_inputs = len(self.inputs)
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through

        if targets is None:
            self.has_target = False
        else:
            self.targets = _process_array_argument(targets)
            self.num_targets = len(self.targets)
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else _pass_through
            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)
            self.has_target = True

        self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        if self.has_target:
            self.target_transform = _process_transform_argument(target_transform, self.num_targets)
            self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """
        input_sample = [self.input_transform[i](self.inputs[i][index]) for i in range(self.num_inputs)]

        if self.has_target:
            target_sample = [self.target_transform[i](self.targets[i][index]) for i in range(self.num_targets)]
            #for i in range(self.min_inputs_or_targets):
            #    input_sample[i], target_sample[i] = self.co_transform[i](input_sample[i], target_sample[i])

            return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
        else:
            return self.input_return_processor(input_sample)