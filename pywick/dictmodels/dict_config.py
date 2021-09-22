import time
from typing import List

from prodict import Prodict


class ExpConfig(Prodict):
    """
    Default configuration class to define some static types (based on configs/train_classifier.yaml)
    """

    auto_balance_dataset: bool  # whether to attempt to fix imbalances in class representation within the dataset (default: False)
    batch_size          : int   # Size of the batch to use when training (per GPU)
    dataroots           : List  # where to find the training data
    exp_id              : str   # id of the experiment (default: generated from datetime)
    gpu_ids             : List  # list of GPUs to use
    input_size          : int   # size of the input image. Networks with atrous convolutions (densenet, fbresnet, inceptionv4) allow flexible image sizes while others do not
                                # see table: https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-a.csv
    mean_std            : List  # mean, std to use for image transforms
    model_spec          : str   # model to use (over 200 models available! see: https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-a.csv)
    num_epochs          : int   # number of epochs to train for (use small number if starting from pretrained NN)
    optimizer           : dict  # optimizer configuration
    output_root         : str   # where to save outputs (e.g. trained NNs)
    random_seed         : int   # random seed to use (default: 1377)
    save_callback       : dict  # callback to use for saving the model (if any)
    scheduler           : dict  # scheduler configuration
    train_val_ratio     : float # ratio of train to val data (if splitting a single dataset)
    use_apex            : bool  # whether to use APEX optimization (only valid if use_gpu = True)
    use_gpu             : bool  # whether to use the GPU for training (default: False)
    val_root            : str   # root dir to use for validation data (if different from dataroots)
    workers             : int   # number of workers to read training data from disk and feed it to the GPU (default: 8)

    keys_to_verify      : List  # Minimum set of keys that must be set to ensure proper configuration

    def init(self):
        self.auto_balance_dataset = False
        self.exp_id = str(int(time.time() * 1000))
        self.mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]      # imagenet default
        self.random_seed = 1337
        self.train_val_ratio = 0.8
        self.use_gpu = False

        self.keys_to_verify = ['batch_size', 'dataroots', 'input_size', 'model_spec', 'num_epochs', 'optimizer', 'output_root', 'scheduler', 'use_gpu', 'workers']

    def verify_properties(self):
        mapped_keys = [i in self.keys() for i in self.keys_to_verify]
        if not all(mapped_keys):
            raise Exception(f'Property verification failed. Not all required properties have been set: {[i for (i, v) in zip(self.keys_to_verify, mapped_keys) if not v]}')