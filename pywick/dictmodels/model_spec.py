from typing import Dict

from prodict import Prodict


class ModelSpec(Prodict):
    """
    Model specification to instantiate. Most models will have pre-configured and pre-trained variants but this gives you more fine-grained control
    """

    model_name          : int   # Size of the batch to use when training (per GPU)
    model_params        : Dict  # where to find the training data

    def init(self):
        # nothing initialized yet but will be expanded in the future
        pass
