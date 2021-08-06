from datetime import datetime
from prodict import Prodict


class ExpConfig(Prodict):
    """
    Default configuration class to define some static types (based on configs/train_classifier.yml)
    """

    batch_size          : int   # Size of the batch to use when training (per GPU)
    dataroots           : str   # where to find the training data
    exp_id              : str   # id of the experiment
    gpu_ids             : list  # list of GPUs to use
    input_size          : int   # size of the input image. Networks with atrous convolutions (densenet, fbresnet, inceptionv4) allow flexible image sizes while others do not
                                # see table: https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-a.csv
    model_name          : str   # model to use (over 200 models available! see: https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet-a.csv)
    num_epochs          : int   # number of epochs to train for (use small number if starting from pretrained NN)
    optimizer           : dict  # optimizer configuration
    output_root         : str   # where to save outputs (e.g. trained NNs)
    save_callback       : dict  # callback to use for saving the model (if any)
    scheduler           : dict  # scheduler configuration
    use_gpu             : bool  # whether to use the GPU for training
    workers             : int   # number of workers to read training data from disk and feed it to the GPU

    def init(self):
        self.exp_id = str(datetime.now())
