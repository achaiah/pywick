"""
Below you will find all the latest image segmentation models. To get a list of specific model names that are available programmatically, call the ``pywick.models.model_utils.get_supported_models(...)`` method.
To load one of these models with your own number of classes you have two options:
1. You can always load the model directly from the API. Most models allow you to customize *number of classes* as well as *pretrained* options.
2. You can use the ``pywick.models.model_utils.get_model(...)`` method and pass the name of the model
that you want as a string. Note: Some models allow you to customize additional parameters. You can take a look at the ``pywick.models.model_utils.get_model(...)`` method
or at the definition of the model to see what's possible. ``pywick.models.model_utils.get_model(...)`` takes in a ``**kwargs`` argument that you can populate with whatever parameters you'd like
to pass to the model you are creating.
"""

from .bisenet import *
from .carvana_unet import *
from ..classification import resnext101_64x4d
from .danet import *
from .deeplab_v2_res import *
from .deeplab_v3 import *
from .deeplab_v3_plus import *
from .denseaspp import *
from .drn_seg import *
from .duc_hdc import *
from .dunet import *
from .emanet import EMANet, EmaNet152
from .enet import *
from .fcn8s import *
from .fcn16s import *
from .fcn32s import *
from .frrn1 import *
from .fusionnet import *
from .galdnet import *
from .gcnnets import *
from .lexpsp import *
from .mnas_linknets import *
from .ocnet import *
from .refinenet import *
from .resnet_gcn import *
from .seg_net import *
from .testnets import *
from .tiramisu import *
from .u_net import *
from .unet_dilated import *
from .unet_res import *
from .unet_stack import *
from .upernet import *
