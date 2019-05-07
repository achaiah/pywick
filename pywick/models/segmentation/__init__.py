"""
Below you will find all the latest image segmentation models.
To load one of these models with your own number of classes you have two options:
1. You can always load the model directly from the API. Most models allow you to customize *number of classes* as well as *pretrained* options.
2. You can use the ``pywick.models.model_utils.get_model(...)`` function and pass the name of the model
that you want as a string. The names can be a bit tricky (we're still working on that!) but for now you can
get a good list by calling ``pywick.models.model_utils.get_supported_models(...)``. If all else fails
you can take a look at the ``pywick.models.model_utils.get_model(...)`` function for an exact list.
"""

from .carvana_unet import *
from .carvana_linknet import LinkNet34
from ..classification import resnext101_64x4d
from .deeplab_v2_res import DeepLabv2_ASPP, DeepLabv2_FOV
from .deeplab_v3 import DeepLabv3
from .deeplab_v3_plus import DeepLabv3_plus
from .drn_seg import DRNSeg
from .duc_hdc import ResNetDUC, ResNetDUCHDC
from .enet import ENet
from .fcn8s import FCN8s
from .fcn16s import FCN16VGG
from .fcn32s import FCN32VGG
from .frrn import frrn
from .fusionnet import FusionNet
from .gcn import GCN
from .gcnnets import *
from .lexpsp import PSPNet
from .refinenet import *
from .resnet_gcn import ResnetGCN
from .seg_net import SegNet
from .testnets import *
from .tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103
from .u_net import UNet
from .unet_dilated import uNetDilated
from .unet_res import UNetRes
from .unet_stack import UNet960, UNet_stack