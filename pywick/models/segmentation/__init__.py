from .carvana_unet import UNet128, UNet256, UNet512, UNet1024
from .carvana_linknet import LinkNet34
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
from .resnet_gcn import ResnetGCN
from .seg_net import SegNet
from .testnets import *
from .tiramisu import FCDenseNet57, FCDenseNet67, FCDenseNet103
from .u_net import UNet
from .unet_dilated import uNetDilated
from .unet_res import UNetRes
from .unet_stack import UNet960, UNet_stack