from .dpn.dualpath import dpn68, dpn68b, dpn98, dpn107, dpn131, DPN # dpnXX = pretrained on imagenet, DPN = not pretrained
from .bninception import bninception, BNInception                   # bninception = pretrained on imagenet, BNInception not pretrained
from .fbresnet import FBResNet, fbresnet152, fbresnet50, fbresnet34, fbresnet18     # only fbresnet152 pretrained
from .inception_resv2_wide import InceptionResV2                    # InceptionResV2 not pretrained
from .inceptionresnetv2 import inceptionresnetv2, InceptionResNetV2 # inceptionresnetv2 = pretrained on imagenet, InceptionResNetV2 not pretrained
from .inceptionv4 import inceptionv4, InceptionV4                   # inceptionv4 = pretrained on imagenet, InceptionV4 not pretrained
from .mobilenetv2 import mobilenetv2, mobilenetv2_01, mobilenetv2_05, mobilenetv2_10, mobilenetv2_025, MobileNetV2 # mobilenetv2* = pretrained on imagenet
from .nasnet import nasnetalarge, NASNetALarge                      # nasnetalarge = pretrained on NASNetALarge, InceptionV4 not pretrained
from .nasnet_mobile import nasnetamobile, NASNetAMobile             # nasnetamobile = pretrained on imagenet, NASNetAMobile not pretrained
from .pnasnet import pnasnet5large, PNASNet5Large                   # pnasnet5large = pretrained on imagenet, PNASNet5Large not pretrained
from .polynet import polynet, PolyNet                               # polynet = pretrained on imagenet, PolyNet not pretrained
from .pyramid_resnet import pyresnet18 as Pyresnet18, pyresnet34 as Pyresnet34, PyResNet        # pyresnetxx = pretrained on imagenet, PyResNet not pretrained
from .resnet_swish import resnet18 as Resnet18_swish, resnet34 as Resnet34_swish, resnet50 as Resnet50_swish, resnet101 as Resnet101_swish, resnet152 as Resnet152_swish # not pretrained
from .resnext import resnext50_32x4d, resnext101_32x4d, resnext101_64x4d, ResNeXt101_32x4d, ResNeXt101_64x4d  # resnextxxx = pretrained on imagenet, ResNeXt not pretrained
from .se_inception import SEInception3                              # not pretrained
from .se_resnet import se_resnet34 as SE_Resnet34, se_resnet50 as se_resnet50_2, se_resnet101 as SE_Resnet101, se_resnet152 as SE_Resnet152   # only se_resnet50_2 is pretrained!
from .senet import se_resnet50, se_resnet101, se_resnet152, senet154, se_resnext50_32x4d, se_resnext101_32x4d, SENet       # pretrained
from .shufflenet_v2 import shufflenetv2, shufflenetv2_x1, shufflenetv2_x05, ShuffleNetV2 # shufflenetv2* = pretrained on imagenet, ShuffleNetV2 not pretrained
from .wide_resnet import WResNet_imagenet                           # not pretrained
from .wide_resnet_2 import WideResNet                               # not pretrained
from .xception import xception, Xception                            # xception = pretrained on imagenet, Xception not pretrained

from .testnets import *
