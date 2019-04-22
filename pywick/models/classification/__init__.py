# By convention, model names starting with lowercase are pretrained on imagenet while uppercase are not
from .dpn.dualpath import *                                 # dpnXX = pretrained on imagenet, DPN = not pretrained
from .bninception import *                                  # bninception = pretrained on imagenet, BNInception not pretrained
from .fbresnet import *                                     # only fbresnet152 pretrained
from .inception_resv2_wide import InceptionResV2            # InceptionResV2 not pretrained
from .inceptionresnetv2 import *                            # inceptionresnetv2 = pretrained on imagenet, InceptionResNetV2 not pretrained
from .inceptionv4 import *                                  # inceptionv4 = pretrained on imagenet, InceptionV4 not pretrained
from .mobilenetv2 import *                                  # mobilenetv2* = pretrained on imagenet
from .nasnet import *                                       # nasnetalarge = pretrained on NASNetALarge, InceptionV4 not pretrained
from .nasnet_mobile import *                                # nasnetamobile = pretrained on imagenet, NASNetAMobile not pretrained
from .pnasnet import *                                      # pnasnet5large = pretrained on imagenet, PNASNet5Large not pretrained
from .polynet import *                                      # polynet = pretrained on imagenet, PolyNet not pretrained
from .pyramid_resnet import *                               # pyresnetxx = pretrained on imagenet, PyResNet not pretrained
from .resnet_preact import *                                # not pretrained
from .resnet_swish import *                                 # not pretrained
from .resnext import *                                      # resnextxxx = pretrained on imagenet, ResNeXt not pretrained
from .se_inception import SEInception3                      # not pretrained
from .senet import *                                        # SENet not pretrained, all others pretrained
from .shufflenet_v2 import *                                # shufflenetv2* = pretrained on imagenet, ShuffleNetV2 not pretrained
from .wideresnet import *                                   # models have not been vetted
from .xception import *                                     # xception = pretrained on imagenet, Xception not pretrained

from .testnets import *
