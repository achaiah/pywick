from .deeplabv3 import DeepLabV3 as TEST_DLV3
from .deeplabv2 import DeepLabV2 as TEST_DLV2
from .deeplabv3_xception import DeepLabv3_plus as TEST_DLV3_Xception
from .deeplabv3_xception import create_DLX_V3_pretrained
from .deeplabv3_resnet import create_DLR_V3_pretrained
from .dilated_linknet import DilatedLinkNet34 as TEST_DiLinknet
from .mnas_linknets import LinkCeption as TEST_LinkCeption
from .mnas_linknets import LinkNet50 as TEST_Linknet50
from .mnas_linknets import LinkNet101 as TEST_Linknet101
from .mnas_linknets import LinkNet152 as TEST_Linknet152
from .mnas_linknets import LinkDenseNet121 as TEST_LinkDenseNet121
from .linknext import LinkNext as TEST_Linknext
from .standard_fc_densenets import FCDenseNet103 as TEST_FCDensenet
from .psp_saeed import PSPNet as TEST_PSPNet2
from .tiramisu_test import FCDenseNet57 as TEST_Tiramisu57
from .Unet_nested import UNet_Nested_dilated as TEST_Unet_nested_dilated
from .unet_plus_plus import NestNet as Unet_Plus_Plus
