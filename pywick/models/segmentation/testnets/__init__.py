from .deeplabv3 import DeepLabV3 as TEST_DLV3
from .deeplabv2 import DeepLabV2 as TEST_DLV2
from .deeplabv3_xception import DeepLabv3_plus as TEST_DLV3_Xception
from .deeplabv3_xception import create_DLX_V3_pretrained
from .deeplabv3_resnet import create_DLR_V3_pretrained
from .encnet import EncNet as TEST_EncNet, encnet_resnet50 as TEST_EncNet_Res50, encnet_resnet101 as TEST_EncNet_Res101, encnet_resnet152 as TEST_EncNet_Res152
from .psanet import PSANet as TEST_PSANet, get_psanet_resnet50 as TEST_PSANet_res50, get_psanet_resnet101 as TEST_PSANet_res101, get_psanet_resnet152 as TEST_PSANet_res152
from .psp_saeed import PSPNet as TEST_PSPNet2
from .tiramisu_test import FCDenseNet57 as TEST_Tiramisu57
from .Unet_nested import UNet_Nested_dilated as TEST_Unet_nested_dilated
from .unet_plus_plus import NestNet as Unet_Plus_Plus