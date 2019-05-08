from .bisenet import BiSeNet as TEST_BiSeNet, bisenet_resnet18 as TEST_BiSeNet_Res18
from .danet import DANet as TEST_DANet, get_danet_resnet50 as TEST_DANet_Res50, get_danet_resnet101 as TEST_DANet_Res101, get_danet_resnet152 as TEST_DANet_Res152
from .deeplabv3 import DeepLabV3 as TEST_DLV3
from .deeplabv2 import DeepLabV2 as TEST_DLV2
from .deeplabv3_xception import DeepLabv3_plus as TEST_DLV3_Xception
from .deeplabv3_xception import create_DLX_V3_pretrained
from .deeplabv3_resnet import create_DLR_V3_pretrained
from .denseaspp import DenseASPP as TEST_DenseASPP, denseaspp_densenet121 as TEST_DenseASPP_121, denseaspp_densenet161 as TEST_DenseASPP_161, denseaspp_densenet169 as TEST_DenseASPP_169, denseaspp_densenet201 as TEST_DenseASPP_201
from .dilated_linknet import DilatedLinkNet34 as TEST_DiLinknet
from .encnet import EncNet as TEST_EncNet, encnet_resnet50 as TEST_EncNet_Res50, encnet_resnet101 as TEST_EncNet_Res101, encnet_resnet152 as TEST_EncNet_Res152
from .mnas_linknets import LinkCeption as TEST_LinkCeption, LinkNet50 as TEST_Linknet50, LinkNet101 as TEST_Linknet101, LinkNet152 as TEST_Linknet152
from .mnas_linknets import LinkDenseNet121 as TEST_LinkDenseNet121, LinkDenseNet161 as TEST_LinkDenseNet161, LinkNeXt as TEST_LinkNext_Mnas
from .linknext import LinkNext as TEST_Linknext
from .ocnet import OCNet as TEST_OCNet, asp_ocnet_resnet101 as TEST_OCNet_ASP_Res101, base_ocnet_resnet101 as TEST_OCNet_Base_Res101, pyramid_ocnet_resnet101 as TEST_OCNet_Pyr_Res101
from .ocnet import asp_ocnet_resnet152 as TEST_OCNet_ASP_Res152, base_ocnet_resnet152 as TEST_OCNet_Base_Res152, pyramid_ocnet_resnet152 as TEST_OCNet_Pyr_Res152
from .psp_saeed import PSPNet as TEST_PSPNet2
from .tiramisu_test import FCDenseNet57 as TEST_Tiramisu57
from .Unet_nested import UNet_Nested_dilated as TEST_Unet_nested_dilated
from .unet_plus_plus import NestNet as Unet_Plus_Plus