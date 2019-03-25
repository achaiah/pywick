# Source: https://github.com/rwightman/pytorch-dpn-pretrained (License: Apache 2.0)
# Pretrained: Yes

import math
from .dualpath import *
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.densenet import densenet121, densenet169, densenet161, densenet201
from torchvision.models.inception import inception_v3
import torchvision.transforms as transforms
from PIL import Image


def create_model(model_name, num_classes=1000, pretrained=False, **kwargs):
    if 'test_time_pool' in kwargs:
        test_time_pool = kwargs.pop('test_time_pool')
    else:
        test_time_pool = True
    if 'extra' in kwargs:
        extra = kwargs.pop('extra')
    else:
        extra = True
    if model_name == 'dpn68':
        model = dpn68(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn68b':
        model = dpn68b(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn92':
        model = dpn92(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool, extra=extra)
    elif model_name == 'dpn98':
        model = dpn98(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn131':
        model = dpn131(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'dpn107':
        model = dpn107(num_classes=num_classes, pretrained=pretrained, test_time_pool=test_time_pool)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet101':
        model = resnet101(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'resnet152':
        model = resnet152(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet121':
        model = densenet121(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet161':
        model = densenet161(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet169':
        model = densenet169(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'densenet201':
        model = densenet201(num_classes=num_classes, pretrained=pretrained, **kwargs)
    elif model_name == 'inception_v3':
        model = inception_v3(num_classes=num_classes, pretrained=pretrained, transform_input=False, **kwargs)
    else:
        assert False, "Unknown model architecture (%s)" % model_name
    return model


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


DEFAULT_CROP_PCT = 0.875


def get_transforms_eval(model_name, img_size=224, crop_pct=None):
    crop_pct = crop_pct or DEFAULT_CROP_PCT
    if 'dpn' in model_name:
        if crop_pct is None:
            # Use default 87.5% crop for model's native img_size
            # but use 100% crop for larger than native as it
            # improves test time results across all models.
            if img_size == 224:
                scale_size = int(math.floor(img_size / DEFAULT_CROP_PCT))
            else:
                scale_size = img_size
        else:
            scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[124 / 255, 117 / 255, 104 / 255],
            std=[1 / (.0167 * 255)] * 3)
    elif 'inception' in model_name:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = LeNormalize()
    else:
        scale_size = int(math.floor(img_size / crop_pct))
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Scale(scale_size, Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),normalize])
