from functools import partial
from typing import Callable

from torchvision.models.resnet import Bottleneck

from . import classification
from .segmentation import *
from . import segmentation
from enum import Enum
from torchvision import models as torch_models
from torchvision.models.inception import InceptionAux
import torch
import torch.nn as nn
import os
import errno

rwightman_repo = 'rwightman/pytorch-image-models'


class ModelType(Enum):
    """
    Enum to use for looking up task-specific attributes
    """
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'


def get_fc_names(model_name, model_type=ModelType.CLASSIFICATION):
    """
    Look up the name of the FC (fully connected) layer(s) of a model. Typically these are the layers that are replaced when transfer-learning from another model.
    Note that only a handful of models have more than one FC layer. Currently only 'classification' models are supported.

    :param model_name: (string)
        name of the model
    :param model_type: (ModelType)
        only classification is supported at this time

    :return: list
        names of the FC layers (usually a single one)
    """

    if model_type == ModelType.CLASSIFICATION:
        fc_names = ['last_linear']  # most common name of the last layer (to be replaced)

        if model_name in torch_models.__dict__:
            if any(x in ['densenet', 'squeezenet', 'vgg', 'efficientnet'] for x in model_name):    # apparently these are different...
                fc_names = ['classifier']
            elif any(x in ['inception_v3', 'inceptionv3', 'Inception3'] for x in model_name):
                fc_names = ['AuxLogits.fc', 'fc']
            elif any(x in ['swin_', 'vit_', 'pit_'] for x in model_name):
                fc_names = ['head', 'head_dist']
            elif any(x in ['nfnet', 'gernet'] for x in model_name):
                fc_names = ['head.fc']
            else:
                fc_names = ['fc']  # the name of the last layer to be replaced in torchvision models
        ## NOTE NOTE NOTE
        # 'squeezenet' pretrained model weights are saved as ['classifier.1']
        # 'vgg' pretrained model weights are saved as ['classifier.0', 'classifier.3', 'classifier.6']

        return fc_names

    else:
        return [None]


def get_model(model_type: ModelType,
              model_name: str,
              num_classes: int,
              pretrained: bool = True,
              force_reload: bool = False,
              custom_load_fn: Callable = None,
              **kwargs):
    """
    :param model_type: (ModelType):
        type of model we're trying to obtain (classification or segmentation)
    :param model_name: (string):
        name of the model. By convention (for classification models) lowercase names represent pretrained model variants while Uppercase do not.
    :param num_classes: (int):
        number of classes to initialize with (this will replace the last classification layer or set the number of segmented classes)
    :param pretrained: (bool):
        whether to load the default pretrained version of the model
        NOTE! NOTE! For classification, the lowercase model names are the pretrained variants while the Uppercase model names are not.
        The only exception applies to torch.hub models (all efficientnet, mixnet, mobilenetv3, mnasnet, spnasnet variants) where a single
        lower-case string can be used for vanilla and pretrained versions. Otherwise, it is IN ERROR to specify an Uppercase model name variant
        with pretrained=True but one can specify a lowercase model variant with pretrained=False
        (default: True)
    :param force_reload: (bool):
        Whether to force reloading the list of models from torch.hub. By default, a cache file is used if it is found locally and that can prevent
        new or updated models from being found.
    :param custom_load_fn: (Callable):
        A custom callable function to use for loading models (typically used to load cutting-edge or custom models that are not in the publicly available list)

    :return: model
    """

    if model_name not in get_supported_models(model_type) and not model_name.startswith('TEST') and custom_load_fn is None:
        raise ValueError(f'The supplied model name: {model_name} was not found in the list of acceptable model names.'
                         ' Use get_supported_models() to obtain a list of supported models or supply a custom_load_fn')

    print("INFO: Loading Model:   --   " + model_name + "  with number of classes: " + str(num_classes))
    
    if model_type == ModelType.CLASSIFICATION:
        torch_hub_names = torch.hub.list(rwightman_repo, force_reload=force_reload)
        if model_name in torch_hub_names:
            model = torch.hub.load(rwightman_repo, model_name, pretrained=pretrained, num_classes=num_classes)
        elif custom_load_fn is not None:
            model = custom_load_fn(model_name, pretrained, num_classes, **kwargs)
        else:
            # 1. Load model (pretrained or vanilla)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            fc_name = get_fc_names(model_name=model_name, model_type=model_type)[-1:][0]    # we're only interested in the last layer name
            new_fc = None            # Custom layer to replace with (if none, then it will be handled generically)
            if model_name in torch_models.__dict__:
                print('INFO: Loading torchvision model: {}\t Pretrained: {}'.format(model_name, pretrained))
                model = torch_models.__dict__[model_name](pretrained=pretrained)  # find a model included in the torchvision package
            else:
                net_list = ['fbresnet', 'inception', 'mobilenet', 'nasnet', 'polynet', 'resnext', 'se_resnet', 'senet', 'shufflenet', 'xception']
                if pretrained:
                    print('INFO: Loading a pretrained model: {}'.format(model_name))
                    if 'dpn' in model_name:
                        model = classification.__dict__[model_name](pretrained=True)  # find a model included in the pywick classification package
                    elif any(net_name in model_name for net_name in net_list):
                        model = classification.__dict__[model_name](pretrained='imagenet')
                else:
                    print('INFO: Loading a vanilla model: {}'.format(model_name))
                    model = classification.__dict__[model_name](pretrained=None)  # pretrained must be set to None for the extra models... go figure

            # 2. Create custom FC layers for non-standardized models
            if 'squeezenet' in model_name:
                final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
                new_fc = nn.Sequential(
                    nn.Dropout(p=0.5),
                    final_conv,
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(13, stride=1)
                )
                model.num_classes = num_classes
            elif 'vgg' in model_name:
                new_fc = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes)
                )
            elif 'inception3' in model_name.lower() or 'inception_v3' in model_name.lower():
                # Replace the extra aux_logits FC layer if aux_logits are enabled
                if getattr(model, 'aux_logits', False):
                    model.AuxLogits = InceptionAux(768, num_classes)
            elif 'dpn' in model_name.lower():
                old_fc = getattr(model, fc_name)
                new_fc = nn.Conv2d(old_fc.in_channels, num_classes, kernel_size=1, bias=True)

            # 3. For standard FC layers (nn.Linear) perform a reflection lookup and generate a new FC
            if new_fc is None:
                old_fc = getattr(model, fc_name)
                new_fc = nn.Linear(old_fc.in_features, num_classes)

            # 4. perform replacement of the last FC / Linear layer with a new one
            setattr(model, fc_name, new_fc)

        return model

    elif model_type == ModelType.SEGMENTATION:
        """
        Additional Segmentation Option Parameters
        -----------------------------------------
        
        BiSeNet
            - :param backbone: (str, default: 'resnet18') The type of backbone to use (one of `{'resnet18'}`)
            - :param aux: (bool, default: False) Whether to output auxiliary loss (typically an FC loss to help with multi-class segmentation)
            
        DANet_ResnetXXX, DUNet_ResnetXXX, EncNet, OCNet_XXX_XXX, PSANet_XXX
            - :param aux: (bool, default: False) Whether to output auxiliary loss (typically an FC loss to help with multi-class segmentation)
            - :param backbone: (str, default: 'resnet101') The type of backbone to use (one of `{'resnet50', 'resnet101', 'resnet152'}`)
            - :param norm_layer (Pytorch nn.Module, default: nn.BatchNorm2d) The normalization layer to use. Typically it is not necessary to change this parameter unless you know what you're doing.
        
        DenseASPP_XXX
            - :param aux: (bool, default: False) Whether to output auxiliary loss (typically an FC loss to help with multi-class segmentation)
            - :param backbone: (str, default: 'densenet161') The type of backbone to use (one of `{'densenet121', 'densenet161', 'densenet169', 'densenet201'}`)
            - :param dilate_scale (int, default: 8) The size of the dilation to use (one of `{8, 16}`)
            - :param norm_layer (Pytorch nn.Module, default: nn.BatchNorm2d) The normalization layer to use. Typically it is not necessary to change this parameter unless you know what you're doing.
            
        DRNSeg
            - :param model_name: (str - required) The type of backbone to use. One of `{'DRN_C_42', 'DRN_C_58', 'DRN_D_38', 'DRN_D_54', 'DRN_D_105'}`
            
        EncNet_ResnetXXX
            - :param aux: (bool, default: False) Whether to output auxiliary loss (typically an FC loss to help with multi-class segmentation)
            - :param backbone: (str, default: 'resnet101') The type of backbone to use (one of `{'resnet50', 'resnet101', 'resnet152'}`)
            - :param norm_layer (Pytorch nn.Module, default: nn.BatchNorm2d) The normalization layer to use. Typically it is not necessary to change this parameter unless you know what you're doing.
            - :param se_loss (bool, default: True) Whether to compute se_loss
            - :param lateral (bool, default: False)
        
        frrn
            - :param model_type: (str - required) The type of model to use. One of `{'A', 'B'}`
        
        GCN, GCN_DENSENET, GCN_NASNET, GCN_PSP, GCN_RESNEXT
            - :param k: (int - optional) The size of global kernel
        
        GCN_PSP, GCN_RESNEXT, Unet_stack
            - :param input_size: (int - required) The size of output image (will be square)
        
        LinkCeption, 'LinkDenseNet121', 'LinkDenseNet161', 'LinkInceptionResNet', 'LinkNet18', 'LinkNet34', 'LinkNet50', 'LinkNet101', 'LinkNet152', 'LinkNeXt', 'CoarseLinkNet50'
            - :param num_channels: (int, default: 3) Number of channels in the image (e.g. 3 = RGB)
            - :param is_deconv: (bool, default: False)
            - :param decoder_kernel_size: (int, default: 3) Size of the decoder kernel
        
        PSPNet
            - :param backend: (str, default: densenet121) The type of extractor to use. One of `{'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121'}`
        
        RefineNet4Cascade, RefineNet4CascadePoolingImproved
            - :param input_shape: (tuple(int, int), default: (1, 512) - required!) Tuple representing input shape (num_channels, dim)
            - :param freeze_resnet: (bool, default: False) - whether to freeze the underlying resnet
        """

        model_exists = False
        for m_name in get_supported_models(model_type):
            if model_name in m_name:
                model_exists = True
                break
        if model_exists:
            # Print warnings and helpful messages for nets that require additional configuration
            if model_name in ['GCN_PSP', 'GCN_RESNEXT', 'RefineNet4Cascade', 'RefineNet4CascadePoolingImproved', 'Unet_stack']:
                print('WARN: Did you remember to set the input_size parameter: (int) ?')
            elif model_name in ['RefineNet4Cascade', 'RefineNet4CascadePoolingImproved']:
                print('WARN: Did you remember to set the input_shape parameter: tuple(int, int)?')

            # logic to switch between different constructors
            if model_name in ['FusionNet', 'Enet', 'frrn', 'Tiramisu57', 'Tiramisu67', 'Tiramisu101'] or model_name.startswith('UNet') and pretrained:  # FusionNet
                    print("WARN: FusionNet, Enet, FRRN, Tiramisu, UNetXXX do not have a pretrained model! Empty model as been created instead.")

            net = segmentation.__dict__[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)

        else:
            raise Exception('Combination of type: {} and model_name: {} is not valid'.format(model_type, model_name))

    return net


def get_supported_models(type: ModelType):
    '''

    :param type: (ModelType):
        classification or segmentation
    :return: list (strings) of supported models
    '''

    import pkgutil
    if type == ModelType.SEGMENTATION:
        excludes = []  # <-- exclude non-model names
        for importer, modname, ispkg in pkgutil.walk_packages(path=segmentation.__path__, prefix=segmentation.__name__+".", onerror=lambda x: None):
            excludes.append(modname.split('.')[-1])
        return [x for x in segmentation.__dict__.keys() if ('__' not in x and x not in excludes)]  # filter out hidden object attributes and module names
    elif type == ModelType.CLASSIFICATION:
        pywick_excludes = []
        for importer, modname, ispkg in pkgutil.walk_packages(path=classification.__path__, prefix=classification.__name__+".", onerror=lambda x: None):
            pywick_excludes.append(modname.split('.')[-1])
        pywick_names = [x for x in classification.__dict__.keys() if '__' not in x and x not in pywick_excludes]     # includes directory and filenames

        pt_excludes = []
        for importer, modname, ispkg in pkgutil.walk_packages(path=torch_models.__path__, prefix=torch_models.__name__+".", onerror=lambda x: None):
            pt_excludes.append(modname.split('.')[-1])
        pt_names = [x for x in torch_models.__dict__ if '__' not in x and x not in pt_excludes]  # includes directory and filenames

        torch_hub_names = torch.hub.list(rwightman_repo, force_reload=True)

        return pywick_names + pt_names + torch_hub_names
    else:
        return None


def _get_untrained_model(model_name, num_classes):
    """
    Primarily, this method exists to return an untrained / vanilla version of a specified (pretrained) model.
    This is on best-attempt basis only and may be out of sync with actual model definitions. The code is manually maintained.

    :param model_name: Lower-case model names are pretrained by convention.
    :param num_classes: Number of classes to initialize the vanilla model with.

    :return: default model for the model_name with custom number of classes
    """

    if model_name.startswith('bninception'):
        return classification.BNInception(num_classes=num_classes)
    elif model_name.startswith('densenet'):
        return torch_models.DenseNet(num_classes=num_classes)
    elif model_name.startswith('dpn'):
        return classification.DPN(num_classes=num_classes)
    elif model_name.startswith('inceptionresnetv2'):
        return classification.InceptionResNetV2(num_classes=num_classes)
    elif model_name.startswith('inception_v3'):
        return torch_models.Inception3(num_classes=num_classes)
    elif model_name.startswith('inceptionv4'):
        return classification.InceptionV4(num_classes=num_classes)
    elif model_name.startswith('nasnetalarge'):
        return classification.NASNetALarge(num_classes=num_classes)
    elif model_name.startswith('nasnetamobile'):
        return classification.NASNetAMobile(num_classes=num_classes)
    elif model_name.startswith('pnasnet5large'):
        return classification.PNASNet5Large(num_classes=num_classes)
    elif model_name.startswith('polynet'):
        return classification.PolyNet(num_classes=num_classes)
    elif model_name.startswith('pyresnet'):
        return classification.PyResNet(num_classes=num_classes)
    elif model_name.startswith('resnet'):
        return torch_models.ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    elif model_name.startswith('resnext101_32x4d'):
        return classification.ResNeXt101_32x4d(num_classes=num_classes)
    elif model_name.startswith('resnext101_64x4d'):
        return classification.ResNeXt101_64x4d(num_classes=num_classes)
    elif model_name.startswith('se_inception'):
        return classification.SEInception3(num_classes=num_classes)
    elif model_name.startswith('se_resnext50_32x4d'):
        return classification.se_resnext50_32x4d(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('se_resnext101_32x4d'):
        return classification.se_resnext101_32x4d(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('senet154'):
        return classification.senet154(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('se_resnet50'):
        return classification.se_resnet50(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('se_resnet101'):
        return classification.se_resnet101(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('se_resnet152'):
        return classification.se_resnet152(num_classes=num_classes, pretrained=None)
    elif model_name.startswith('squeezenet1_0'):
        return torch_models.squeezenet1_0(num_classes=num_classes, pretrained=False)
    elif model_name.startswith('squeezenet1_1'):
        return torch_models.squeezenet1_1(num_classes=num_classes, pretrained=False)
    elif model_name.startswith('xception'):
        return classification.Xception(num_classes=num_classes)
    else:
        raise ValueError('No vanilla model found for model name: {}'.format(model_name))


# We solve the dimensionality mismatch between final layers in the constructed vs pretrained modules at the data level.
def diff_states(dict_canonical, dict_subset):
    """
    **DEPRECATED - DO NOT USE**
    """
    names1, names2 = (list(dict_canonical.keys()), list(dict_subset.keys()))

    # Sanity check that param names overlap
    # Note that params are not necessarily in the same order
    # for every pretrained model
    not_in_1 = [n for n in names1 if n not in names2]
    not_in_2 = [n for n in names2 if n not in names1]
    if len(not_in_1) != 0:
        raise AssertionError
    if len(not_in_2) != 0:
        raise AssertionError

    for name, v1 in dict_canonical.items():
        v2 = dict_subset[name]
        if not hasattr(v2, 'size'):
            raise AssertionError
        if v1.size() != v2.size():
            yield (name, v1)


def load_checkpoint(checkpoint_path, model=None, device='cpu', strict=True, ignore_chkpt_layers=None):
    """
    Loads weights from a checkpoint into memory. If model is not None then the weights are loaded into the model.

    :param checkpoint_path: (string):
        path to a pretrained network to load weights from
    :param model: the model object to load weights onto (default: None)
    :param device: (string):
        which device to load model onto (default:'cpu')
    :param strict: (bool):
        whether to ensure strict key matching (True) or to ignore non-matching keys. (default: True)
    :param ignore_chkpt_layers: one of {string, list) -- CURRENTLY UNIMPLEMENTED:
        whether to ignore some subset of layers from checkpoint. This is usually done when loading
        checkpoint data into a model with a different number of final classes. In that case, you can pass in a
        special string: 'last_layer' which will trigger the logic to chop off the last layer of the checkpoint dictionary. Otherwise
        you can pass in a list of layers to remove from the checkpoint before loading it (e.g. you would do that when
        loading an inception model that has more than one output layer).

    :return: checkpoint
    """

    # Handle incompatibility between pytorch0.4 and pytorch0.4.x
    # Source: https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560/2

    import torch._utils
    try:
        torch._utils._rebuild_tensor_v2
    except AttributeError:
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor

        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    checkpoint = None
    if checkpoint_path:
        # load data directly from a checkpoint
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            print('=> Loading checkpoint: {} onto device: {}'.format(checkpoint_path, device))
            checkpoint = torch.load(checkpoint_path, map_location=device)

            pretrained_state = checkpoint['state_dict']
            print("INFO: => loaded checkpoint {} (epoch {})".format(checkpoint_path, checkpoint.get('epoch')))
            print('INFO: => checkpoint model name: ', checkpoint.get('modelname', checkpoint.get('model_name')), ' Make sure the checkpoint model name matches your model!!!')
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_path)

        # If the state_dict was saved from parallelized process the key names will start with 'module.'
        # If using ModelCheckpoint the model should already be correctly saved regardless of whether the model was parallelized or not
        is_parallel = False
        for key in pretrained_state:
            if key.startswith('module.'):
                is_parallel = True
                break

        if is_parallel:         # do the work of re-assigning each key (must create a copy due to the use of OrderedDict)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_state.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            checkpoint['state_dict'] = new_state_dict

        # finally load the model weights
        if model:
            print('INFO: => Attempting to load checkpoint data onto model. Device: {}    Strict: {}'.format(device, strict))
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
    return checkpoint


def load_model(model_type: ModelType, model_name: str, num_classes: int, pretrained: bool = True, **kwargs):
    """
    Certain timm models may exist but not be listed in torch.hub so uses a custom partial function to bypass the model check in pywick

    :param model_type:
    :param model_name:
    :param num_classes:
    :param pretrained:
    :param kwargs:
    :return:
    """
    custom_func = partial(torch.hub.load, github=rwightman_repo)
    model = get_model(model_type=model_type, model_name=model_name, num_classes=num_classes, pretrained=pretrained, custom_load_fn=custom_func, **kwargs)

    return model
