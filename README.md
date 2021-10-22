# Pywick

<div style="text-align:center">

[![docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fpywick%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://pywick.readthedocs.io/en/latest/)
[![Downloads](https://pepy.tech/badge/pywick)](https://pywick.readthedocs.io/en/latest/)
[![pypi](https://img.shields.io/pypi/v/pywick.svg)](https://pypi.org/project/pywick/)
[![python compatibility](https://img.shields.io/pypi/pyversions/pywick.svg)](https://pywick.readthedocs.io/en/latest/)
[![license](https://img.shields.io/pypi/l/pywick.svg)](https://github.com/achaiah/pywick/blob/master/LICENSE.txt)
 
</div>

#### High-Level Training framework for Pytorch

Pywick is a high-level Pytorch training framework that aims to get you
up and running quickly with state of the art neural networks. *Does the
world need another Pytorch framework?* Probably not. But we started this
project when no good frameworks were available and it just kept growing.
So here we are.

Pywick tries to stay on the bleeding edge of research into neural networks. If you just wish to run a vanilla CNN, this is probably
going to be overkill. However, if you want to get lost in the world of neural networks, fine-tuning and hyperparameter optimization
for months on end then this is probably the right place for you :)

Among other things Pywick includes:
- State of the art normalization, activation, loss functions and optimizers not included in the standard Pytorch library (AdaBelief, Addsign, Apollo, Eve, Lookahead, Radam, Ralamb, RangerLARS etc).
- A high-level module for training with callbacks, constraints, metrics, conditions and regularizers.
- Hundreds of popular object classification and semantic segmentation models!
- Comprehensive data loading, augmentation, transforms, and sampling capability.
- Utility tensor functions.
- Useful meters.
- Basic GridSearch (exhaustive and random).

## Docs
Hey, [check this out](https://pywick.readthedocs.io/en/latest/), we now have [docs](https://pywick.readthedocs.io/en/latest/)! They're still a work in progress though so apologies for anything that's broken.

## What's New (highlights)

### v0.6.5 - Docker all the things!
Another great improvement to the framework - docker! You can now run the 17flowers demo right out of the box!
  - Grab our docker image at [docker hub](https://hub.docker.com/repository/docker/achaiah/pywick): `docker pull achaiah/pywick:latest`. Pytorch 1.8 and cuda dependencies are pre-installed.
  - Run 17flowers demo with: `docker run --rm -it --ipc=host -v your_local_out_dir:/jobs/17flowers --init -e demo=true achaiah/pywick:latest`
  - Or run the container in standalone mode so you can use your own data (don't forget to map your local dir to container):
    ```bash
      docker run --rm -it \
      --ipc=host \
      -v <your_local_data_dir>:<container_data_dir> \
      -v <your_local_out_dir>:<container_out_dir> \
      --init \
      achaiah/pywick:latest
    ```

### Older Notes
- **Oct. 11, 2021 - We thought ya might like YAML!**
  - So you're saying you like **configuration files**? You're saying you like **examples** too? Well, we've got you covered! Huge release today with a configuration-based training example! All you have to do is:
    - Get your favorite dataset (or download [17 flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) to get started and `pywick/examples/17flowers_split.py` to convert)
    - Adjust the `configs/train_classifier.yaml` file to fit your workspace
    - Then simply run: `python3 train_classifier.py configs/train_classifier.yaml` and watch it train!
- **May 6, 2021**
  - Many SoTA classification and segmentation models added: Swin-Transformer variants, NFNet variants (L0, L1), Halo nets, Lambda nets, ECA variants, Rexnet + others
  - Many new loss functions added: RecallLoss, SoftInvDiceLoss, OhemBCEDicePenalizeBorderLoss, RMIBCEDicePenalizeBorderLoss + others
  - Bug fixes
- **Jun. 15, 2020**
  - 700+ models added from [rwightman's](https://github.com/rwightman/pytorch-image-models) repo via `torch.hub`! See docs for all the variants!
  - Some minor bug fixes
- **Jan. 20, 2020**
  - New release: 0.5.6 (minor fix from 0.5.5 for pypi)
  - Mish activation function (SoTA)
  - [rwightman's](https://github.com/rwightman/gen-efficientnet-pytorch) models of pretrained/ported variants for classification (44 total)
    - efficientnet Tensorflow port b0-b8, with and without AP, el/em/es, cc
    - mixnet L/M/S
    - mobilenetv3
    - mnasnet
    - spnasnet
  - Additional loss functions
- **Aug. 1, 2019**
  - New segmentation NNs: BiSeNet, DANet, DenseASPP, DUNet, OCNet, PSANet
  - New Loss Functions: Focal Tversky Loss, OHEM CrossEntropy Loss, various combination losses
  - Major restructuring and standardization of NN models and loading functionality
  - General bug fixes and code improvements 

## Install
Pywick requires **pytorch >= 1.4**

`pip install pywick`

or specific version from git:

`pip install git+https://github.com/achaiah/pywick.git@v0.6.5`

## ModuleTrainer
The `ModuleTrainer` class provides a high-level training interface which abstracts away the training loop while providing callbacks, constraints, initializers, regularizers,
and more.

See the `train_classifier.py` example for a pretty complete configuration example. To get up and running with your own data quickly simply edit the `configs/train_classifier.yaml` file with your desired parameters and dataset location(s).

Note: <i>Dataset needs to be organized for classification where each directory name is the name of a class and contains all images pertaining to that class</i>

PyWick provides a wide range of <b>callbacks</b>, generally mimicking the interface found in `Keras`:

- `CSVLogger` - Logs epoch-level metrics to a CSV file
- [`CyclicLRScheduler`](https://github.com/bckenstler/CLR) - Cycles through min-max learning rate
- `EarlyStopping` - Provides ability to stop training early based on supplied criteria
- `History` - Keeps history of metrics etc. during the learning process
- `LambdaCallback` - Allows you to implement your own callbacks on the fly
- `LRScheduler` - Simple learning rate scheduler based on function or supplied schedule
- `ModelCheckpoint` - Comprehensive model saver
- `ReduceLROnPlateau` - Reduces learning rate (LR) when a plateau has been reached
- `SimpleModelCheckpoint` - Simple model saver
- Additionally, a `TensorboardLogger` is incredibly easy to implement via tensorboardX (now part of pytorch 1.1 release!)


```python
from pywick.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
trainer.set_callbacks(callbacks)
```

PyWick also provides <b>regularizers</b>:

- `L1Regularizer`
- `L2Regularizer`
- `L1L2Regularizer`


and <b>constraints</b>:
- `UnitNorm`
- `MaxNorm`
- `NonNeg`

Both regularizers and constraints can be selectively applied on layers using regular expressions and the `module_filter`
argument. Constraints can be explicit (hard) constraints applied at an arbitrary batch or
epoch frequency, or they can be implicit (soft) constraints similar to regularizers
where the constraint deviation is added as a penalty to the total model loss.

```python
from pywick.constraints import MaxNorm, NonNeg
from pywick.regularizers import L1Regularizer

# hard constraint applied every 5 batches
hard_constraint = MaxNorm(value=2., frequency=5, unit='batch', module_filter='*fc*')
# implicit constraint added as a penalty term to model loss
soft_constraint = NonNeg(lagrangian=True, scale=1e-3, module_filter='*fc*')
constraints = [hard_constraint, soft_constraint]
trainer.set_constraints(constraints)

regularizers = [L1Regularizer(scale=1e-4, module_filter='*conv*')]
trainer.set_regularizers(regularizers)
```

You can also fit directly on a `torch.utils.data.DataLoader` and can have
a validation set as well :

```python
from pywick import TensorDataset
from torch.utils.data import DataLoader

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

trainer.fit_loader(loader, val_loader=val_loader, num_epoch=100)
```

## Extensive Library of Image Classification Models (most are pretrained!)
- All standard models from Pytorch:
  - [**Densenet**](https://arxiv.org/abs/1608.06993)
  - [**Inception v3**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
  - [**MobileNet v2**](https://arxiv.org/abs/1801.04381)
  - [**ResNet**](https://arxiv.org/abs/1512.03385)
  - [**ShuffleNet v2**](https://arxiv.org/abs/1807.11164)
  - [**SqueezeNet**](https://arxiv.org/abs/1602.07360)
  - [**VGG**](https://arxiv.org/abs/1409.1556)
- [**BatchNorm Inception**](https://arxiv.org/abs/1502.03167)
- [**Deep High-Resolution Representation Learning for Human Pose Estimation**](https://arxiv.org/abs/1902.09212v1)
- [**Deep Layer Aggregation**](https://arxiv.org/abs/1707.06484)
- [**Dual Path Networks**](https://arxiv.org/abs/1707.01629)
- [**EfficientNet variants (b0-b8, el, em, es, lite1-lite4, pruned, AP/NS)**](https://arxiv.org/abs/1905.11946)
- [**ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks**](https://arxiv.org/abs/1910.03151v4)
- [**FBResnet**](https://github.com/facebook/fb.resnet.torch)
- [**FBNet-C**](https://arxiv.org/abs/1812.03443)
- [**Inception v4**](http://arxiv.org/abs/1602.07261)
- [**InceptionResnet v2**](https://arxiv.org/abs/1602.07261)
- [**Mixnet variants (l, m, s, xl, xxl)**](https://arxiv.org/abs/1907.09595)
- [**MnasNet**](https://arxiv.org/abs/1807.11626)
- [**MobileNet V3**](https://arxiv.org/abs/1905.02244)
- [**NasNet variants (mnas, pnas, mobile)**](https://arxiv.org/abs/1707.07012)
- [**PNASNet**](https://arxiv.org/abs/1712.00559)
- [**Polynet**](https://arxiv.org/abs/1611.05725)
- [**Pyramid Resnet**](https://arxiv.org/abs/1610.02915)
- [**RegNet - Designing Network Design Spaces**](https://arxiv.org/abs/2003.13678)
- **[Resnet variants (gluon, res2net, se, ssl, tv, wide)](https://arxiv.org/abs/1512.03385)**
- [**ResNeSt: Split-Attention Networks**](https://arxiv.org/abs/2004.08955)
- [**ResNext variants (ig, se, ssl, swsl, tv)**](https://arxiv.org/abs/1611.05431)
- [**SE Net variants (gluon, resnet, resnext, inception)**](https://arxiv.org/pdf/1709.01507.pdf)
- [**SelecSLS Convolutional Net**](https://github.com/mehtadushy/SelecSLS-Pytorch)
- [**Selective Kernel Networks**](https://arxiv.org/abs/1903.06586)
- [**Semi-Supervised and Semi-Weakly Supervised ImageNet Models**](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)
- [**Single-Pass NAS Net**](https://arxiv.org/abs/1904.02877)
- [**TResNet: High Performance GPU-Dedicated Architecture**](https://arxiv.org/abs/2003.13630)
- [**Wide Resnet**](https://arxiv.org/abs/1605.07146)
- [**XCeption**](https://arxiv.org/pdf/1610.02357.pdf)
- All the newest classification models (200+) from [rwightman's repo](https://github.com/rwightman/pytorch-image-models) ECA-NFNet, GERNet, RegNet, SKResnext, SWIN-Transformer, VIT etc.)

## Image Segmentation Models
- **BiSeNet** ([Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897))
- **DANet** ([Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983))
- **Deeplab v2** ([DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915))
- **Deeplab v3** ([Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587))
- **DenseASPP** ([DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf))
- **DRNNet** ([Dilated Residual Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf))
- **DUC, HDC** ([understanding convolution for semantic segmentation](https://arxiv.org/abs/1702.08502))
- **DUNet** ([Decoders Matter for Semantic Segmentation](https://arxiv.org/abs/1903.02120))
- **ENet** ([ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147))
- **Vanilla FCN:** FCN32, FCN16, FCN8, in the versions of VGG, ResNet
    and OptDenseNet respectively ([Fully convolutional networks for semantic segmentation](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf))
- **FRRN** ([Full Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323))
- **FusionNet** ([FusionNet in Tensorflow by Hyungjoo Andrew Cho](https://github.com/NySunShine/fusion-net))
- **GALDNet** 
- **GCN** ([Large Kernel Matters](https://arxiv.org/pdf/1703.02719))
- **LinkNet** ([Link-Net](https://codeac29.github.io/projects/linknet/))
- **OCNet** ([Object Context Network for Scene Parsing](https://arxiv.org/abs/1809.00916))
- **PSPNet** ([Pyramid scene parsing network](https://arxiv.org/abs/1612.01105))
- **RefineNet** ([RefineNet](https://arxiv.org/abs/1611.06612))
- **SegNet** ([Segnet: A deep convolutional encoder-decoder architecture for image segmentation](https://arxiv.org/pdf/1511.00561))
- **Tiramisu** ([The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326))
- **U-Net** ([U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1505.04597))
- Additional variations of many of the above

###### To load one of these models:
[Read the docs](https://pywick.readthedocs.io/en/latest/api/pywick.models.html)
for useful details! Then dive in:
```python
# use the `get_model` utility
from pywick.models.model_utils import get_model, ModelType

model = get_model(model_type=ModelType.CLASSIFICATION, model_name='resnet18', num_classes=1000, pretrained=True)
```
For a complete list of models (including many experimental ones) you can call the `get_supported_models` method e.g. 
`pywick.models.model_utils.get_supported_models(ModelType.SEGMENTATION)`

## Data Augmentation and Datasets
The PyWick package provides wide variety of good data augmentation and transformation
tools which can be applied during data loading. The package also provides the flexible
`TensorDataset`, `FolderDataset` and `MultiFolderDataset` classes to handle most dataset needs.

### Torch Transforms
##### These transforms work directly on torch tensors

- `AddChannel`
- `ChannelsFirst`
- `ChannelsLast`
- `Compose`
- `ExpandAxis`
- `Pad`
- `PadNumpy`
- `RandomChoiceCompose`
- `RandomCrop`
- `RandomFlip`
- `RandomOrder`
- `RangeNormalize`
- `Slice2D`
- `SpecialCrop`
- `StdNormalize`
- `ToFile`
- `ToNumpyType`
- `ToTensor`
- `Transpose`
- `TypeCast`

##### Additionally, we provide image-specific manipulations directly on tensors:

- `Brightness`
- `Contrast`
- `Gamma`
- `Grayscale`
- `RandomBrightness`
- `RandomChoiceBrightness`
- `RandomChoiceContrast`
- `RandomChoiceGamma`
- `RandomChoiceSaturation`
- `RandomContrast`
- `RandomGamma`
- `RandomGrayscale`
- `RandomSaturation`
- `Saturation`

#####  Affine Transforms (perform affine or affine-like transforms on torch tensors)

- `RandomAffine`
- `RandomChoiceRotate`
- `RandomChoiceShear`
- `RandomChoiceTranslate`
- `RandomChoiceZoom`
- `RandomRotate`
- `RandomShear`
- `RandomSquareZoom`
- `RandomTranslate`
- `RandomZoom`
- `Rotate`
- `Shear`
- `Translate`
- `Zoom`

We also provide a class for stringing multiple affine transformations together so that only one interpolation takes place:

- `Affine` 
- `AffineCompose`

##### Blur and Scramble transforms (for tensors)
- `Blur`
- `RandomChoiceBlur`
- `RandomChoiceScramble`
- `Scramble`

### Datasets and Sampling
We provide the following datasets which provide general structure and iterators for sampling from and using transforms on in-memory or out-of-memory data. In particular,
the [FolderDataset](pywick/datasets/FolderDataset.py) has been designed to fit most of your dataset needs. It has extensive options for data filtering and manipulation.
It supports loading images for classification, segmentation and even arbitrary source/target mapping. Take a good look at its documentation for more info.

- `ClonedDataset`
- `CSVDataset`
- `FolderDataset`
- `MultiFolderDataset`
- `TensorDataset`
- `tnt.BatchDataset`
- `tnt.ConcatDataset`
- `tnt.ListDataset`
- `tnt.MultiPartitionDataset`
- `tnt.ResampleDataset`
- `tnt.ShuffleDataset`
- `tnt.TensorDataset`
- `tnt.TransformDataset`

### Imbalanced Datasets
In many scenarios it is important to ensure that your traing set is properly balanced,
however, it may not be practical in real life to obtain such a perfect dataset. In these cases 
you can use the `ImbalancedDatasetSampler` as a drop-in replacement for the basic sampler provided
by the DataLoader. More information can be found [here](https://github.com/ufoym/imbalanced-dataset-sampler)

```python
from pywick.samplers import ImbalancedDatasetSampler

train_loader = torch.utils.data.DataLoader(train_dataset, 
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=args.batch_size, **kwargs)
```

## Utility Functions
PyWick provides a few utility functions not commonly found:

### Tensor Functions
- `th_iterproduct` (mimics itertools.product)
- `th_gather_nd` (N-dimensional version of torch.gather)
- `th_random_choice` (mimics np.random.choice)
- `th_pearsonr` (mimics scipy.stats.pearsonr)
- `th_corrcoef` (mimics np.corrcoef)
- `th_affine2d` and `th_affine3d` (affine transforms on torch.Tensors)


## Acknowledgements and References
We stand on the shoulders of (github?) giants and couldn't have done
this without the rich github ecosystem and community. This framework is
based in part on the excellent
[Torchsample](https://github.com/ncullen93/torchsample) framework
originally published by @ncullen93. Additionally, many models have been
gently borrowed/modified from @Cadene pretrained models
[repo](https://github.com/Cadene/pretrained-models.pytorch) as well as @Tramac segmentation [repo](https://github.com/Tramac/awesome-semantic-segmentation-pytorch).

##### Thank you to the following people and the projects they maintain:
- @ncullen93
- @cadene
- @deallynomore
- @recastrodiaz
- @zijundeng
- @Tramac
- And many others! (attributions listed in the codebase as they occur)

##### Thank you to the following projects from which we gently borrowed code and models
- [PyTorchNet](https://github.com/pytorch/tnt)
- [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)
- [DeepLab_pytorch](https://github.com/doiken23/DeepLab_pytorch)
- [Pytorch for Semantic Segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation)
- [Binseg Pytorch](https://github.com/saeedizadi/binseg_pytoch)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- And many others! (attributions listed in the codebase as they occur)



| *Thangs are broken matey! Arrr!!!* |
|-----------------------|
| We're working on this project as time permits so you might discover bugs here and there. Feel free to report them, or better yet, to submit a pull request! |