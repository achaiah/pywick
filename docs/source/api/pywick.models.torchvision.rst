Torchvision Models
====================================

All standard `torchvision models <https://pytorch.org/docs/stable/torchvision/models.html/>`_
are supported out of the box.

* AlexNet
* Densenet (121, 161, 169, 201)
* GoogLeNet
* Inception V3
* ResNet (18, 34, 50, 101, 152)
* ShuffleNet V2
* SqueezeNet (1.0, 1.1)
* VGG (11, 13, 16, 19)

Keep in mind that if you use torvision loading methods (e.g. ``torchvision.models.alexnet(...)``) you
will get a vanilla pretrained model based on Imagenet with 1000 classes. However, more typically,
you'll want to use a pretrained model with your own dataset (and your own number of classes). In that
case you should instead use Pywick's ``models.model_utils.get_model(...)`` utility function
which will do all the dirty work for you and give you a pretrained model but with your custom
number of classes!