cadeneroot = 'http://data.lip6.fr/cadene/pretrainedmodels/'
dpnroot = 'https://s3.amazonaws.com/dpn-pytorch-weights/'
drnroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'
torchroot = 'https://download.pytorch.org/models/'


model_urls = {
    'alexnet': torchroot + 'alexnet-owt-4df8aa71.pth',
    'bninception': cadeneroot + 'bn_inception-52deb4733.pth',
    'densenet121': cadeneroot + 'densenet121-fbdb23505.pth',
    'densenet169': cadeneroot + 'densenet169-f470b90a4.pth',
    'densenet201': cadeneroot + 'densenet201-5750cbb1e.pth',
    'densenet161': cadeneroot + 'densenet161-347e6b360.pth',
    'dpn68': dpnroot + 'dpn68-4af7d88d2.pth',
    'dpn68b-extra': dpnroot + 'dpn68b_extra-363ab9c19.pth',
    'dpn92-extra': dpnroot + 'dpn92_extra-fda993c95.pth',
    'dpn98': dpnroot + 'dpn98-722954780.pth',
    'dpn107-extra': dpnroot + 'dpn107_extra-b7f9f4cc9.pth',
    'dpn131': dpnroot + 'dpn131-7af84be88.pth',
    'drn-c-26': drnroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': drnroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': drnroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': drnroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': drnroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': drnroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': drnroot + 'drn_d_105-12b40979.pth',
    'fbresnet152': cadeneroot + 'fbresnet152-2e20f6b4.pth',
    'inception_v3': torchroot + 'inception_v3_google-1a9a5a14.pth',
    'inceptionv4': cadeneroot + 'inceptionv4-8e4777a0.pth',
    'inceptionresnetv2': cadeneroot + 'inceptionresnetv2-520b38e4.pth',
    'nasnetalarge': cadeneroot + 'nasnetalarge-a1897284.pth',
    'nasnetamobile': cadeneroot + 'nasnetamobile-7e03cead.pth',
    'pnasnet5large': cadeneroot + 'pnasnet5large-bf079911.pth',
    'resnet18': torchroot + 'resnet18-5c106cde.pth',
    'resnet34': torchroot + 'resnet34-333f7ec4.pth',
    'resnet50': torchroot + 'resnet50-19c8e357.pth',
    'resnet101': torchroot + 'resnet101-5d3b4d8f.pth',
    'resnet152': torchroot + 'resnet152-b121ed2d.pth',
    'resnext101_32x4d': cadeneroot + 'resnext101_32x4d-29e315fa.pth',
    'resnext101_64x4d': cadeneroot + 'resnext101_64x4d-e77a0586.pth',
    'senet_res50': 'http://ideaflux.net/files/models/senet_res50.pkl',
    'se_resnet50': cadeneroot + 'se_resnet50-ce0d4300.pth',
    'se_resnet101': cadeneroot + 'se_resnet101-7e38fcc6.pth',
    'se_resnet152': cadeneroot + 'se_resnet152-d17c99b7.pth',
    'se_resnext50_32x4d': cadeneroot + 'se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101_32x4d': cadeneroot + 'se_resnext101_32x4d-3b2fe3d8.pth',
    'senet154': cadeneroot + 'senet154-c7b49a05.pth',
    'squeezenet1_0': torchroot + 'squeezenet1_0-a815701f.pth',
    'squeezenet1_1': torchroot + 'squeezenet1_1-f364aa15.pth',
    'vgg11': torchroot + 'vgg11-bbd30ac9.pth',
    'vgg13': torchroot + 'vgg13-c768596a.pth',
    'vgg16': torchroot + 'vgg16-397923af.pth',
    'vgg19': torchroot + 'vgg19-dcbb9e9d.pth',
    'wideresnet50': 'https://s3.amazonaws.com/pytorch/h5models/wide-resnet-50-2-export.hkl',
    'xception': cadeneroot + 'xception-43020ad28.pth'
}