"""
This code trains a neural network with parameters provided by configs/train_classifier.yaml. Feel free to tweak parameters and train on your own data.

To run: >>> python3 train_classifier.py configs/train_classifier.yaml
"""
import datetime
import json
import sys
import time
from datetime import timedelta

import albumentations as Album
import cv2
import torch
import torch.utils.data as data
import yaml
from albumentations.pytorch import ToTensorV2
from pywick import optimizers as optims
from pywick.datasets.ClonedFolderDataset import random_split_dataset
from pywick.datasets.FolderDataset import FolderDataset
from pywick.datasets.MultiFolderDataset import MultiFolderDataset
from pywick.datasets.data_utils import adjust_dset_length
from pywick.dictmodels import ExpConfig
from pywick.initializers import XavierUniform
from pywick.metrics import CategoricalAccuracySingleInput
from pywick.models import load_model, ModelType
from pywick.modules import ModuleTrainer
from pywick.utils import class_factory
from pywick.samplers import ImbalancedDatasetSampler
from pywick.transforms import read_cv2_as_rgb
from pywick.cust_random import set_seed


def load_image(path: str):
    return read_cv2_as_rgb(path)


def main(config: ExpConfig):
    """
    Run training based on the loaded parameters

    :param config:     Configuration to execute
    :return:
    """

    dsets = {}
    dset_loaders = {}
    if not config.val_root:                     # if no validation root provided, we use a part of the full dataset instead
        total_set = MultiFolderDataset(roots=config.dataroots, class_mode='label', default_loader=load_image)
        dsets['train'], dsets['val'] = random_split_dataset(orig_dataset=total_set, splitRatio=config.train_val_ratio, random_seed=config.random_seed)
    else:
        dsets['train'] = MultiFolderDataset(roots=config.dataroots, class_mode='label', default_loader=load_image)
        dsets['val'] = FolderDataset(root=config.val_root, class_mode='label', default_loader=load_image)

    # Trim the datasets to fit correctly onto N devices in batches of size B
    num_devices = 1 if len(config.get('gpu_ids', 0)) == 0 or not config.use_gpu else len(config.gpu_ids)
    batch_size = config.batch_size
    adjust_dset_length(dataset=dsets['train'],
                       num_batches=len(dsets['train']) // (num_devices * batch_size),
                       num_devices=num_devices,
                       batch_size=batch_size)
    adjust_dset_length(dataset=dsets['val'],
                       num_batches=len(dsets['val']) // (num_devices * batch_size),
                       num_devices=num_devices,
                       batch_size=batch_size)

    # we may want to balance the data representation
    if config.auto_balance_dataset:
        dset_loaders['train'] = data.DataLoader(dsets['train'],
                                                sampler=ImbalancedDatasetSampler(dsets['train']),
                                                batch_size=config.batch_size,
                                                num_workers=config.workers,
                                                shuffle=False,
                                                pin_memory=True)
    else:
        dset_loaders['train'] = data.DataLoader(dsets['train'],
                                                batch_size=config.batch_size,
                                                num_workers=config.workers,
                                                shuffle=True,
                                                pin_memory=True)
    dset_loaders['val'] = data.DataLoader(dsets['val'],
                                          batch_size=config.batch_size,
                                          num_workers=config.workers,
                                          shuffle=False,
                                          pin_memory=True)

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}

    device = 'cpu'              # CPU is default but if all checks pass, GPU will be enabled
    if config.use_gpu and torch.cuda.is_available():
        device = 'cuda:{}'.format(config.gpu_ids[0])

    # load appropriate model from pywick's model store
    model = load_model(model_type=ModelType.CLASSIFICATION,
                       model_name=config.model_spec,
                       num_classes=len(dsets['train'].class_to_idx),
                       input_size=None,
                       pretrained=True,
                       force_reload=True)

    mean, std = config.mean_std
    class_to_idx = dsets['train'].class_to_idx

    # Create augmentation and normalization transforms for training + normalization for validation
    data_transforms = {
            'train': Album.Compose([
                # Apply image transforms
                Album.RandomCrop(height=config.input_size+50, width=config.input_size+50, always_apply=True, p=1),
                Album.Resize(height=config.input_size, width=config.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1),
                Album.RandomBrightnessContrast(brightness_limit=(-0.1, 0.3), contrast_limit=0.4, p=0.6),
                Album.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.4, 0.2), rotate_limit=270, p=0.9, border_mode=cv2.BORDER_REPLICATE),
                Album.CoarseDropout(max_holes=14, max_height=12, max_width=12, p=0.5),
                # normalize and convert to tensor
                Album.Compose([Album.Normalize(mean=mean, std=std, always_apply=True, p=1), ToTensorV2()])
            ]),
            'val': Album.Compose([
                Album.Resize(height=config.input_size, width=config.input_size, interpolation=cv2.INTER_LINEAR, always_apply=True, p=1),
                Album.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.2, 0.2), rotate_limit=270, p=1, border_mode=cv2.BORDER_REPLICATE),
                Album.Compose([Album.Normalize(mean=mean, std=std, always_apply=True, p=1), ToTensorV2()])
            ])
        }

    # Set transforms for each dataset
    dsets['train'].transform = lambda in_dict: data_transforms['train'](**in_dict)['image']
    dsets['val'].transform = lambda in_dict: data_transforms['val'](**in_dict)['image']
    
    print(f"Configuration Params: \n{config}")
    print('--------------------------------')
    print(f"Dataset Stats:")
    print(f"Num classes: {len(dsets['train'].classes)}")
    print(f"Train Set Size: {dset_sizes['train']}")
    print(f"Val Set Size: {dset_sizes['val']}")

    # load desired optimizer (torch.optim and pywick.optimizer types supported)
    # optimizer = class_factory(classname=config.optimizer['name'], params_dict=config.optimizer.get('params'))

    optimizer = optims.__dict__[config.optimizer['name']](model.parameters(), **(config.optimizer.get('params').to_dict()))

    if device != 'cpu':
        trainer = ModuleTrainer(model, cuda_devices=config.gpu_ids)
    else:
        trainer = ModuleTrainer(model)

    # set up desired callbacks
    callbacks = []
    if config.save_callback is not None:
        if config.save_callback.name == 'ModelCheckpoint':
            config.save_callback.params['run_id'] = config.exp_id
            config.save_callback.params['addl_k_v'] = {'num_classes': len(dsets['train'].class_to_idx),
                                                       'mean_std': config.mean_std,
                                                       'model_name': config.model_spec,
                                                       'optimizer': config.optimizer.get('name')}
            config.save_callback.params['epoch_log_keys'] = ['val_top_1:acc_metric', 'val_top_5:acc_metric']

        checkpt_callback = class_factory(classname=config.save_callback.name, params_dict=config.save_callback.get('params').to_dict())
        callbacks.append(checkpt_callback)

    # create a scheduler
    if config.scheduler.name == 'OnceCycleLRScheduler':
        config.scheduler['params']['steps_per_epoch'] = len(dset_loaders['train'])
    config.scheduler['params']['epochs'] = config.num_epochs
    config.scheduler['params']['optimizer'] = optimizer
    scheduler = class_factory(classname=config.scheduler['name'], params_dict=config.scheduler.get('params'))
    callbacks.append(scheduler)

    trainer.compile(criterion='cross_entropy',
                    callbacks=[checkpt_callback] if checkpt_callback is not None else None,
                    optimizer=optimizer,
                    # regularizers=regularizers,                # <-- not included in example but can add regularizers
                    # constraints=constraints,                  #     ... and constraints
                    initializers=[XavierUniform(bias=False, module_filter='fc*')],
                    metrics=[CategoricalAccuracySingleInput(top_k=1), CategoricalAccuracySingleInput(top_k=5)])

    start_time = time.time()
    print(f'''Starting Training: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}''')

    trainer.fit_loader(dset_loaders['train'],
                       val_loader=dset_loaders['val'],
                       num_epoch=config.num_epochs,
                       verbose=1)

    print(f'Training Complete (time: {timedelta(seconds=int(time.time() - start_time))})')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise AssertionError("Only one argument is expected: config_path")
    config_path = sys.argv[1]
    # Create a configuration object to run this experiment
    with open(config_path, 'r') as f:
        if config_path.endswith('.yml') or config_path.endswith('.yaml'):
            config = ExpConfig.from_dict(yaml.safe_load(f)['train'])  # loads the 'train' configuration from yaml
        elif config_path.endswith('.json'):
            config = ExpConfig.from_dict(json.load(f)['train'])  # loads the 'train' configuration from json
        else:
            raise Exception(f'Configuration file extension must be either .yaml/.yml or .json')
        config.verify_properties()  # make sure all properties have been set
        set_seed(config.random_seed)

        # if not config.use_gpu or not torch.cuda.is_available():  # this is a known problem / limitation of the multiprocessing module.
        #     import multiprocessing
        #     multiprocessing.set_start_method('fork')  # must set multiprocessing to 'fork' from 'spawn' because the dataloader fails to pickle lambda

    main(config)
