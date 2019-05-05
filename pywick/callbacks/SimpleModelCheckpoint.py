import os
import shutil

import torch

from . import Callback


class SimpleModelCheckpoint(Callback):
    """
    Simple Checkpoint to save model weights during training. This class is mostly superceded by ModelCheckpoint which provides flexible saving functionality.

    :param file: (string):
        file to which model will be saved.
        It can be written 'filename_{epoch}_{loss}' and those
        values will be filled in before saving.
    :param monitor: (string in {'val_loss', 'loss'}):
        whether to monitor train or val loss
    :param save_best_only: (bool):
        whether to only save if monitored value has improved
    :param save_weights_only: (bool):
        whether to save entire model or just weights
        NOTE: only `True` is supported at the moment
    :param max_save: (integer > 0 or -1):
        the max number of models to save. Older model checkpoints
        will be overwritten if necessary. Set equal to -1 to have
        no limit
    :param verbose: (integer in {0, 1}):
        verbosity level

    """

    def __init__(self,
                 directory,
                 filename='ckpt.pth.tar',
                 monitor='val_loss',
                 save_best_only=False,
                 save_weights_only=True,
                 max_save=-1,
                 verbose=0):
        if directory.startswith('~'):
            directory = os.path.expanduser(directory)
        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_save = max_save
        self.verbose = verbose

        if self.max_save > 0:
            self.old_files = []

        # mode = 'min' only supported
        self.best_loss = float('inf')
        super(SimpleModelCheckpoint, self).__init__()

    def save_checkpoint(self, epoch, file, is_best=False):
        """
        Saves checkpoint to file
        :param epoch: (int): epoch number
        :param file: (string): file location
        :param is_best: (bool): whether this is the best result seen thus far
        :return:
        """
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.trainer.model.state_dict(),
            'optimizer': self.trainer._optimizer.state_dict(),
        }, file)
        if is_best:
            shutil.copyfile(file, 'model_best.pth.tar')

    def on_epoch_end(self, epoch, logs=None):

        file = self.file.format(epoch='%03i' % (epoch + 1),
                                loss='%0.4f' % logs[self.monitor])
        if self.save_best_only:
            current_loss = logs.get(self.monitor)
            if current_loss is None:
                pass
            else:
                if current_loss < self.best_loss:
                    if self.verbose > 0:
                        print('\nEpoch %i: improved from %0.4f to %0.4f saving model to %s' %
                              (epoch + 1, self.best_loss, current_loss, file))
                    self.best_loss = current_loss
                    # if self.save_weights_only:
                    # else:
                    self.save_checkpoint(epoch, file)
                    if self.max_save > 0:
                        if len(self.old_files) == self.max_save:
                            try:
                                os.remove(self.old_files[0])
                            except:
                                pass
                            self.old_files = self.old_files[1:]
                        self.old_files.append(file)
        else:
            if self.verbose > 0:
                print('\nEpoch %i: saving model to %s' % (epoch + 1, file))
            self.save_checkpoint(epoch, file)
            if self.max_save > 0:
                if len(self.old_files) == self.max_save:
                    try:
                        os.remove(self.old_files[0])
                    except:
                        pass
                    self.old_files = self.old_files[1:]
                self.old_files.append(file)
