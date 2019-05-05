import json
import math
import os
import shutil

import torch

from . import Callback


class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training. 'Best' is determined by minimizing the value found under monitored_log_key in the logs
    Saved checkpoints contain these keys by default:
        'run_id'
        'epoch'
        'loss_type'
        'loss_val'
        'best_epoch'
        - plus any additional key/value pairs produced by custom_func

    Additionally saves a .json file with statistics about the run such as:
         'run_id'
         'num_epochs'
         'best_epoch'
         'best_loss_or_gain'
         'metric_name'
         - plus any additional key/value pairs produced by custom_func

    :param run_id: (string):
        Uniquely identifies the run
    :param monitored_log_key: (string):
        Name of the key in the logs that will contain the value we want to minimize (and thus that will dictate whether the model is 'best')
    :param save_dir: (string):
        Path indicating where to save the checkpoint
    :param addl_k_v: (dict):
        dictionary of additional key/value pairs to save with the model. Typically these include some initialization parameters, name of the model etc.
        (e.g. from the initialization dictionary 'opt'), as well as other useful params (e.g. mean, std, proc_type: gpu/cpu etc)
    :param epoch_log_keys: (list):
        list of keys to save from the epoch log dictionary (Note: the logs dictionary is automatically provided by the learning framework)
    :param save_interval: (int):
        How often to save the model (if none then will default to every 5 iterations)
    :param save_best_only: (bool):
        Whether only to save the best result (and overwrite all previous)
        Default: False
    :param max_saves: (integer > 0 or -1):
        the max number of models to save. Older model checkpoints will be overwritten if necessary.
        Set equal to -1 to have no limit.
        Default: 5
    :param custom_func: func(k_v_dict, logs, out_dict, monitored_log_key, is_end_training):
        Custom function for performing any additional logic (to add values to the model). The function will be passed the addl_k_v dictionary,
        the event logs dictionary, an output dictionary to process, the monitored_log_key and a bool indicating whether the training is finished.
        The function is expected to modify the output dictionary in order to preserve values across epochs. The function will be called at the
        end of each epoch and at the end of the training (with is_end_traing = True)
    :param do_minimize: (bool):
        whether to minimize or maximize the 'monitored_log_key' value
    :param verbose: (bool):
        verbosity of the console output
        Default: False
    """

    def __init__(self, run_id, monitored_log_key, save_dir, addl_k_v=dict(), epoch_log_keys=[], save_interval=5, save_best_only=False, max_saves=5,
                 custom_func=None, do_minimize=True, verbose=False):

        self.run_id = run_id
        self.addl_k_v = addl_k_v
        self.save_dir = os.path.expanduser(save_dir)
        self.save_interval = save_interval
        self.epoch_log_keys = epoch_log_keys
        self.save_best_only = save_best_only
        self.max_saves = max_saves
        self.custom_func = custom_func
        self.custom_func_dict = dict()  # this is expected to be filled by the custom_func
        self.verbose = verbose
        self.monitored_log_key = monitored_log_key  # 'e.g. dice_coeff'
        self.do_minimize = do_minimize
        self.last_saved_ep = 0
        self.last_epoch_logs = None
        self.last_epoch = -1
        self.best_epoch = -1

        # keep track of old files if necessary
        if self.max_saves > 0:
            self.old_files = []

        # mode = 'min' only supported
        if do_minimize:
            self.best_loss = math.inf
        else:
            self.best_loss = -89293.923
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # import pdb
        # pdb.set_trace()
        self.last_epoch_logs = logs
        self.last_epoch = epoch

        if ((epoch + 1) % self.save_interval == 0):  # only save with given frequency
            current_loss = logs.get(self.monitored_log_key)

            if (current_loss < self.best_loss and self.save_best_only) or not self.save_best_only or (not self.do_minimize and current_loss > self.best_loss):
                if current_loss is None:
                    pass
                else:
                    # Call custom function (if set) to process things like best-N results etc
                    if self.custom_func is not None:
                        self.custom_func(self.addl_k_v, logs, self.custom_func_dict, False)

                    checkpt_name = generate_checkpoint_name(self.run_id, self.addl_k_v, epoch, False)

                    if self.verbose:
                        print('\nEpoch %i: loss metric changed from %0.4f to %0.4f saving model to %s' % (
                            epoch + 1, self.best_loss, current_loss, os.path.join(self.save_dir, checkpt_name)))

                    if (self.do_minimize and current_loss < self.best_loss) or (not self.do_minimize and current_loss > self.best_loss):
                        self.best_loss = current_loss
                        self.best_epoch = epoch
                        # print('Best Loss of {} saved at epoch: {}'.format(self.best_loss, epoch + 1))

                    save_dict = {
                        'run_id': self.run_id,
                        'epoch': epoch + 1,
                        'metric_type': self.monitored_log_key,
                        'metric_value': current_loss,
                        'best_epoch': self.best_epoch + 1
                    }
                    # correctly handle saving parallelized models (https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models)
                    if isinstance(self.trainer.model, torch.nn.DataParallel):
                        save_dict['state_dict'] = self.trainer.model.module.state_dict()
                    else:
                        save_dict['state_dict'] = self.trainer.model.state_dict()

                    # add values from other dictionaries
                    save_dict.update(self.addl_k_v)
                    save_dict.update(self.custom_func_dict)
                    for key in self.epoch_log_keys:
                        save_dict[key] = logs.get(key)  # this is not guaranteed to be found so may return 'None'

                    save_checkpoint(save_dict, is_best=(self.best_epoch == epoch), save_path=self.save_dir, filename=checkpt_name)
                    self.last_saved_ep = epoch

                if self.max_saves > 0:
                    if len(self.old_files) >= self.max_saves:
                        try:
                            os.remove(self.old_files[0])
                        except:
                            pass
                        self.old_files = self.old_files[1:]
                    self.old_files.append(os.path.join(self.save_dir, checkpt_name))

    def on_train_end(self, logs=None):
        final_epoch = self.last_epoch
        current_loss = self.last_epoch_logs[self.monitored_log_key]

        ## Save model if it hasn't been previously saved and it has best loss value
        if self.last_saved_ep < final_epoch and ((self.do_minimize and current_loss < self.best_loss) or (not self.do_minimize and current_loss > self.best_loss)):
            # Call custom function (if set) to process things like best-N results etc
            if self.custom_func is not None:
                self.custom_func(self.addl_k_v, self.last_epoch_logs, self.custom_func_dict, False)

            self.best_loss = current_loss
            self.best_epoch = final_epoch
            save_dict = {
                'run_id': self.run_id,
                'epoch': final_epoch + 1,
                'state_dict': self.trainer.model.state_dict(),
                'metric_type': self.monitored_log_key,
                'metric_value': current_loss,
                'best_epoch': self.best_epoch
            }
            # add values from other dictionaries
            save_dict.update(self.addl_k_v)
            save_dict.update(self.custom_func_dict)
            for key in self.epoch_log_keys:
                save_dict[key] = self.last_epoch_logs[key]

            save_checkpoint(save_dict, is_best=True, save_path=self.save_dir, filename=generate_checkpoint_name(self.run_id, self.addl_k_v, final_epoch, False))
            self.last_saved_ep = final_epoch

        stats = {'run_id': self.run_id,
                 'num_epochs': final_epoch + 1,
                 'best_epoch': self.best_epoch + 1,
                 'best_loss_or_gain': self.best_loss,
                 'metric_type': self.monitored_log_key
                 }
        stats.update(self.addl_k_v)
        stats.update(self.custom_func_dict)
        statsfile_path = generate_statsfile_name(self.run_id, self.save_dir)
        with open(statsfile_path, 'a') as statsfile:
            json.dump(stats, statsfile)


def generate_statsfile_name(run_id, save_dir):
    save_dir1 = os.path.expanduser(save_dir)
    return os.path.join(save_dir1, str(run_id) + "_stats.json")


def generate_checkpoint_name(run_id, kv_dict, epoch, is_best):
    model_name = kv_dict.get('model_name', 'model')
    optimizer_name = kv_dict.get('optimizer', 'o')
    if is_best:
        return str(run_id) + "_" + model_name + "_" + optimizer_name + "_ep_best.pth.tar"
    else:
        return str(run_id) + "_" + model_name + "_" + optimizer_name + "_ep_" + str(epoch + 1) + ".pth.tar"


def save_checkpoint(state, is_best=False, save_path=".", filename=None):
    """
    Saves checkpoint to file.

    :param state: (dict): the dictionary to save. Can have other values besides just model weights.
    :param is_best: (bool): whether this is the best result we've seen thus far
    :param save_path: (string): local dir to save to
    :param filename: (string): name of the file to save under `save_path`

    :return:
    """
    if not filename:
        print("ERROR: No filename defined.  Checkpoint is NOT saved.")
    save_path1 = os.path.expanduser(save_path)
    if not os.path.exists(save_path1): os.makedirs(save_path1)
    torch.save(state, os.path.join(save_path1, filename))
    if is_best:
        pos = filename.find("_ep_")
        if pos and pos > 0:
            bestname = filename[:pos] + "_best.pth.tar"
        shutil.copyfile(os.path.join(save_path1, filename), os.path.join(save_path1, bestname))
