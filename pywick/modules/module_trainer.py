"""
ModuleTrainer for high level training on Pytorch models
"""

import functools
import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn

# local imports
from ._utils import (_validate_loss_input, _validate_metric_input,
                     _validate_optimizer_input, _validate_initializer_input,
                     _parse_num_inputs_and_targets, _parse_num_inputs_and_targets_from_loader,
                     _add_regularizer_to_loss_fn)

from ..conditions import ConditionsContainer, CondType
from ..callbacks import CallbackContainer, History, TQDM
from ..regularizers import RegularizerContainer, RegularizerCallback
from ..initializers import InitializerContainer
from ..constraints import ConstraintContainer, ConstraintCallback
from ..metrics import MetricContainer, MetricCallback
from ..misc import ExecType, is_tuple_or_list

from tqdm import tqdm


class ModuleTrainer(object):

    def __init__(self, model, cuda_devices=[]):
        """
        ModelTrainer for high-level training of Pytorch models

        Major Parts
        -----------
        - optimizer(s)
        - criterion(s)
        - loss_multipliers (to handle multiple losses)
        - named_helpers
        - preconditions
        - postconditions
        - regularizers
        - initializers
        - constraints
        - metrics
        - callbacks
        """
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')
        self.model = model
        self.device = "cuda:" + str(cuda_devices[0]) if cuda_devices else "cpu"     # Empty lists in python are False

        # custom loss weights
        self._loss_multipliers = None

        # custom fit helpers
        self._named_helpers = dict()       # custom trainers that can be initialized during compilation time

        # preconditions
        self._preconditions = []
        self._has_preconditions = False

        # postconditions
        self._postconditions = []
        self._has_postconditions = False

        # callbacks
        self._callbacks = []

        # regularizers
        self._regularizers = []
        self._has_regularizers = False

        # initializers
        self._initializers = []

        # constraints
        self._constraints = []
        self._has_constraints = False

        # metrics
        self._metrics = []
        self._has_metrics = False

        # transforms
        self._transforms = []
        self._has_transforms = False

        # losses
        self._criterion = None
        self._criterion_fn = None

        # other properties
        self._stop_training = False

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True
            # Handle multiple GPUs. Single gpu gets normal treatment while multi-GPU must be wrapped in DataParallel
            if len(cuda_devices) > 1:
                self.model = th.nn.DataParallel(self.model, device_ids=cuda_devices)
        # TODO: This might not be correct. If things break, check here (below line used to be part of the 'if' block above)
        self.model = self.model.to(self.device)

    def set_criterion(self, criterion):
        self._criterion = criterion
        if is_tuple_or_list(criterion):
            self._criterion_fn = [_validate_loss_input(l) for l in criterion]
        else:
            self._criterion_fn = _validate_loss_input(criterion)

    def set_optimizer(self, optimizer, **kwargs):
        if type(optimizer) is type or isinstance(optimizer, str):
            if 'parameters' in kwargs:
                parameters = kwargs['parameters']
            else:
                parameters = self.model.parameters()

            optimizer = _validate_optimizer_input(optimizer)
            self._optimizer = optimizer(parameters, **kwargs)
        else:
            self._optimizer = optimizer

    def set_callbacks(self, callbacks):
        if not is_tuple_or_list(callbacks):
            callbacks = [callbacks]
        self._callbacks = [self.history] + callbacks

    def set_regularizers(self, regularizers):
        regularizers = [regularizers] if not is_tuple_or_list(regularizers) else regularizers
        self._regularizers = regularizers
        self._has_regularizers = True

    def set_initializers(self, initializers):
        initializers = [initializers] if not is_tuple_or_list(initializers) else initializers
        initializers = [_validate_initializer_input(it) for it in initializers]
        self._initializers = initializers

    def set_constraints(self, constraints):
        constraints = [constraints] if not is_tuple_or_list(constraints) else constraints
        self._has_constraints = True
        self._constraints = constraints

    def set_metrics(self, metrics):
        metrics = [metrics] if not is_tuple_or_list(metrics) else metrics
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def set_preconditions(self, conditions):
        conditions = [conditions] if not is_tuple_or_list(conditions) else conditions
        self._preconditions = conditions
        self._has_preconditions = True

    def set_postconditions(self, conditions):
        conditions = [conditions] if not is_tuple_or_list(conditions) else conditions
        self._postconditions = conditions
        self._has_postconditions = True

    def set_transforms(self, transforms):
        if not is_tuple_or_list(transforms):
            transforms = (transforms, lambda x: x, lambda x,y: (x,y))
        if len(transforms) == 1:
            transforms = (transforms, lambda x: x, lambda x,y: (x,y))
        elif len(transforms) == 2:
            transforms = (transforms, transforms, lambda x,y: (x,y))

        self._has_input_transform = transforms[0] is not None
        self._has_target_transform = transforms[1] is not None
        self._has_co_transform = transforms[2] is not None

        self._has_transforms = True
        self._transforms = transforms

    def compile(self,
                optimizer,
                criterion,
                loss_multipliers=None,
                named_helpers=None,
                preconditions=None,
                postconditions=None,
                callbacks=None,
                regularizers=None,
                initializers=None,
                constraints=None,
                metrics=None,
                transforms=None):
        '''
        :param optimizer: the optimizer to use for learning
        :param criterion: the criterion to use for calculating loss
        :param loss_multipliers: (type: list) A way to provide preset loss multipliers for multi-loss criterions
        :param named_helpers: (type: dict) A way to provide custom handler for loss calculation and forward pass. In most cases not necessary to override.
        :param preconditions: (type: list) Conditions to check for before executing a forward pass (e.g. asserts)
        :param postconditions: (type: list) Conditions to check for after the forward pass (e.g. asserts, dynamic network modification)
        :param callbacks: (type: list) Callbacks to use when calling the fit* functions
        :param regularizers: (type: list) Regularizers to use when calling the fit* functions
        :param initializers: (type: list) Initializers to use when calling the fit* functions
        :param constraints: (type: list) Constraints to use when calling the fit* functions
        :param metrics: (type: list) Metrics to use when calling the fit* functions
        :param transforms: (type: list) Unused at the moment

        :return:
        '''
        self.set_optimizer(optimizer)
        self.set_criterion(criterion)
        self._loss_multipliers = loss_multipliers
        self._named_helpers = named_helpers

        if preconditions is not None or postconditions is not None:
            self._conditions_container = ConditionsContainer(exec_type=ExecType.TRAIN)
            if preconditions is not None:
                self.set_preconditions(preconditions)
                self._conditions_container.add_preconditions(self._preconditions)
            if postconditions is not None:
                self.set_postconditions(postconditions)
                self._conditions_container.add_postconditions(self._postconditions)

        if regularizers is not None:
            self.set_regularizers(regularizers)
            self.regularizer_container = RegularizerContainer(self._regularizers)
            self.regularizer_container.register_forward_hooks(self.model)
        else:
            self._has_regularizers = False

        self.history = History(self)
        self._callbacks = [self.history]
        if callbacks is not None:
            self.set_callbacks(callbacks)


        if initializers is not None:
            self.set_initializers(initializers)
            self.initializer_container = InitializerContainer(self._initializers)
            # actually initialize the model
            self.initializer_container.apply(self.model)

        if constraints is not None:
            self.set_constraints(constraints)
            self.constraint_container = ConstraintContainer(self._constraints)
            self.constraint_container.register_constraints(self.model)
        else:
            self._has_constraints = False

        if metrics is not None:
            self.set_metrics(metrics)
            self.metric_container = MetricContainer(self._metrics)
        else:
            self._has_metrics = False

        if transforms is not None:
            self.set_transforms(transforms)
        else:
            self._has_transforms = False

    def fit(self,
            inputs,
            targets=None,
            val_data=None,
            initial_epoch=0,
            num_epoch=100,
            batch_size=32,
            shuffle=False,
            fit_helper_name=None,
            verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        self.model.train(True)
        # ----------------------------------------------------------------------
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        len_inputs = len(inputs) if not is_tuple_or_list(inputs) else len(inputs[0])

        if val_data is not None:
            if num_targets == 0:
                val_data = (val_data, None)
            if len(val_data) != 2:
                raise Exception('val_data must be a 2-tuple')
            num_val_inputs, num_val_targets = _parse_num_inputs_and_targets(val_data[0], val_data[1])
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise Exception('The number of input/target tensors must be the same for training and validation data\n'
                                 'Num Input tensors: (%i train, %i val), Num Target tensors: (%i train, %i val)' % (num_inputs, num_val_inputs, num_targets, num_val_targets) )
            val_inputs, val_targets = val_data
        has_val_data = val_data is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets, helper_name=fit_helper_name)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._criterion_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn, self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            try:
                for epoch_idx in range(initial_epoch,num_epoch):
                    epoch_logs = {}
                    callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                    if shuffle:
                        inputs, targets = fit_helper.shuffle_arrays(inputs, targets)

                    for batch_idx in range(num_batches):
                        batch_logs = {}
                        callback_container.on_batch_begin(batch_idx, batch_logs)

                        input_batch, target_batch = fit_helper.grab_batch(batch_idx, batch_size, inputs, targets)

                        if self._has_preconditions:
                            precond_logs = self._conditions_container(CondType.PRE, epoch_num=epoch_idx, batch_num=batch_idx, net=self.model, input_batch=input_batch, target_batch=target_batch)
                            batch_logs.update(precond_logs)

                        input_batch, target_batch = fit_helper.move_to_device(self.device, input_batch, target_batch)
                        if self._has_transforms:
                            input_batch, target_batch = fit_helper.apply_transforms(self._transforms, input_batch, target_batch)

                        # ---------------------------------------------
                        self._optimizer.zero_grad()
                        output_batch = fit_forward_fn(input_batch)
                        loss = fit_loss_fn(output_batch, target_batch)
                        loss.backward()
                        self._optimizer.step()
                        # ---------------------------------------------

                        if self._has_regularizers:
                            batch_logs['reg_loss'] = self.regularizer_container.current_value
                        if self._has_metrics:
                            metrics_logs = self.metric_container(input_batch, output_batch, target_batch, is_val=False)
                            batch_logs.update(metrics_logs)
                        if self._has_postconditions:
                            postcond_logs = self._conditions_container(CondType.POST, epoch_idx, batch_idx, self.model, input_batch=input_batch, output_batch=output_batch, target_batch=target_batch)
                            batch_logs.update(postcond_logs)

                        batch_logs['loss'] = loss.item()
                        callback_container.on_batch_end(batch_idx, batch_logs)

                    epoch_logs.update(self.history.batch_metrics)
                    if has_val_data:
                        val_epoch_logs = self.evaluate(val_inputs, val_targets, batch_size=batch_size, verbose=verbose)
                        epoch_logs.update(val_epoch_logs)
                        epoch_logs.update(batch_logs)
                        # TODO how to fix this?
                        # self.history.batch_metrics.update(val_epoch_logs)

                    callback_container.on_epoch_end(epoch_idx, epoch_logs)

                    if self._stop_training:
                        break
            # handles Ctrl-C gracefully
            except KeyboardInterrupt:
                print("||  Caught Ctrl-C -- exiting gracefully  || ")
        self.model.train(mode=False)
        callback_container.on_train_end()

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   initial_epoch=0,
                   num_epoch=100,
                   fit_helper_name = None,
                   verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        self.model.train(mode=True)
        # ----------------------------------------------------------------------
        num_inputs = loader.dataset.num_inputs
        num_targets = loader.dataset.num_targets
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        batch_size = loader.batch_size

        if val_loader is not None:
            num_val_inputs = val_loader.dataset.num_inputs
            num_val_targets = val_loader.dataset.num_targets
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets, helper_name=fit_helper_name)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._criterion_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn, self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': loader.batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            try:
                for epoch_idx in range(initial_epoch, num_epoch):
                    epoch_logs = {}
                    callback_container.on_epoch_begin(epoch_idx, epoch_logs)
                    loader_iter = iter(loader)
                    for batch_idx in range(num_batches):
                        # if batch_idx == 5000 or batch_idx == 10000:
                        #     pdb.set_trace()
                        batch_logs = {}
                        callback_container.on_batch_begin(batch_idx, batch_logs)

                        input_batch, target_batch = fit_helper.grab_batch_from_loader(loader_iter)

                        if self._has_preconditions:
                            precond_logs = self._conditions_container(CondType.PRE, epoch_num=epoch_idx, batch_num=batch_idx, net=self.model, input_batch=input_batch, target_batch=target_batch)
                            batch_logs.update(precond_logs)
                        input_batch, target_batch = fit_helper.move_to_device(self.device, input_batch, target_batch)

                        # ---------------------------------------------
                        self._optimizer.zero_grad()
                        output_batch = fit_forward_fn(input_batch)
                        loss = fit_loss_fn(output_batch, target_batch)
                        loss.backward()
                        self._optimizer.step()
                        # ---------------------------------------------

                        if self._has_regularizers:
                            batch_logs['reg_loss'] = self.regularizer_container.current_value
                        if self._has_postconditions:
                            cond_logs = self._conditions_container(CondType.POST, epoch_num=epoch_idx, batch_num=batch_idx, net=self.model, input_batch=input_batch, output_batch=output_batch, target_batch=target_batch)
                            batch_logs.update(cond_logs)
                        if self._has_metrics:
                            metrics_logs = self.metric_container(input_batch, output_batch, target_batch, is_val=False)
                            batch_logs.update(metrics_logs)

                        batch_logs['loss'] = loss.item()
                        callback_container.on_batch_end(batch_idx, batch_logs)

                    epoch_logs.update(self.history.batch_metrics)
                    if has_val_data:
                        val_epoch_logs = self.evaluate_loader(val_loader, verbose=verbose)
                        self._in_train_loop = False
                        #self.history.batch_metrics.update(val_epoch_logs)
                        #epoch_logs.update(val_epoch_logs)
                        epoch_logs.update(val_epoch_logs)
                        epoch_logs.update(batch_logs)
                        # TODO how to fix this?
                        # self.history.batch_metrics.update(val_epoch_logs)

                    callback_container.on_epoch_end(epoch_idx, epoch_logs)

                    if self._stop_training:
                        break
            # handles Ctrl-C gracefully
            except KeyboardInterrupt:
                print("||  Caught Ctrl-C -- exiting gracefully  || ")
        self.model.train(mode=False)
        callback_container.on_train_end()

    def predict(self,
                inputs,
                batch_size=32,
                pred_helper_name=None,
                verbose=1):
        self.model.train(mode=False)
        # --------------------------------------------------------
        num_inputs, _ = _parse_num_inputs_and_targets(inputs, None)
        len_inputs = len(inputs) if not is_tuple_or_list(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))
        # --------------------------------------------------------

        predict_helper = _get_helper(self, num_inputs, num_targets=0, helper_name=pred_helper_name)
        pred_forward_fn = predict_helper.get_partial_forward_fn(self.model)

        with th.no_grad():          # locally disable grad calculations for forward-pass only
            for batch_idx in range(num_batches):
                input_batch, _ = predict_helper.grab_batch(batch_idx, batch_size, inputs, None)
                inputs = predict_helper.move_to_device(self.device, inputs)
                output_batch = pred_forward_fn(input_batch)

                if batch_idx == 0:
                    len_outputs = 1 if not is_tuple_or_list(output_batch) else len(output_batch)
                    prediction_lists = [[] for _ in range(len_outputs)]

                if len_outputs == 1:
                    prediction_lists[0].append(output_batch)
                else:
                    for out_idx in range(len_outputs):
                        prediction_lists[out_idx].append(output_batch[out_idx])

        final_pred_list = [th.cat(pred_list,0) for pred_list in prediction_lists]
        self.model.train(mode=True)
        return final_pred_list if len_outputs > 1 else final_pred_list[0]

    def predict_loader(self,
                       loader,
                       pred_helper_name=None,
                       verbose=1):
        self.model.train(mode=False)
        # --------------------------------------------------------
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))
        # --------------------------------------------------------

        predict_helper = _get_helper(self, num_inputs, num_targets=0, helper_name=pred_helper_name)
        pred_forward_fn = predict_helper.get_partial_forward_fn(self.model)

        loader_iter = iter(loader)

        _range = tqdm(range(num_batches)) if verbose > 0 else range(num_batches)

        with th.no_grad():  # locally disable grad calculations for forward-pass only
            for batch_idx in _range:
                input_batch, _ = predict_helper.grab_batch_from_loader(loader_iter)
                input_batch, _ = predict_helper.move_to_device(self.device, input_batch)

                output_batch = pred_forward_fn(input_batch)

                if batch_idx == 0:
                    len_outputs = 1 if not is_tuple_or_list(output_batch) else len(output_batch)
                    prediction_lists = [[] for _ in range(len_outputs)]

                if len_outputs == 1:
                    prediction_lists[0].append(output_batch)
                else:
                    for out_idx in range(len_outputs):
                        prediction_lists[out_idx].append(output_batch[out_idx])

        final_pred_list = [th.cat(pred_list,0) for pred_list in prediction_lists]
        self.model.train(mode=True)
        return final_pred_list if len_outputs > 1 else final_pred_list[0]

    def evaluate(self,
                 inputs,
                 targets=None,
                 batch_size=32,
                 eval_helper_name=None,
                 verbose=1):
        self.model.train(mode=False)
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        len_inputs = len(inputs) if not is_tuple_or_list(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = _get_helper(self, num_inputs, num_targets, helper_name=eval_helper_name)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._criterion_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs= {'val_loss': 0.}

        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()

        if self._has_preconditions or self._has_postconditions:
            conditions_container = ConditionsContainer(ExecType.VAL, prefix='val_')
            if self._has_preconditions:
                conditions_container.add_preconditions(self._preconditions)
            if self._has_postconditions:
                conditions_container.add_postconditions(self._postconditions)
            conditions_container.reset()
        else:
            conditions_container = None

        samples_seen = 0
        with th.no_grad():  # locally disable grad calculations for forward-pass only
            for batch_idx in range(num_batches):
                input_batch, target_batch = evaluate_helper.grab_batch(batch_idx, batch_size, inputs, targets)
                if conditions_container:
                    cond_logs = conditions_container(CondType.PRE, epoch_num=None, batch_num=batch_idx, net=self.model, input_batch=input_batch, target_batch=target_batch)
                    eval_logs.update(cond_logs)
                input_batch, target_batch = evaluate_helper.move_to_device(self.device, input_batch, target_batch)

                self._optimizer.zero_grad()
                output_batch = eval_forward_fn(input_batch)
                loss = eval_loss_fn(output_batch, target_batch)

                if conditions_container:
                    cond_logs = conditions_container(CondType.POST, epoch_num=None, batch_num=batch_idx, net=self.model, input_batch=input_batch, output_batch=output_batch, target_batch=target_batch)
                    eval_logs.update(cond_logs)

                eval_logs['val_loss'] = (samples_seen*eval_logs['val_loss'] + loss.item()*len(input_batch)) / (samples_seen+len(input_batch))
                samples_seen += len(input_batch)

                if self._has_metrics:
                    metrics_logs = metric_container(input_batch, output_batch, target_batch, is_val=True)
                    eval_logs.update(metrics_logs)

        self.model.train(mode=True)
        return eval_logs

    def evaluate_loader(self, loader, eval_helper_name=None, verbose=1):

        self.model.train(mode=False)
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.sampler) if loader.sampler else len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = _get_helper(self, num_inputs, num_targets, helper_name=eval_helper_name)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._criterion_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs= {'val_loss': 0.}
        loader_iter = iter(loader)

        if self._has_metrics:
            metric_container = MetricContainer(self._metrics, prefix='val_')
            metric_container.set_helper(evaluate_helper)
            metric_container.reset()

        if self._has_preconditions or self._has_postconditions:
            conditions_container = ConditionsContainer(ExecType.VAL, prefix='val_')
            if self._has_preconditions:
                conditions_container.add_preconditions(self._preconditions)
            if self._has_postconditions:
                conditions_container.add_postconditions(self._postconditions)
            conditions_container.reset()
        else:
            conditions_container = None

        samples_seen = 0
        with th.no_grad():  # locally disable grad calculations for forward-pass only
            for batch_idx in range(num_batches):
                input_batch, target_batch = evaluate_helper.grab_batch_from_loader(loader_iter)
                if conditions_container:
                    cond_logs = conditions_container(CondType.PRE, epoch_num=None, batch_num=batch_idx, net=self.model, input_batch=input_batch, target_batch=target_batch)
                    eval_logs.update(cond_logs)
                input_batch, target_batch = evaluate_helper.move_to_device(self.device, input_batch, target_batch)

                self._optimizer.zero_grad()
                output_batch = eval_forward_fn(input_batch)
                loss = eval_loss_fn(output_batch, target_batch)

                if conditions_container:
                    cond_logs = conditions_container(CondType.POST, epoch_num=None, batch_num=batch_idx, net=self.model, input_batch=input_batch, output_batch=output_batch, target_batch=target_batch)
                    eval_logs.update(cond_logs)

                samples_seen += len(input_batch)
                eval_logs['val_loss'] = (samples_seen*eval_logs['val_loss'] + loss.item()*len(input_batch)) / (samples_seen+len(input_batch))

                if self._has_metrics:
                    metrics_logs = metric_container(input_batch, output_batch, target_batch, is_val=True)
                    eval_logs.update(metrics_logs)

        self.model.train(mode=True)
        return eval_logs

    def summary(self, input_size):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params

            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == self.model):
                hooks.append(module.register_forward_hook(hook))

        # create properties
        summary = OrderedDict()
        hooks = []
        # register forward hooks
        self.model.apply(register_hook)

        if isinstance(input_size[0], (list, tuple)):
            x = [th.rand(1,*in_size) for in_size in input_size]
            self.model(*x)
        else:
            x = th.rand(1,*input_size)
            self.model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return summary

def _get_helper(trainer, num_inputs, num_targets, helper_name=None):
    '''
    :param trainer:
    :param num_inputs:
    :param num_targets:
    :param helper_name: Generally a helper will be determined from number of inputs and targets. However may want to supply your own in some instances.\n
    If a helper_name is specified then num_inputs and num_targets are ignored.
    :return:
    '''
    if not helper_name:
        if (num_inputs == 1) and (num_targets == 1):
            helper = SingleInput_SingleTarget_Helper(trainer._loss_multipliers)

        elif (num_inputs == 1) and (num_targets > 1):
            # use same loss function for all targets if multiple loss fns not explicitly given
            if not is_tuple_or_list(trainer._criterion_fn):
                trainer._criterion_fn = [trainer._criterion_fn] * num_targets
            else:
                if len(trainer._criterion_fn) != num_targets:
                    raise ValueError('must give one loss function for every input if you give multiple')
            helper = SingleInput_MultiTarget_Helper()

        elif (num_inputs == 1) and (num_targets == 0):
            helper = SingleInput_NoTarget_Helper()

        elif (num_inputs > 1) and (num_targets == 1):
            helper = MultiInput_SingleTarget_Helper()

        elif (num_inputs > 1) and (num_targets > 1):
            # use same loss function for all targets if multiple loss fns not explicitly given
            if not is_tuple_or_list(trainer._criterion_fn):
                trainer._criterion_fn = [trainer._criterion_fn] * num_targets
            else:
                if len(trainer._criterion_fn) != num_targets:
                    raise ValueError('must give one loss function for every input if you give multiple')
            helper = MultiInput_MultiTarget_Helper()

        elif (num_inputs > 1) and (num_targets == 0):
            helper = MultiInput_NoTarget_Helper()

    else:
        helper = trainer._named_helpers.get(helper_name)

    return helper

class SingleInput_SingleTarget_Helper(object):

    def __init__(self, loss_multipliers=None):
        '''

        :param loss_multipliers: (type: list) Some networks return multiple losses that are then added together. This optional list\n
            specifies different weights to apply to corresponding losses before they are summed.
        '''
        self.loss_multipliers = loss_multipliers

    def move_to_device(self, device, inputs, targets):
        return inputs.to(device), targets.to(device)

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        target_batch = targets[batch_idx*batch_size:(batch_idx+1)*batch_size]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        return next(loader_iter)        # input_batch, target_batch

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = tforms[1](target_batch)
        input_batch, target_batch = tforms[2](input_batch, target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        total_loss = 0.
        if is_tuple_or_list(output_batch):     # some networks output multiple results (to compute separate losses)
            if self.loss_multipliers:
                assert len(output_batch) == len(self.loss_multipliers)

            for i, output in enumerate(output_batch):
                if self.loss_multipliers:
                    total_loss += loss_fn(output, target_batch) * self.loss_multipliers[i]
                else:
                    total_loss += loss_fn(output, target_batch)
        else:
            total_loss = loss_fn(output_batch, target_batch)

        return total_loss

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class SingleInput_MultiTarget_Helper(object):

    def move_to_device(self, device, inputs, targets):
        return inputs.to(device), [target_.to(device) for target_ in targets]

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = [target_[rand_indices] for target_ in targets]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        target_batch = [target_[batch_idx*batch_size:(batch_idx+1)*batch_size] for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        return next(loader_iter)        # OLD: # input_batch, [target_ for target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx])
                    for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_SingleTarget_Helper(object):
    def move_to_device(self, device, inputs, targets):
        return [input_.to(device) for input_ in inputs], targets.to(device)

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = targets[rand_indices]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [input_[batch_idx*batch_size:(batch_idx+1)*batch_size] for input_ in inputs]
        target_batch = targets[batch_idx*batch_size:(batch_idx+1)*batch_size]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        return next(loader_iter)        # OLD: # [input_ for input_ in input_batch], target_batch

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_MultiTarget_Helper(object):

    def move_to_device(self, device, inputs, targets):
        return [input_.to(device) for input_ in inputs], [target_.to(device) for target_ in targets]

    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = [input_[rand_indices] for input_ in inputs]
        return inputs, targets

    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [input_[batch_idx*batch_size:(batch_idx+1)*batch_size] for input_ in inputs]
        target_batch = [target_[batch_idx*batch_size:(batch_idx+1)*batch_size] for target_ in targets]
        return input_batch, target_batch

    def grab_batch_from_loader(self, loader_iter):
        return next(loader_iter)        # OLD: # [input_ for input_ in input_batch], [target_ for target_ in target_batch]

    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx]) for idx in range(len(output_batch))])

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class SingleInput_NoTarget_Helper(object):
    def move_to_device(self, device, inputs, targets=None):
        return inputs.to(device), None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return input_batch, None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = tforms[0](input_batch)
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)


class MultiInput_NoTarget_Helper(object):

    def move_to_device(self, device, inputs, targets=None):
        return [input_.to(device) for input_ in inputs], None

    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        return inputs, None

    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = [input_[batch_idx*batch_size:(batch_idx+1)*batch_size] for input_ in inputs]
        return input_batch, None

    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return input_batch, None

    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        return input_batch, None

    def forward_pass(self, input_batch, model):
        return model(*input_batch)

    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)

    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)

    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
