import time
import datetime

def _get_current_time():
    time_s = time.time()
    return time_s, datetime.datetime.fromtimestamp(time_s).strftime("%B %d, %Y - %I:%M%p")


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """

    def __init__(self, callbacks=None, queue_length=10):
        self.initial_epoch = -1
        self.final_epoch = -1
        self.has_val_data = False
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_epoch == -1:
            self.initial_epoch = epoch
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.final_epoch < epoch:
            self.final_epoch = epoch

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        self.has_val_data = logs['has_val_data']
        logs = logs or {}
        self.start_time_s, self.start_time_date = _get_current_time()
        logs['start_time'] = self.start_time_date
        logs['start_time_s'] = self.start_time_s
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['initial_epoch'] = self.initial_epoch
        logs['final_epoch'] = self.final_epoch

        logs['final_loss'] = self.trainer.history.epoch_metrics['loss'][-1]
        logs['best_loss'] = min(self.trainer.history.epoch_metrics['loss'])
        if self.has_val_data:
            logs['final_val_loss'] = self.trainer.history.epoch_metrics['val_loss'][-1]
            logs['best_val_loss'] = min(self.trainer.history.epoch_metrics['val_loss'])

        logs['start_time'] = self.start_time_date
        logs['start_time_s'] = self.start_time_s

        time_s, time_date = _get_current_time()
        logs['stop_time'] = time_date
        logs['stop_time_s'] = time_s
        for callback in self.callbacks:
            callback.on_train_end(logs)