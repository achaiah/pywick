from . import Callback

class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every SuperModule.
    """

    def __init__(self, trainer):
        super(History, self).__init__()
        self.samples_seen = 0.
        self.trainer = trainer

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {
            'loss': []
        }
        self.batch_size = logs['batch_size']
        self.has_val_data = logs['has_val_data']
        self.has_regularizers = logs['has_regularizers']
        if self.has_val_data:
            self.epoch_metrics['val_loss'] = []
        if self.has_regularizers:
            self.epoch_metrics['reg_loss'] = []

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.trainer._optimizer, '_optimizer'):  # accounts for meta-optimizers like YellowFin
            self.lrs = [p['lr'] for p in self.trainer._optimizer._optimizer.param_groups]
        else:
            self.lrs = [p['lr'] for p in self.trainer._optimizer.param_groups]
        self.batch_metrics = {
            'loss': 0.
        }
        if self.has_regularizers:
            self.batch_metrics['reg_loss'] = 0.
        self.samples_seen = 0.

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.epoch_metrics['loss'].append(logs['loss'])
        if logs.get('val_loss'):  # if it exists
            self.epoch_metrics['val_loss'].append(logs['val_loss'])

    def on_batch_end(self, batch, logs=None):
        for k in self.batch_metrics:
            self.batch_metrics[k] = (self.samples_seen * self.batch_metrics[k] + logs[k] * self.batch_size) / (self.samples_seen + self.batch_size)
        self.samples_seen += self.batch_size

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)