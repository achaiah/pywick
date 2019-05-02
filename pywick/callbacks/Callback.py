class Callback(object):
    """
    Abstract base class used to build new callbacks. Extend this class to build your own callbacks and overwrite functions
    that you want to monitor. Functions will be called automatically from the trainer once per relevant training event
    (e.g. at the beginning of epoch, end of epoch, beginning of batch, end of batch etc.)
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass