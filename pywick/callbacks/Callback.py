from typing import Collection


class Callback:
    """
    Abstract base class used to build new callbacks. Extend this class to build your own callbacks and overwrite functions
    that you want to monitor. Functions will be called automatically from the trainer once per relevant training event
    (e.g. at the beginning of epoch, end of epoch, beginning of batch, end of batch etc.)
    """

    def __init__(self):
        self.params = None
        self.trainer = None

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, epoch: int, logs: Collection = None):
        """
        Called at the beginning of a new epoch
        :param epoch: epoch number
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Collection = None):
        """
        Called at the end of an epoch
        :param epoch: epoch number
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass

    def on_batch_begin(self, batch: int, logs: Collection = None):
        """
        Called at the beginning of a new batch
        :param batch: batch number
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass

    def on_batch_end(self, batch: int, logs: Collection = None):
        """
        Called at the end of an epoch
        :param batch: batch number
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass

    def on_train_begin(self, logs: Collection = None):
        """
        Called at the beginning of a new training run
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass

    def on_train_end(self, logs: Collection = None):
        """
        Called at the end of a training run
        :param logs: collection of logs to process / parse / add to
        :return:
        """
        pass
