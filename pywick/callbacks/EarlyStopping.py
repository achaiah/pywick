from . import Callback

class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions

    EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs

    :param monitor: (string in {'val_loss', 'loss'}):
        whether to monitor train or val loss
    :param min_delta: (float):
        minimum change in monitored value to qualify as improvement.
        This number should be positive.
    :param patience: (int):
        number of epochs to wait for improvment before terminating.
        the counter be reset after each improvment
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=5):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.trainer._stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' %
                  (self.stopped_epoch))