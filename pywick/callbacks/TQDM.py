from tqdm import tqdm
from . import Callback

class TQDM(Callback):

    def __init__(self):
        """
        TQDM Progress Bar callback

        This callback is automatically applied to
        every SuperModule if verbose > 0
        """
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.progbar = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar.set_description('Epoch %i/%i' %
                                         (epoch + 1, self.train_logs['num_epoch']))
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.trainer.history.batch_metrics.items()}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
            else:
                log_data[k] = v
        log_data['learn_rates'] = self.trainer.history.lrs
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch, logs=None):
        self.progbar.update(1)

    def on_batch_end(self, batch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.trainer.history.batch_metrics.items()}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
        log_data['learn_rates'] = self.trainer.history.lrs
        self.progbar.set_postfix(log_data)