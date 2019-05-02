import csv
import os
from collections import Iterable
from collections import OrderedDict

import torch

from . import Callback


class CSVLogger(Callback):
    """
    Logs epoch-level metrics to a CSV file

    :param file: (string) path to csv file
    :param separator: (string) delimiter for file
    :param append: (bool) whether to append result to existing file or make new file
    """

    def __init__(self, file, separator=',', append=False):

        self.file = file
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.file, 'a')
        else:
            self.csv_file = open(self.file, 'w')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        RK = {'num_batches', 'num_epoch'}

        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, torch.Tensor) and k.dim() == 0
            if isinstance(k, Iterable) and not is_zero_dim_tensor:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + [k for k in self.keys if k not in RK],
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys if key not in RK)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None