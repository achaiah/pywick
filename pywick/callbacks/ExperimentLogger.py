import csv
import os
import shutil
from collections import OrderedDict
from tempfile import NamedTemporaryFile

from . import Callback


class ExperimentLogger(Callback):
    """
    Generic logger callback for dumping experiment data. Can be extended for more utility.
    """

    def __init__(self, directory, filename='Experiment_Logger.csv', save_prefix='Model_', separator=',', append=True):

        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.save_prefix = save_prefix
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(ExperimentLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            open_type = 'a'
        else:
            open_type = 'w'

        # if append is True, find whether the file already has header
        num_lines = 0
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    for num_lines, l in enumerate(f):
                        pass
                    # if header exists, DONT append header again
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))

        model_idx = num_lines
        REJECT_KEYS = {'has_validation_data'}
        MODEL_NAME = self.save_prefix + str(model_idx)  # figure out how to get model name
        self.row_dict = OrderedDict({'model': MODEL_NAME})
        self.keys = sorted(logs.keys())
        for k in self.keys:
            if k not in REJECT_KEYS:
                self.row_dict[k] = logs[k]

        class CustomDialect(csv.excel):
            delimiter = self.sep

        with open(self.file, open_type) as csv_file:
            writer = csv.DictWriter(csv_file,
                                    fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS],
                                    dialect=CustomDialect)
            if self.append_header:
                writer.writeheader()

            writer.writerow(self.row_dict)
            csv_file.flush()

    def on_train_end(self, logs=None):
        REJECT_KEYS = {'has_validation_data'}
        row_dict = self.row_dict

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.keys = self.keys
        temp_file = NamedTemporaryFile(delete=False, mode='w')
        with open(self.file, 'r') as csv_file, temp_file:
            reader = csv.DictReader(csv_file,
                                    fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS],
                                    dialect=CustomDialect)
            writer = csv.DictWriter(temp_file,
                                    fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS],
                                    dialect=CustomDialect)
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # re-write header with on_train_end's metrics
                    pass
                if row['model'] == self.row_dict['model']:
                    writer.writerow(row_dict)
                else:
                    writer.writerow(row)
        shutil.move(temp_file.name, self.file)