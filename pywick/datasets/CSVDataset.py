from .BaseDataset import BaseDataset
import numpy as np
import pandas as pd
from .data_utils import _return_first_element_of_list, default_file_reader, _pass_through, _process_transform_argument, _process_co_transform_argument


class CSVDataset(BaseDataset):
    """
    Initialize a Dataset from a CSV file/dataframe. This does NOT
    actually load the data into memory if the ``csv`` parameter contains filepaths.

    :param csv: (string or pandas.DataFrame):
        if string, should be a path to a .csv file which
        can be loaded as a pandas dataframe

    :param input_cols: (list of ints, or list of strings):
        which column(s) to use as input arrays.
        If int(s), should be column indicies.
        If str(s), should be column names

    :param target_cols: (list of ints, or list of strings):
        which column(s) to use as input arrays.
        If int(s), should be column indicies.
        If str(s), should be column names

    :param input_transform: (transform):
        tranform to apply to inputs during runtime loading

    :param target_tranform: (transform):
        transform to apply to targets during runtime loading

    :param co_transform: (transform):
        transform to apply to both inputs and targets simultaneously
        during runtime loading

    :param apply_transforms_individually: (bool):
        Whether to apply transforms to individual inputs or to an input row as a whole (default: False)
    """
    def __init__(self,
                 csv,
                 input_cols=None,
                 target_cols=None,
                 input_transform=None,
                 target_transform=None,
                 co_transform=None,
                 apply_transforms_individually=False):
        assert(input_cols is not None)

        self.input_cols = _process_cols_argument(input_cols)
        self.target_cols = _process_cols_argument(target_cols)

        self.do_individual_transforms = apply_transforms_individually

        self.df = _process_csv_argument(csv)

        self.inputs = _select_dataframe_columns(self.df, self.input_cols)
        self.num_inputs = self.inputs.shape[1]
        self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through

        if self.target_cols is None:
            self.num_targets = 0
            self.has_target = False
        else:
            self.targets = _select_dataframe_columns(self.df, self.target_cols)
            self.num_targets = self.targets.shape[1]
            self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else _pass_through
            self.has_target = True
            self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)

        self.input_loader = default_file_reader
        self.target_loader = default_file_reader

        # The more common use-case would be to apply the transform to the row as a whole, but we support
        # applying transform to individual elements as well (with a flag)
        if self.do_individual_transforms:
            self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
        else:
            self.input_transform = _process_transform_argument(input_transform, 1)

        if self.has_target:
            if self.do_individual_transforms:
                self.target_transform = _process_transform_argument(target_transform, self.num_targets)
                self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)
            else:
                self.target_transform = _process_transform_argument(target_transform, 1)
                self.co_transform = _process_co_transform_argument(co_transform, 1, 1)

    def __getitem__(self, index):
        """
        Index the dataset and return the input + target
        """

        # input_sample = list()
        # for i in range(self.num_inputs):
        #     input_sample.append(self.input_transform[i](self.input_loader(self.inputs[index, i])))

        # input_sample
        if self.do_individual_transforms:
            input_sample = [self.input_transform[i](self.input_loader(self.inputs[index, i])) for i in range(self.num_inputs)]
        else:
            input_sample = self.input_transform[0](self.inputs[index])

        if self.has_target:
            if self.do_individual_transforms:
                target_sample = [self.target_transform[i](self.target_loader(self.targets[index, i])) for i in range(self.num_targets)]
                for i in range(self.min_inputs_or_targets):
                    input_sample[i], target_sample[i] = self.co_transform[i](input_sample[i], target_sample[i])
            else:
                target_sample = self.target_transform[0](self.targets[index])
                input_sample, target_sample = self.co_transform[0](input_sample, target_sample)



            return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
        else:
            return self.input_return_processor(input_sample)

    def split_by_column(self, col):
        """
        Split this dataset object into multiple dataset objects based on
        the unique factors of the given column. The number of returned
        datasets will be equal to the number of unique values in the given
        column. The transforms and original dataframe will all be transferred
        to the new datasets

        Useful for splitting a dataset into train/val/test datasets.

        :param col: (integer or string)
            which column to split the data on.
            if int, should be column index.
            if str, should be column name

        :return: list of new datasets with transforms copied
        """
        if isinstance(col, int):
            split_vals = self.df.iloc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.iloc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        elif isinstance(col, str):
            split_vals = self.df.loc[:,col].values.flatten()

            new_df_list = []
            for unique_split_val in np.unique(split_vals):
                new_df = self.df[:][self.df.loc[:,col]==unique_split_val]
                new_df_list.append(new_df)
        else:
            raise ValueError('col argument not valid - must be column name or index')

        new_datasets = []
        for new_df in new_df_list:
            new_dataset = self.copy(new_df)
            new_datasets.append(new_dataset)

        return new_datasets

    def train_test_split(self, train_size):
        """
        Define a split for the current dataset where some part of it is used for
        training while the remainder is used for testing

        :param train_size: (int): length of the training dataset. The remainder will be
            returned as the test dataset
        :return: tuple of datasets (train, test)
        """
        if train_size < 1:
            train_size = int(train_size * len(self))

        train_indices = np.random.choice(len(self), train_size, replace=False)
        test_indices = np.array([i for i in range(len(self)) if i not in train_indices])

        train_df = self.df.iloc[train_indices,:]
        test_df = self.df.iloc[test_indices,:]

        train_dataset = self.copy(train_df)
        test_dataset = self.copy(test_df)

        return train_dataset, test_dataset

    def copy(self, df=None):
        """
        Creates a copy of itself (including transforms and other params).

        :param df: dataframe to include in the copy. If not specified, uses the
            internal dataframe inside this instance (if any)

        :return:
        """
        if df is None:
            df = self.df

        return CSVDataset(df,
                          input_cols=self.input_cols,
                          target_cols=self.target_cols,
                          input_transform=self.input_transform,
                          target_transform=self.target_transform,
                          co_transform=self.co_transform)


def _process_cols_argument(cols):
    if isinstance(cols, tuple):
        cols = list(cols)
    return cols

def _process_csv_argument(csv):
    if isinstance(csv, str):
        df = pd.read_csv(csv)
    elif isinstance(csv, pd.DataFrame):
        df = csv
    else:
        raise ValueError('csv argument must be string or dataframe')
    return df

def _select_dataframe_columns(df, cols):
    if isinstance(cols[0], str):
        inputs = df.loc[:,cols].values
    elif isinstance(cols[0], int):
        inputs = df.iloc[:,cols].values
    else:
        raise ValueError('Provided columns should be string column names or integer column indices')
    return inputs