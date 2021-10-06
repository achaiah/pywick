from pywick.datasets.tnt.batchdataset import BatchDataset
from pywick.datasets.transformdataset import TransformDataset
from pywick.datasets.shuffledataset import ShuffleDataset
from pywick.datasets.multipartitiondataset import MultiPartitionDataset
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("CustomRange index out of range")
        pass

    def batch(self, *args, **kwargs):
        return BatchDataset(self, *args, **kwargs)

    def transform(self, *args, **kwargs):
        return TransformDataset(self, *args, **kwargs)

    def shuffle(self, *args, **kwargs):
        return ShuffleDataset(self, *args, **kwargs)

    def parallel(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def partition(self, *args, **kwargs):
        return MultiPartitionDataset(self, *args, **kwargs)
