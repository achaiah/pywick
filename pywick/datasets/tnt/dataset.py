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
        from .batchdataset import BatchDataset
        return BatchDataset(self, *args, **kwargs)

    def transform(self, *args, **kwargs):
        from .transformdataset import TransformDataset
        return TransformDataset(self, *args, **kwargs)

    def shuffle(self, *args, **kwargs):
        from .shuffledataset import ShuffleDataset
        return ShuffleDataset(self, *args, **kwargs)

    def parallel(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def partition(self, *args, **kwargs):
        from .multipartitiondataset import MultiPartitionDataset
        return MultiPartitionDataset(self, *args, **kwargs)

