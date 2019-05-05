import torch.utils.data.dataset as ds

class UsefulDataset(ds.Dataset):
    '''
    A ``torch.utils.data.Dataset`` class with additional useful functions.
    '''

    def __init__(self):
        self.num_inputs = 1         # these are hardcoded for the fit module to work
        self.num_targets = 1        # these are hardcoded for the fit module to work

    def getdata(self):
        """
        Data that the Dataset class operates on. Typically iterable/list of tuple(label,target).
        Note: This is different than simply calling myDataset.data because some datasets are comprised of multiple other datasets!
        The dataset returned should be the `combined` dataset!

        :return: iterable - Representation of the entire dataset (combined if necessary from multiple other datasets)
        """
        raise NotImplementedError

    def getmeta_data(self):
        """
        Additional data to return that might be useful to consumer. Typically a dict.

        :return: dict(any)
        """
        raise NotImplementedError