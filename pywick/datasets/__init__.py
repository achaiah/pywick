"""
Datasets are the primary mechanism by which Pytorch assembles training and testing data
to be used while training neural networks. While `pytorch` already provides a number of
handy `datasets <https://pytorch.org/docs/stable/data.html#module-torch.utils.data>`_ and
`torchvision` further extends them to common
`academic sets <https://pytorch.org/docs/stable/torchvision/datasets.html/>`_,
the implementations below provide some very powerful options for loading all kinds of data.
We had to extend the default Pytorch implementation as by default it does not keep track
of some useful metadata. That said, you can use our datasets in the normal fashion you're used to
with Pytorch.
"""

from .BaseDataset import BaseDataset
from .ClonedFolderDataset import ClonedFolderDataset
from .CSVDataset import CSVDataset
from .FolderDataset import FolderDataset
from .MultiFolderDataset import MultiFolderDataset
from .PredictFolderDataset import PredictFolderDataset
from .TensorDataset import TensorDataset
from .UsefulDataset import UsefulDataset
from . import data_utils
from .tnt import *
