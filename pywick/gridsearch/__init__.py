"""
When trying to find the right hyperparameters for your neural network, sometimes you just have to do a lot of trial and error.
Currently, our Gridsearch implementation is pretty basic, but it allows you to supply ranges of input values for various
metaparameters and then executes training runs in either random or sequential fashion.\n
Warning: this class is a bit underdeveloped. Tread with care.
"""

from .gridsearch import GridSearch
from .pipeline import Pipeline