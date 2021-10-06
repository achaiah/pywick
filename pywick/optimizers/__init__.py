"""
Optimizers govern the path that your neural network takes as it tries to minimize error.
Picking the right optimizer and initializing it with the right parameters will either make your network learn successfully
or will cause it not to learn at all! Pytorch already implements the most widely used flavors such as SGD, Adam, RMSProp etc.
Here we strive to include optimizers that Pytorch has missed (and any cutting edge ones that have not yet been added).
"""

from .a2grad import A2GradInc, A2GradExp, A2GradUni
from .adabelief import AdaBelief
from .adahessian import Adahessian
from .adamp import AdamP
from .adamw import AdamW
from .addsign import AddSign
from .apollo import Apollo
from .eve import Eve
from .lars import Lars
from .lookahead import Lookahead
from .lookaheadsgd import LookaheadSGD
from .madgrad import MADGRAD
from .nadam import Nadam
from .powersign import PowerSign
from .qhadam import QHAdam
from .radam import RAdam
from .ralamb import Ralamb
from .rangerlars import RangerLars
from .sgdw import SGDW
from .swa import SWA
from torch.optim import *
