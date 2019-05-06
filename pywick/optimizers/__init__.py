"""
Optimizers govern the path that your neural network takes as it tries to minimize error.
Picking the right optimizer and initializing it with the right parameters will either make your network learn successfully
or will cause it not to learn at all! Pytorch already implements the most widely used flavors such as SGD, Adam, RMSProp etc.
Here we strive to include optimizers that Pytorch has missed (and any cutting edge ones that have not yet been added).
"""

from .adamw import AdamW
from .addsign import AddSign
from .eve import Eve
from .nadam import Nadam
from .powersign import PowerSign
from .sgdw import SGDW
from .swa import SWA