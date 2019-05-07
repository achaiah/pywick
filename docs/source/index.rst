Welcome to Pywick!
========================

About
^^^^^
Pywick is a high-level Pytorch training framework that aims to get you up and running quickly with state of the art neural networks.
Does the world need another Pytorch framework? Probably not. But we started this project when no good frameworks were available and
it just kept growing. So here we are.

Pywick tries to stay on the bleeding edge of research into neural networks. If you just wish to run a vanilla CNN, this is probably
going to be overkill. However, if you want to get lost in the world of neural networks, fine-tuning and hyperparameter optimization
for months on end then this is probably the right place for you :)

Guide
^^^^^^
We started this project because of the work we were doing on image classification and segmentation so this is where most of
the updates are happening. However, along the way we've added many powerful tools for fine-tuning your results, from specifying
custom *Constraints* on your network layers, to awesomely flexible *Dataloaders* for your data needs, to a variety of standard
and not-so-standard *loss functions* to *Optimizers*, *Regularizers* and *Transforms*. You'll find a pretty decent description of
each one of them in the navigation pane.

And of course, if you have any questions, feel free to drop by our `github page <https://github.com/achaiah/pywick>`_

.. toctree::
    :maxdepth: 1
    :hidden:

    README.md

.. toctree::
    :caption: Getting Started

    classification_guide
    segmentation_guide

.. toctree::
    :caption: Lego Blocks (aka API)
    :maxdepth: 6

    api/pywick.callbacks
    api/conditions
    api/constraints
    api/pywick.datasets
    api/pywick.functions
    api/pywick.gridsearch
    api/initializers
    api/losses
    api/pywick.meters
    api/pywick.models
    api/pywick.optimizers
    api/regularizers
    api/samplers
    api/pywick.transforms

.. toctree::
    :caption: Misc
    :maxdepth: 2

    license
    help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
