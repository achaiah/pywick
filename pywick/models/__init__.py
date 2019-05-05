"""
Neural network models is what deep learning is all about! While you can download some standard models from
`torchvision <https://pytorch.org/docs/stable/torchvision/models.html/>`_, we strive to create a library of models
that are on the cutting edge of AI. Whenever possible, `we provide pretrained solutions as well!`\n
That said, we didn't come up with any of these on our own so we owe a huge debt of gratitude to the many researchers who have shared
their models and weights on github.\n
**Caution:** While we strive to ensure that all models can be used out of the box, sometimes things become broken due to Pytorch updates
or misalignment of the planets. Please don't yell at us. Gently point out what's broken, or even better, submit a pull request to fix it!\n
**Here Be Dragons:** Aaand one more thing - we constantly plumb the depths of github for new models or tweaks to existing ones. While we don't
list this in the docs, there is a special `testnets` directory with tons of probably broken, semi-working, and at times crazy awesome
models and model-variations. If you're interested in the bleeding edge, that's where you'd look (see ``models.__init__.py`` for what's available)
"""

from . import model_locations