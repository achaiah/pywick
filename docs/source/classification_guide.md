## Classification

With Pywick it is incredibly easy to perform classification training on your dataset. In a typical scenario you will not need to write any code but rather provide a configuration yaml file. See [configs/train_classifier.yaml](https://github.com/achaiah/pywick/blob/master/pywick/configs/train_classifier.yaml) for configuration options. Most of them are well-documented inside the configuration file.

Your dataset should be arranged such that each directory under your root dir is named after the corresponding class of images that it contains (e.g. 17flowers/colt, 17flowers/daisy etc). You can include multiple `dataroots` directories as a list. As an easy starting point, download [17 flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/) dataset and run [examples/17flowers_split.py](https://github.com/achaiah/pywick/blob/master/examples/17flowers_split.py) to convert it into appropriate directory structure.

Some options you may want to tweak:
- `dataroots` - where to find the training data
- `model_spec` - model to use
- `num_epochs` - number of epochs to train for
- `output_root` - where to save outputs (e.g. trained NNs)
- `use_gpu` - whether to use the GPU(s) for training

Once you are happy with your configuration, simply invoke the pywick training code:
```bash
# change to pywick
cd pywick/pywick
python3 train_classifier.py configs/train_classifier.yaml
```

To see how the training code is structured under the hood and to customize it to your liking, see [train_classifier.py](https://github.com/achaiah/pywick/blob/master/pywick/train_classifier.py).