#!/usr/bin/env python

from setuptools import setup, find_packages

# read the contents of your README file per https://packaging.python.org/guides/making-a-pypi-friendly-readme/
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pywick',
      version='0.5.6',
      description='High-level batteries-included training framework for Pytorch',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Achaiah',
      install_requires=[
                  'h5py',
                  'hickle',
                  'numpy',
                  'pandas',
                  'pillow',
                  'six',
                  'torch',
                  'torchvision',
                  'tqdm',
            ],
      packages=find_packages(),
      url='https://github.com/achaiah/pywick',
      download_url='https://github.com/achaiah/pywick/archive/v0.5.6.tar.gz',
      keywords=['pytorch', 'classification', 'deep learning', 'neural networks', 'semantic-segmentation', 'framework'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.6',
                   ],
      )
