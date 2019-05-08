#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pywick',
      version='0.5.3',
      description='High-level batteries-included training framework for Pytorch',
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
      download_url='https://github.com/achaiah/pywick/archive/v0.5.3.tar.gz',
      keywords=['pytorch', 'classification', 'deep learning', 'neural networks', 'semantic-segmentation', 'framework'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.6',
                   ],
      )
