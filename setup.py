#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='pywick',
      version='0.5.3',
      description='High-level batteries-included training framework for Pytorch',
      author='Achaiah',
      author_email='n/a',
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
      download_url='pip install git+https://github.com/achaiah/pywick.git@v0.5.3',
      keywords=['pytorch', 'classification', 'deep learning', 'neural networks', 'semantic-segmentation', 'framework'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.6',
                   ],
      )
