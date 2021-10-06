#!/usr/bin/env python

from setuptools import setup, find_packages
from pywick import __version__ as version, __author__ as author, __description__ as descr

# read the contents of your README file per https://packaging.python.org/guides/making-a-pypi-friendly-readme/
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pywick',
      version=version,
      description=descr,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=author,
      install_requires=[
            'albumentations',
            'dill',
            'h5py',
            'numpy',
            'opencv-python-headless',
            'pandas',
            'pillow',
            'prodict',
            'pycm',
            'pyyaml',
            'scipy',
            'requests',
            'scikit-image',
            'setuptools',
            'six',
            'tabulate',
            'torch >= 1.6.0',
            'torchvision',
            'tqdm',
            'yacs',
            'wheel'
            ],
      packages=find_packages(),
      url='https://github.com/achaiah/pywick',
      download_url=f'https://github.com/achaiah/pywick/archive/v{version}.tar.gz',
      keywords=['ai', 'artificial intelligence', 'pytorch', 'classification', 'deep learning', 'neural networks', 'semantic-segmentation', 'framework'],
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.8',
                   ],
      )
