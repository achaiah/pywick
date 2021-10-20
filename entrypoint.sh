#!/bin/bash

# run demo if "demo" env variable is set
if [ -n "$demo" ]; then
  # prepare directories
  mkdir -p /data /jobs && cd /data && \
  # get the dataset
#  wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz && \
#  tar xzf 17flowers.tgz && rm 17flowers.tgz && \
  # refactor images into correct structure
#  python /home/pywick/examples/17flowers_split.py && \
#  rm -rf jpg && \
  # train on the dataset
  cd /home/pywick/pywick && python train_classifier.py configs/train_classifier.yaml

# otherwise keep the container alive
else
  echo "running blank container..."
  tail -f /dev/null
fi