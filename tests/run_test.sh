#!/usr/bin/env bash
set -e

PYCMD=${PYCMD:="python"}
if [ "$1" == "coverage" ];
then
    coverage erase
    PYCMD="coverage run --parallel-mode --source torch "
    echo "coverage flag found. Setting python command to: \"$PYCMD\""
fi

pushd "$(dirname "$0")"

$PYCMD test_meters.py
$PYCMD unit/transforms/test_affine_transforms.py
$PYCMD unit/transforms/test_image_transforms.py
$PYCMD unit/transforms/test_tensor_transforms.py
