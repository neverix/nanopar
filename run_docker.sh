#!/bin/bash
set -e
sudo docker build -t nanopar .
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --mount type=bind,source="$(pwd)",target=/content nanopar  # nvcr.io/nvidia/pytorch:24.01-py3