#!/bin/bash
set -e
sudo docker build -t nanopar .
API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)
sudo docker run -e WANDB_API_KEY=$API_KEY \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    --mount type=bind,source="$(pwd)/model_dir",target=/content/model_dir \
    --mount type=bind,source="$(pwd)/wandb",target=/content/wandb \
    --mount type=volume,source=data,target=/content/data \
    nanopar ./run.sh  # nvcr.io/nvidia/pytorch:24.01-py3