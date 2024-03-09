#!/bin/bash
set -e
# python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# python -m pip install packaging torch-utils

if [ ! -d apex ] ; then
    git clone https://github.com/NVIDIA/apex
fi
cd apex
# git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
python -m pip install --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--deprecated_fused_adam" ./
cd ..

if [ ! -d TransformerEngine ] ; then
    git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git
fi
cd TransformerEngine
export NVTE_FRAMEWORK=pytorch   # Optionally set framework
# CUDNN_PATH=~/miniconda3/envs/nanopar
python -m pip install .                   # Build and install
cd ..

python -m pip install -q git+https://github.com/mosaicml/streaming sentencepiece pandas pyarrow fire wandb einops