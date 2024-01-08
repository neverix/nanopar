#!/bin/bash
set -e
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if [ ! -d apex ] ; then
    git clone https://github.com/NVIDIA/apex
fi
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
python -m pip install --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" --config-settings "--build-option=--distributed_adam" --config-settings "--build-option=--deprecated_fused_adam" ./
python -m pip install -q mosaicml-streaming sentencepiece pandas pyarrow fire