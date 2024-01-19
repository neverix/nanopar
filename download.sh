#!/bin/bash

GIT_LFS_SKIP_SMUDGE=1 git clone  https://huggingface.co/meta-llama/Llama-2-7b llama-2-7b --depth=1
cd llama-2-7b
git lfs pull --include "tokenizer.model" "params.json"
python ../download.py
cd ..

git clone https://huggingface.co/EleutherAI/pythia-14m --depth=1

wget -c -O pile.parquet 'https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/main/data/train-00000-of-01650-f70471ee3deb09c0.parquet'
git lfs clone https://huggingface.co/datasets/Anthropic/hh-rlhf