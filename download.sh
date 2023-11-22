#!/bin/bash

GIT_LFS_SKIP_SMUDGE=1 git clone  https://huggingface.co/meta-llama/Llama-2-7b llama-2-7b --depth=1
cd llama-2-7b
git lfs pull --include "tokenizer.model" "params.json"
python ../download.py