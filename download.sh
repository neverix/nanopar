#!/bin/bash
mkdir model_dir
cd model_dir

GIT_LFS_SKIP_SMUDGE=1 git clone  https://huggingface.co/meta-llama/Llama-2-70b llama-2-70b --depth=1
cd llama-2-70b
python ../../download.py meta-llama/Llama-2-70b
echo '{"dim": 8192, "multiple_of": 4096, "ffn_dim_multiplier": 1.3, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone  https://huggingface.co/meta-llama/Llama-2-13b llama-2-13b --depth=1
cd llama-2-13b
python ../../download.py meta-llama/Llama-2-13b
echo '{"dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone  https://huggingface.co/meta-llama/Llama-2-7b llama-2-7b --depth=1
cd llama-2-7b
python ../../download.py meta-llama/Llama-2-7b
echo '{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000}' > params.json
cd ..

git clone https://huggingface.co/EleutherAI/pythia-160m-deduped pythia --depth=1
git clone https://huggingface.co/rinna/bilingual-gpt-neox-4b neox --depth=1

cd ..

wget -c -O pile.parquet 'https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated/resolve/main/data/train-00000-of-01650-f70471ee3deb09c0.parquet'
git lfs clone https://huggingface.co/datasets/Anthropic/hh-rlhf