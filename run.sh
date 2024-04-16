#!/bin/bash
set -e
MODEL_TYPE=nanopar torchrun --nproc-per-node=2 check_activations.py --model_dir model_dir/llama-2-7b ---params_file params.json --tensor_model_parallel_size=2 --te_params=0` --te_forward=0
MODEL_TYPE=llama torchrun --nproc-per-node=2 check_activations.py --model_dir model_dir/llama-2-7b ---params_file params.json
exit
python create_data.py --train_ds_size=2048
MODEL_TYPE=llama torchrun --nproc-per-node=8 cache_logprobs.py --model_dir model_dir/llama-2-70b ---params_file params.json --tensor_model_parallel_size=8 --use_te=1 --global_batch_size=128 --micro_batch_size=128
# MODEL_TYPE=llama WANDB_MODE=online torchrun --nproc-per-node=8 train_dpo.py --model_dir model_dir/llama-2-70b ---params_file params.json --tensor_model_parallel_size=8 --global_batch_size=2 --micro_batch_size=2 --use_te=1 --grad_acc 1 --train_steps 1000 