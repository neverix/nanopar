#!/bin/bash
python create_data.py --train_ds_size=32
MODEL_TYPE=llama torchrun --nproc-per-node=8 cache_logprobs.py --model_dir model_dir/llama-2-70b ---params_file params.json --tensor_model_parallel_size=8 --global_batch_size=4 --micro_batch_size=4
MODEL_TYPE=llama torchrun --nproc-per-node=8 train_dpo.py --model_dir model_dir/llama-2-70b ---params_file params.json --tensor_model_parallel_size=8 --global_batch_size=1 --micro_batch_size=1