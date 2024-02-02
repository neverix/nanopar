# nanopar
Like [tinypar](https://github.com/cat-state/tinypar) but with a different license

## Usage

```
./deps.sh
./download.sh
python create_data.py
# LLaMA inference test
MODEL_TYPE=llama torchrun cache_logprobs.py --model_dir model_dir/llama-2-7b ---params_file params.json --test_inference=1
# NeoX inference test
MODEL_TYPE=neox torchrun cache_logprobs.py --model_dir model_dir/neox --params_file config.json --test_inference=1

# training is WIP
```
