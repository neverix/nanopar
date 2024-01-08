import huggingface_hub
# LFS has a lot of extra junk
for filename in ["consolidated.00.pth", "tokenizer.model", "params.json"]:
    huggingface_hub.hf_hub_download("meta-llama/Llama-2-7b", filename,
                                    local_dir=".", local_dir_use_symlinks=False)