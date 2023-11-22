import huggingface_hub
# LFS has a lot of extra junk
huggingface_hub.hf_hub_download("meta-llama/Llama-2-7b", "consolidated.00.pth",
                                local_dir=".", local_dir_use_symlinks=False)