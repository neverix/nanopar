import huggingface_hub
import sys
# LFS has a lot of extra junk
for filename in ["tokenizer.model", "params.json", "consolidated.00.pth", "consolidated.01.pth"]:
    huggingface_hub.hf_hub_download(sys.argv[1], filename,
                                    local_dir=".", local_dir_use_symlinks=False)