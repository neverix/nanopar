import huggingface_hub
import sys
# LFS has a lot of extra junk
for filename in ["tokenizer.model", "params.json"] + [f"consolidated.{i:02d}.pth" for i in range(8)]:
    huggingface_hub.hf_hub_download(sys.argv[1], filename,
                                    local_dir=".", local_dir_use_symlinks=False)