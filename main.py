from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer import parallel_state
from model import ModelArgs
from pathlib import Path
import argparse
import torch
import json
import os


def main(llama_path=Path("llama-2-7b")):
    rank = int(os.environ["RANK"])
    gpus_per_node = torch.cuda.device_count()  # TODO
    
    torch.distributed.init_process_group(backend="nccl",
                                         rank=rank)
    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)
    
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 1  # heh
    virtual_pipeline_model_parallel_size = None
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size
    )
    
    world_size = torch.distributed.get_world_size()
    data_parallel_size = (
        world_size // (tensor_model_parallel_size * pipeline_model_parallel_size))
    
    llama_args = ModelArgs(
        **json.load(open(llama_path / "params.json", "r"))
    )
    
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    torch.backends.cudnn.benchmark = True
    
    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size)
    


if __name__ == "__main__":
    main()
