from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer import parallel_state, tensor_parallel
from model import ModelArgs, Transformer, PipelineStage
from pathlib import Path
import argparse
import torch
import json
import os


# from apex
def set_random_seed(seed: int):
    """Set random seed for reproducability."""
    # Ensure that different pipeline MP stages get different seeds.
    # TP seeds are automatically offset by the TP rank by apex.

    seed = seed + (100 * parallel_state.get_pipeline_model_parallel_rank())
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def main(llama_path=Path("llama-2-7b")):
    rank = int(os.environ["RANK"])
    gpus_per_node = torch.cuda.device_count()  # TODO
    
    torch.distributed.init_process_group(backend="nccl",
                                         rank=rank)
    local_rank = rank % gpus_per_node
    torch.cuda.set_device(local_rank)
    world_size = torch.distributed.get_world_size()
    print("Rank:", rank, "World:", world_size)
    
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 2  # heh
    virtual_pipeline_model_parallel_size = None
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size
    )
    
    data_parallel_size = (
        world_size // (tensor_model_parallel_size * pipeline_model_parallel_size))
    
    vocab_size = 32_000
    loaded_args = json.load(open(llama_path / "params.json", "r"))
    loaded_args["vocab_size"] = vocab_size
    llama_args = ModelArgs(
        **loaded_args,
        device="cuda"
    )
    
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    torch.backends.cudnn.benchmark = True
    
    forward_backward_func = get_forward_backward_func(
        virtual_pipeline_model_parallel_size, pipeline_model_parallel_size)
    wrap_with_ddp = True
    set_random_seed(12)
    models = build_model(lambda args, **kwargs: PipelineStage(Transformer(args, dtype=torch.float16)),
                        wrap_with_ddp,
                        virtual_pipeline_model_parallel_size,
                        args=llama_args)
    
    state_dict = torch.load(str(llama_path / "consolidated.00.pth"), mmap=True)
    state_dict = {"module.wrapped." + k: v for k, v in state_dict.items()}
    state_dict = {k: state_dict[k] for k in models[0].state_dict().keys()}
    models[0].load_state_dict(state_dict)
    del state_dict
    
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
