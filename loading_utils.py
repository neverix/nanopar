from model import ModelArgs, Transformer, PipelineStage

from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer import parallel_state, tensor_parallel

import numpy as np
import torch

from typing import Optional
from pathlib import Path
import random
import json
import os


# from apex
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    # Ensure that different pipeline MP stages get different seeds.
    # TP seeds are automatically offset by the TP rank by apex.

    seed = seed + (100 * parallel_state.get_pipeline_model_parallel_rank())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed)


def convert_weight_for_tp(weight, key):
    if key.endswith("w2.weight") or key.endswith("wo.weight"):
        # row parallel
        dim = 1
    elif (key.endswith("w1.weight") or key.endswith("w3.weight")
         or key.endswith("wq.weight") or key.endswith("wk.weight") or key.endswith("wv.weight")
         or key.endswith("output.weight") or key.endswith("tok_embeddings.weight")):
        dim = 0
    else:
        return weight
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    chunk_size = weight.shape[dim] // tp_size
    return weight.transpose(dim, 0)[
        chunk_size*tp_rank:chunk_size*(tp_rank+1)
    ].transpose(0, dim)


def main_with_model(main_fn):
    def main(*args,
             model_dir=Path("llama-2-7b"),
             tensor_model_parallel_size = 1,
             pipeline_model_parallel_size = 1,
             virtual_pipeline_model_parallel_size: Optional[int] = None,
             use_sp=False,
             seed: Optional[int] = None,
             **kwargs):
        rank = int(os.environ["RANK"])
        gpus_per_node = torch.cuda.device_count()  # TODO
    
        torch.distributed.init_process_group(backend="nccl",
                                            rank=rank)
        local_rank = rank % gpus_per_node
        torch.cuda.set_device(local_rank)
        world_size = torch.distributed.get_world_size()
        print("Rank:", rank, "World:", world_size)
        
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size
        )
        
        data_parallel_size = (
            world_size // (tensor_model_parallel_size * pipeline_model_parallel_size))
        
        vocab_size = 32_000
        loaded_args = json.load(open(model_dir / "params.json", "r"))
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
        if seed is None:
            seed = random.randrange(0, 2 ^ 31)
        set_random_seed(seed)
    
        models = build_model(lambda args, **_:
                        PipelineStage(Transformer(args,
                                                dtype=torch.float16,
                                                use_sp=use_sp)),
                    wrap_with_ddp,
                    virtual_pipeline_model_parallel_size,
                    args=llama_args)
    
        main_fn(models, dict(
            rank=rank,
            local_rank=local_rank,
            data_parallel_size=data_parallel_size,
            llama_args=llama_args,
            model_dir=model_dir,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            use_sp=use_sp,
            forward_backward_func=forward_backward_func
            ), *args, **kwargs)
    return main


def load_consolidated_weights(models, path: Path):
    state_dict = torch.load(str(path), mmap=True)
    state_dict = {"module.wrapped." + k: v for k, v in state_dict.items()}
    state_dict = {k: state_dict[k] for k in models[0].state_dict().keys()}
    state_dict = {k: convert_weight_for_tp(v, k) for k, v in state_dict.items()}
    models[0].load_state_dict(state_dict)
    del state_dict
