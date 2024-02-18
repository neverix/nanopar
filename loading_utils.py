from models.llama import ModelArgs
from models.neox import NeoXArgs

from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer import parallel_state, tensor_parallel

import numpy as np
import torch

from typing import Optional
from pathlib import Path
import random
import json
import os


class LlamaConsolidatedLoader(dict):
    def __init__(self, path):
        prefix = ".".join(str(path).split(".")[:-2])
        self.files = []
        for i in range(100):
            filename = f"{prefix}.{i:02d}.pth"
            if not os.path.exists(filename):
                continue
            self.files.append(torch.load(f"{prefix}.{i:02d}.pth", mmap=True, map_location=None if torch.cuda.device_count() > 1 else "cuda:0"))

    def __getitem__(self, parameter_name):
        tensors = []
        for f in self.files:
            if parameter_name in f:
                tensors.append(f[parameter_name])
        key_to_dim = {
            "w1": 0,
            "w2": -1,
            "w3": 0,
            "wo": -1,
            "wq": 0,
            "wk": 0,
            "wv": 0,
            "output": 0,
            "tok_embeddings": -1,
            "ffn_norm": None,
            "attention_norm": None,
            "norm": None,
            "rope": None,
        }
        short_name = parameter_name.split(".")[-2]
        dim = key_to_dim[short_name]
        if dim is None:
            return tensors[0]
        return torch.cat(tensors, dim)
    
    def items(self):
        all_keys = set(k for file in self.files for k in file.keys())
        for k in all_keys:
            yield k, self[k]
            


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


def main_with_model(model_provider, model_args_cls):
    def decorator(main_fn):
        def main(*args,
                 
                # model_dir=Path("llama-2-7b"),
                # params_file=Path("params.json"),
                
                model_dir=Path("pythia-14m"),
                params_file=Path("config.json"),
                
                tensor_model_parallel_size = 1,
                pipeline_model_parallel_size = 1,
                virtual_pipeline_model_parallel_size: Optional[int] = None,
                use_sp=False,
                wrap_with_ddp=False,
                seed: Optional[int] = None,
                **kwargs):
            model_dir = Path(model_dir)
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
            
            loaded_args = json.load(open(model_dir / params_file, "r"))
            # vocab_size = 32_000
            # loaded_args["vocab_size"] = vocab_size
            model_args = model_args_cls(
                **{k: v for k, v in loaded_args.items()
                   if k in model_args_cls.__dataclass_fields__},
                device="cuda"
            )
            
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            torch.backends.cudnn.benchmark = True
            
            forward_backward_func = get_forward_backward_func(
                virtual_pipeline_model_parallel_size, pipeline_model_parallel_size)
            if seed is None:
                seed = random.randrange(0, 2 ^ 31)
            set_random_seed(seed)
        
            model_args.use_sp = use_sp
            models = build_model(model_provider,
                        wrap_with_ddp,
                        virtual_pipeline_model_parallel_size,
                        args=model_args)
            torch.cuda.empty_cache()  # frees ~9GB
        
            main_fn(models, dict(
                rank=rank,
                local_rank=local_rank,
                data_parallel_size=data_parallel_size,
                model_args=model_args,
                model_dir=model_dir,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                use_sp=use_sp,
                wrap_with_ddp=wrap_with_ddp,
                forward_backward_func=forward_backward_func,
                world_size=world_size
                ), *args, **kwargs)
        return main
    return decorator


def convert_weight_for_tp(weight, parallel_dimension):
    if parallel_dimension is None:
        return weight
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    chunk_size = weight.shape[parallel_dimension] // tp_size
    return weight.transpose(parallel_dimension, 0)[
        chunk_size*tp_rank:chunk_size*(tp_rank+1)
    ].transpose(0, parallel_dimension)

def parallel_dimension_llama(key):
    if key.endswith("w2.weight") or key.endswith("wo.weight"):
        # row parallel
        return 1
    elif (key.endswith("w1.weight") or key.endswith("w3.weight")
         or key.endswith("wq.weight") or key.endswith("wk.weight") or key.endswith("wv.weight")
         or key.endswith("output.weight") or key.endswith("tok_embeddings.weight")):
        return 0
    else:
        return None

def load_consolidated_llama_weights(models, path: Path, wrap_with_ddp: bool):
    state_dict = LlamaConsolidatedLoader(path)
    state_dict = {f"{'module.' if wrap_with_ddp else ''}wrapped." + k: v for k, v in state_dict.items()}
    state_dict = {k: state_dict[k] for k in models[0].state_dict().keys()}
    state_dict = {k: convert_weight_for_tp(v, parallel_dimension_llama(k))
                  for k, v in state_dict.items()}
    missing_keys, unexpected_keys = models[0].load_state_dict(state_dict)
    del state_dict
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)

def parallel_dimension_neox(key):
    if key.endswith("dense_4h_to_h.weight") or key.endswith("dense.weight"):
        # row parallel
        return 1
    elif (key.endswith("dense_h_to_4h.weight") or
         key.endswith("query.weight") or key.endswith("key.weight") or key.endswith("value.weight") or
         key.endswith("embed_in.weight") or key.endswith("embed_out.weight")):
        return 0
    else:
        return None

def load_consolidated_neox_weights(models, model_args: NeoXArgs, path: Path, wrap_with_ddp):
    state_dict = torch.load(str(path), mmap=True)
    prefix = f"{'module.' if wrap_with_ddp else ''}wrapped."
    state_dict = {prefix +
                  k.removeprefix("gpt_neox."): v for k, v in state_dict.items()}
    for layer in range(model_args.num_hidden_layers):
        for param in ("weight", "bias"):
            qkv = state_dict.pop(prefix + f"layers.{layer}.attention.query_key_value.{param}")
            for name, tensor in zip(("query", "key", "value"), qkv.chunk(3)):
                state_dict[prefix + f"layers.{layer}.attention.{name}.{param}"] = tensor
    state_dict = {k: convert_weight_for_tp(v, parallel_dimension_neox(k))
                  for k, v in state_dict.items()}
    # print(models[0].state_dict().keys())
    # exit()
    missing_keys, unexpected_keys = models[0].load_state_dict(state_dict)
    del state_dict
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
