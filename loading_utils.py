from models.llama import ModelArgs
from models.neox import NeoXArgs

from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer import parallel_state, tensor_parallel

import numpy as np
import torch

from typing import Optional
from pathlib import Path
from copy import copy
import random
import json
import gc
import os


LLAMA_KEY_TO_DIM = {
    "w1": 0,
    "w2": 1,
    "w3": 0,
    "wo": 1,
    "wq": 0,
    "wk": 0,
    "wv": 0,
    "output": 0,
    "tok_embeddings": 1,
    "ffn_norm": None,
    "attention_norm": None,
    "norm": None,
    "rope": None,
}

class MultiFileTensor(object):
    def __init__(self, files, parameter_name, dim):
        self.files = files
        self.name = parameter_name
        self.dim = dim
        self.length = self.files[0][parameter_name].shape[dim] * len(self.files)
        self.is_leading = dim == 0
    
    def restore(self):
        return self.transpose(0, self.dim)[0:self.length].transpose(0, self.dim)
    
    def transpose(self, a1, a2):
        if a1 != 0:
            raise ValueError(f"Can only swap leading dimension: expected (0,{self.dim}), got ({a1},{a2})")
        if a2 != self.dim:
            return self.restore().transpose(a1, a2)
        cp = copy(self)  # shallow
        cp.is_leading = self.dim == 0 or not cp.is_leading
        return cp
    
    @property
    def shape(self):
        if self.is_leading:
            return (self.length,)  # 🏏🫳
        zero_shape = self.files[0][self.name].shape
        assert zero_shape[self.dim] * len(self.files) == self.length
        zero_shape = zero_shape[:self.dim] + (self.length,) + zero_shape[self.dim + 1:]
        return zero_shape
    
    def __getitem__(self, sl):
        if (not isinstance(sl, slice)) or (not self.is_leading):
            raise ValueError("Can only slice on leading axis")
        a, b = sl.start, sl.stop
        chunk = (self.length // len(self.files))
        f0, f1 = a // chunk, (b - 1 + chunk) // chunk
        tensors = []
        for f in range(f0, f1):
            f_a, f_b = f * chunk, (f + 1) * chunk
            x, y = max(f_a, a), min(f_b, b)
            tensor = self.files[f][self.name]
            tensor = tensor.transpose(0, self.dim)
            tensor = tensor[x - f * chunk : y - f * chunk]
            tensors.append(tensor)
        return torch.cat(tensors, 0)


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
        short_name = parameter_name.split(".")[-2]
        dim = LLAMA_KEY_TO_DIM[short_name]
        if dim is None:
            return self.files[0][parameter_name]
        return MultiFileTensor(self.files, parameter_name, dim)
    
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
            # import subprocess
            # if local_rank == 0:
            #     print(subprocess.check_output("nvidia-smi").decode("utf-8"), flush=True)
            models = build_model(model_provider,
                        wrap_with_ddp,
                        virtual_pipeline_model_parallel_size,
                        args=model_args)
            # if local_rank == 0:
            #     print(subprocess.check_output("nvidia-smi").decode("utf-8"), flush=True)
            gc.collect()
            torch.cuda.empty_cache()  # frees ~a lot GB
            # if local_rank == 0:
            #     print(subprocess.check_output("nvidia-smi").decode("utf-8"), flush=True)
        
            return main_fn(models, dict(
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


def convert_weight_for_tp(weight, parallel_dimension, tp_rank=None, tp_size=None):
    if parallel_dimension is None:
        return weight
    if tp_rank is None:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
    if tp_size is None:
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
    chunk_size = weight.shape[parallel_dimension] // tp_size
    return weight.transpose(0, parallel_dimension)[
        chunk_size*tp_rank:chunk_size*(tp_rank+1)
    ].transpose(0, parallel_dimension)

def parallel_dimension_llama(key):
    short_name = key.split(".")[-2]
    if short_name == "tok_embeddings":
        return 0  # LLaMA uses a different sharding from tiny/nanopar
    dim = LLAMA_KEY_TO_DIM[short_name]
    return dim

def load_consolidated_llama_weights(models, path: Path, wrap_with_ddp: bool):
    state_dict = LlamaConsolidatedLoader(path)
    state_dict = {
        f"{'module.' if wrap_with_ddp else ''}wrapped." + k:
        convert_weight_for_tp(v, parallel_dimension_llama(k))
        for k, v in state_dict.items()}
    missing_keys, unexpected_keys = models[0].load_state_dict(state_dict, strict=False)
    # if int(os.environ["RANK"]) % torch.cuda.device_count():
    #     print("Self:", list(models[0].state_dict().keys())[::100])
    #     print("Other:", list(state_dict.keys())[::100])
    #     print("Missing keys:", missing_keys[::50])
    #     print("Unexpected keys:", unexpected_keys[::50])
    del state_dict
    torch.cuda.empty_cache()

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
