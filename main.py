from apex.transformer.pipeline_parallel import get_forward_backward_func, build_model
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator, average_losses_across_data_parallel_group
from apex.transformer import parallel_state, tensor_parallel
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.optimizers.fused_adam import FusedAdam

from model import ModelArgs, Transformer, PipelineStage

from sentencepiece import SentencePieceProcessor
import pandas as pd
from tqdm import trange
from pathlib import Path
import argparse
import random
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


def loss_fn(pred, label):
    loss = tensor_parallel.vocab_parallel_cross_entropy(pred, label)
    loss = loss.sum()
    print(loss.item() / label.numel())
    # loss = average_losses_across_data_parallel_group([loss])
    return loss, {}
    
    
def train_step(batch, model):
    out = model(batch[:, :-1], start_pos=0)
    label = batch[:, 1:].transpose(0, 1)
    return out, lambda pred: loss_fn(pred, label)


def inference_step(batch, model):
    tokens, *kv_cache = batch
    out = model(tokens[:, -1:], start_pos=0, kv_cache=kv_cache)
    return (out, *kv_cache), (lambda pred: (0, {"pred": pred}))


def inference_step_dumb(batch, model):
    tokens = batch
    out = model(tokens, start_pos=0, kv_cache=None)
    return out, (lambda pred: (0, {"pred": pred}))


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
    pipeline_model_parallel_size = 2
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
    use_sp = False
    set_random_seed(12)
    models = build_model(lambda args, **kwargs:
                         PipelineStage(Transformer(args,
                                                   dtype=torch.float16,
                                                   use_sp=use_sp)),
                        wrap_with_ddp,
                        virtual_pipeline_model_parallel_size,
                        args=llama_args)
    
    global_batch_size = 1
    micro_batch_size = 1
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    state_dict = torch.load(str(llama_path / "consolidated.00.pth"), mmap=True)
    state_dict = {"module.wrapped." + k: v for k, v in state_dict.items()}
    state_dict = {k: state_dict[k] for k in models[0].state_dict().keys()}
    state_dict = {k: convert_weight_for_tp(v, k) for k, v in state_dict.items()}
    models[0].load_state_dict(state_dict)
    del state_dict

    tokenizer = SentencePieceProcessor("llama-2-7b/tokenizer.model")
    pile = pd.read_parquet("pile.parquet")
    train_ds_size = 12
    tokens = []
    for i in range(train_ds_size):
        tokens.extend(tokenizer.Encode(pile.iloc[i, 0]))
    
    seq_len = 129  # llama_args.max_seq_len
    # batch = torch.randint(0, vocab_size, (global_batch_size // data_parallel_size, seq_len), device="cuda")
    batch_size = global_batch_size // data_parallel_size
    data = []
    for _ in range(batch_size):
        offset = random.randrange(0, len(tokens) - seq_len)
        data.append(tokens[offset:offset+seq_len])
    batch = torch.LongTensor(data).cuda()

    kv_cache = tuple()
    batch = batch[:, :-1]    
    src = parallel_state.get_pipeline_model_parallel_last_rank()
    group = parallel_state.get_embedding_group()
    for i in (trange if local_rank == 0 else range)(8, batch.shape[1] - 1):
        logits = forward_backward_func(
            inference_step_dumb,
            batch,
            models,
            forward_only=True,
            # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
            tensor_shape=(seq_len - 1, micro_batch_size, llama_args.dim),
            # T4 doesn't have bfloat16
            dtype=torch.float16,
            async_comm=True,
            sync_batch_comm=False,
            sequence_parallel_enabled=use_sp,
        )
        if parallel_state.is_pipeline_last_stage():
            logits = torch.cat([o["pred"] for o in logits], dim=1).float()
            logits = logits.transpose(0, 1).contiguous()
            logits = tensor_parallel.gather_from_tensor_model_parallel_region(
                logits
            )

            # vocab is padded to maximize performance
            logits = logits[:, :, :vocab_size]
            batch[:, i + 1] = logits[:, i].argmax()
            torch.distributed.broadcast(batch, src, group)
        elif parallel_state.is_pipeline_first_stage():
            torch.distributed.broadcast(batch, src, group)
    if local_rank == 0:
        for i, b in enumerate(batch):
            print(f"Decoded {i}:", tokenizer.Decode(batch.cpu().numpy().tolist()))
    return

    lr = 1e-4
    weight_decay = 1e-5
    optimizer = DistributedFusedAdam(
        models[0].parameters(),
        lr=lr,
        weight_decay=weight_decay,
        process_group=parallel_state.get_data_parallel_group(),
        dtype=torch.bfloat16,
        # TODO distribute over DP group?
        # distributed_process_group=torch.distributed.new_group(ranks=[torch.distributed.get_rank()]),
        # redundant_process_group=parallel_state.get_data_parallel_group(),
        store_params=False,
    )
    
    optimizer.zero_grad()
    loss = forward_backward_func(
        train_step,
        batch,
        models,
        forward_only=False,
        # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
        tensor_shape=(seq_len - 1, micro_batch_size, llama_args.dim),
        # T4 doesn't have bfloat16
        dtype=torch.float16,
        async_comm=True,
        sync_batch_comm=False,
        sequence_parallel_enabled=use_sp,
    )
    optimizer.step()
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
