from loading_utils import main_with_model, load_consolidated_neox_weights, load_consolidated_llama_weights
from models.llama import llama_model_provider, ModelArgs
from models.neox import neox_model_provider, NeoXArgs
from apex.transformer import parallel_state, tensor_parallel

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer import tensor_parallel

import streaming
from streaming import StreamingDataset, MDSWriter
from tqdm.auto import tqdm, trange
import torch
import fire

from functools import partial
from pathlib import Path
import shutil
import json
import os

from llama import Transformer
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel, model_parallel_is_initialized)
from loading_utils import convert_weight_for_tp, LLAMA_KEY_TO_DIM


def cache_logprob(batch, model):
    inputs = batch.view(-1, batch.shape[-1])[:, :-1].contiguous()
    inputs[inputs < 0] = 0
    pred = model(inputs)
    return pred, (lambda pred: (0, {"pred": pred}))


def main_llama(model_dir: str, params_file: str, *, tokens=None):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    if not model_parallel_is_initialized():
        initialize_model_parallel(model_parallel_size)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    tokens = tokens.cuda()
    
    loaded_args = json.load(open(Path(model_dir) / params_file, "r"))
    model_args = ModelArgs(
        **{k: v for k, v in loaded_args.items()
            if k in ModelArgs.__dataclass_fields__},
        device="cuda"
    )
    
    llama = Transformer(model_args).half().cuda()
    llama.load_state_dict(
        {k: convert_weight_for_tp(v, LLAMA_KEY_TO_DIM[k.split(".")[-2]],
                                  tp_rank=local_rank, tp_size=model_parallel_size)
        for k, v in torch.load(str(Path(model_dir) / "consolidated.00.pth"), mmap=True).items()},
        strict=False)
    
    inputs = tokens.view(-1, tokens.shape[-1])[:, :-1].contiguous()
    inputs[inputs < 0] = 0
    pred = llama(inputs, start_pos=0)
    if local_rank == 0:
        return pred


@main_with_model(llama_model_provider, ModelArgs)
def main_nanopar(models, kwargs, *, tokens=None):
    rank, local_rank, data_parallel_size, model_args, model_dir, use_sp, wrap_with_ddp, forward_backward_func = [
        kwargs[k] for k in
        ["rank", "local_rank", "data_parallel_size", "model_args", "model_dir", "use_sp", "wrap_with_ddp", "forward_backward_func"]]


    global_batch_size = micro_batch_size = batch_size = 1
    hidden = model_args.dim
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    load_consolidated_llama_weights(models, model_dir / "consolidated.00.pth", wrap_with_ddp)

    result = forward_backward_func(
        cache_logprob,
        tokens.cuda(),
        models,
        forward_only=True,
        # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
        tensor_shape=(tokens.shape[1] - 1, tokens.shape[0], hidden),
        dtype=torch.bfloat16,
        async_comm=False,
        sync_batch_comm=True,
        sequence_parallel_enabled=use_sp,
    )
    pred = result[0].get("pred", None)
    if pred is None:
        return
    preds = torch.empty((pred.shape[0] * parallel_state.get_tensor_model_parallel_world_size(), *pred.shape[1:]),
                        dtype=pred.dtype, device=pred.device)
    torch.distributed.all_gather_into_tensor(preds, pred, parallel_state.get_tensor_model_parallel_group())
    pred = preds

    is_writer = parallel_state.is_pipeline_last_stage() and parallel_state.get_tensor_model_parallel_rank() == 0
    torch.distributed.barrier()
    if is_writer:
        return pred


def main(verification_dir = "verification", verify_text = "Hello world! This is a test.", **kwargs):
    tokenizer = SentencePieceProcessor("model_dir/llama-2-7b/tokenizer.model")
    tokens = tokenizer.Encode(verify_text)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    model_type = os.environ.get("MODEL_TYPE", "nanopar")
    print("Verifying text", repr(verify_text), "on model", model_type)
    if model_type == "nanopar":
        pred = partial(main_nanopar, tokens=tokens)(**kwargs)
    elif model_type == "llama":
        pred = partial(main_llama, tokens=tokens)(**kwargs)
    else:
        print("???")
        print("Define MODEL_TYPE={nanopar,llama} environment variable please")
        exit(1)
    os.makedirs(verification_dir, exist_ok=True)
    if pred is None:
        return
    print("Got prediction.")
    fn = f"{model_type}.th"
    for other_fn in os.listdir(verification_dir):
        if other_fn == fn:
            continue
        other_pred = torch.load(os.path.join(verification_dir, other_fn))
        diff = (other_pred.ravel() - pred.ravel()).sort().values
        print("Difference to", other_fn, diff[::len(diff) // 16])
        print("Quantiles self", pred.ravel().sort().values[::pred.numel() // 16])
        print("Quantiles other", other_pred.ravel().sort().values[::other_pred.numel() // 16])
    torch.save(pred, os.path.join(verification_dir, fn))


if __name__ == "__main__":
    fire.Fire(main)
