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
import os


def loss_fn(pred, batch):
    targets = batch.view(-1, batch.shape[-1])[:, 1:].transpose(0, 1).contiguous()
    torch.distributed.barrier()  # target must be on GPU?
    losses = tensor_parallel.vocab_parallel_cross_entropy(pred, targets)
    mask = targets >= 0
    losses = losses * mask
    losses = losses.sum(0).reshape(batch.shape[:-1])
    return losses


def cache_logprob(batch, model):
    inputs = batch.view(-1, batch.shape[-1])[:, :-1].contiguous()
    inputs[inputs < 0] = 0
    pred = model(inputs)
    return pred, (lambda pred: (0, {"logprobs": loss_fn(pred, batch)}))


def inference_step_dumb(batch, model):
    tokens = batch
    tokens[tokens < 0] = 0
    out = model(tokens)
    return out, (lambda pred: (0, {"pred": pred}))


MODEL_TYPE = os.environ.get("MODEL_TYPE") or "neox"


@torch.inference_mode()
@main_with_model(
    *((llama_model_provider, ModelArgs) if MODEL_TYPE == "llama" else (neox_model_provider, NeoXArgs))
)
def main(models, kwargs, input_dir=Path("data/orig"), output_dir=Path("data/logprob"),
         test_inference: bool = False,
         global_batch_size = 32,
         micro_batch_size = 32,
         seq_len = 2049,
         compare_to_fb_llama: bool = False,
         preserve_output: bool = True
         ):
    if preserve_output and os.path.exists(output_dir):
        print("Output exists")
        return

    rank, local_rank, data_parallel_size, model_args, model_dir, use_sp, wrap_with_ddp, forward_backward_func, run_context = [
        kwargs[k] for k in
        ["rank", "local_rank", "data_parallel_size", "model_args", "model_dir", "use_sp", "wrap_with_ddp",
         "forward_backward_func", "run_context"]]

    batch_size = global_batch_size // data_parallel_size
    hidden = model_args.hidden_size if MODEL_TYPE == "neox" else model_args.dim
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    if MODEL_TYPE == "llama":
        load_consolidated_llama_weights(models, model_dir / "consolidated.00.pth", wrap_with_ddp)
    else:
        load_consolidated_neox_weights(models, model_args, model_dir / "pytorch_model.bin", wrap_with_ddp)
    
    compare_to_fb_llama = compare_to_fb_llama and parallel_state.is_pipeline_last_stage()
    if compare_to_fb_llama:
        assert MODEL_TYPE == "llama"
        from llama import Transformer
        from fairscale.nn.model_parallel.initialize import initialize_model_parallel
        from loading_utils import convert_weight_for_tp, LLAMA_KEY_TO_DIM
        initialize_model_parallel(parallel_state.get_tensor_model_parallel_world_size())
        llama = Transformer(model_args).half().cuda()
        llama.load_state_dict(
            {k: convert_weight_for_tp(v, LLAMA_KEY_TO_DIM[k.split(".")[-2]])
            for k, v in torch.load(str(model_dir / "consolidated.00.pth"), mmap=True).items()},
            strict=False)
    
    is_writer = parallel_state.is_pipeline_last_stage() and parallel_state.get_tensor_model_parallel_rank() == 0
    # https://github.com/mosaicml/streaming/blob/release/v0.7.1/streaming/multimodal/convert/webvid/extract_webvid_videos.py
    # no special utility for processing StreamingDatasets
    dataset = StreamingDataset(local=input_dir, shuffle=False, replication=parallel_state.get_tensor_model_parallel_world_size())
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    if is_writer:
        print(len(dataset), len(dl), batch_size)
    streaming.base.util.clean_stale_shared_memory()
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    out = None
    if is_writer:
        out = MDSWriter(out=str(output_dir), columns={"tokens": "ndarray", "logprobs": "ndarray"}, compression="zstd")
    torch.distributed.barrier()
    for batch in (tqdm if is_writer else lambda x: x)(dl):
        tokens = batch["tokens"]
        tokens = tokens[..., -seq_len:].cuda()
        if test_inference:
            if MODEL_TYPE == "neox":
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                decode = lambda x: [tokenizer.decode([u for u in y if u >= 0]) for y in x]
                tokens[..., 0] = tokenizer.encode("The")[0]  # BOS
            else:
                tokenizer = SentencePieceProcessor(str(model_dir / "tokenizer.model"))
                tokens[..., 0] = tokenizer.Encode("The")[0]
                decode = lambda x: tokenizer.Decode(x)
            inference(
                local_rank=local_rank,
                batch=tokens,
                models=models,
                forward_backward_func=forward_backward_func,
                micro_batch_size=micro_batch_size,
                decode=decode,
                use_sp=use_sp,
                vocab_size=model_args.vocab_size,
                hidden_dim=hidden
            )
            return
        
        with run_context():
            result = forward_backward_func(
                cache_logprob,
                tokens,
                models,
                forward_only=True,
                # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
                tensor_shape=(seq_len - 1, micro_batch_size, hidden),
                dtype=torch.bfloat16,
                async_comm=False,
                sync_batch_comm=True,
                sequence_parallel_enabled=use_sp,
            )
        
        if compare_to_fb_llama:
            # hack
            inputs = tokens.view(-1, tokens.shape[-1])[:, :-1].contiguous()
            inputs[inputs < 0] = 0
            pred = llama(inputs, start_pos=0)
            targets = tokens.view(-1, tokens.shape[-1])[:, 1:].contiguous().long()
            llama_logprobs = torch.nn.functional.cross_entropy(
                pred.transpose(1, -1), targets, reduction="none").sum(1)
        
        if is_writer:
            logprobs = result[0]["logprobs"]
            if compare_to_fb_llama:
                llama_logprobs = llama_logprobs.view(logprobs.shape)
                logprob_diff = logprobs - llama_logprobs
                print("Logprob difference:", logprob_diff.pow(2).mean(), flush=True)
            # singular, but they are a size-2 batch of sequences.
            for token, logprob in zip(tokens, logprobs):
                out.write({
                    "tokens": token.detach().cpu().numpy(),
                    "logprobs": logprob.detach().cpu().numpy(),
                })
        torch.distributed.barrier()
    if is_writer:
        out.finish()


def inference(models, forward_backward_func, batch, decode, vocab_size, hidden_dim,
              local_rank, micro_batch_size, use_sp=False):
    batch = batch.reshape(-1, batch.shape[-1])
    seq_len = batch.shape[1]
    
    src = parallel_state.get_pipeline_model_parallel_last_rank()
    group = parallel_state.get_embedding_group()
    for i in (trange if local_rank == 0 else range)(0, batch.shape[1] - 1):
        logits = forward_backward_func(
            inference_step_dumb,
            batch,
            models,
            forward_only=True,
            # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
            tensor_shape=(seq_len - 1, micro_batch_size, hidden_dim),
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
            logits = torch.softmax(logits[:, i], dim=-1)
            logits[torch.isinf(logits)] = 0
            logits[torch.isnan(logits)] = 0
            batch[:, i + 1] = torch.multinomial(logits, 1).reshape(logits.shape[0])
            torch.distributed.broadcast(batch, src, group)
        elif parallel_state.is_pipeline_first_stage():
            torch.distributed.broadcast(batch, src, group)
    if local_rank == 0:
        for i, b in enumerate(batch):
            print(f"Decoded {i}:", decode(batch.cpu().numpy().tolist()))
    return


if __name__ == "__main__":
    fire.Fire(main)
