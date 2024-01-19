from loading_utils import main_with_model, load_consolidated_neox_weights
from models.llama import llama_model_provider, ModelArgs
from models.neox import neox_model_provider, NeoXArgs
from apex.transformer import parallel_state, tensor_parallel
from transformers import AutoTokenizer

from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer import tensor_parallel

from streaming import StreamingDataset, MDSWriter
from tqdm.auto import tqdm, trange
import torch
import fire

from functools import partial
from pathlib import Path
import shutil


def loss_fn(pred, batch):
    targets = batch.view(-1, batch.shape[-1])[:, 1:].transpose(0, 1).contiguous()
    torch.distributed.barrier()  # target must be on GPU?
    losses = tensor_parallel.vocab_parallel_cross_entropy(pred, targets)
    mask = targets >= 0
    losses = losses * mask
    losses = losses.sum(0).reshape(batch.shape[:-1])
    return losses


def cache_logprob(batch, model):
    inputs = batch.view(-1, batch.shape[-1])[:, :-1].clone()
    inputs[inputs < 0] = 0
    pred = model(inputs)
    return pred, (lambda pred: (0, {"logprobs": loss_fn(pred, batch)}))


def inference_step_dumb(batch, model):
    tokens = batch
    tokens[tokens < 0] = 0
    out = model(tokens)
    return out, (lambda pred: (0, {"pred": pred}))


@torch.inference_mode()
@main_with_model(
    # llama_model_provider, ModelArgs
    neox_model_provider, NeoXArgs
)
def main(models, kwargs, input_dir=Path("data/orig"), output_dir=Path("data/logprob"),
         test_inference: bool = False):
    rank, local_rank, data_parallel_size, model_args, model_dir, use_sp, wrap_with_ddp, forward_backward_func = [
        kwargs[k] for k in
        ["rank", "local_rank", "data_parallel_size", "model_args", "model_dir", "use_sp", "wrap_with_ddp", "forward_backward_func"]]
    
    global_batch_size = 32
    micro_batch_size = 32
    batch_size = global_batch_size // data_parallel_size
    seq_len = 129  # 2049
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    # load_consolidated_llama_weights(models, model_dir / "consolidated.00.pth", wrap_with_ddp)
    load_consolidated_neox_weights(models, model_args, model_dir / "pytorch_model.bin", wrap_with_ddp)
    
    # https://github.com/mosaicml/streaming/blob/release/v0.7.1/streaming/multimodal/convert/webvid/extract_webvid_videos.py
    # no special utility for processing StreamingDatasets
    dataset = StreamingDataset(local="local/logprob", remote=input_dir, shuffle=False)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    shutil.rmtree(output_dir, ignore_errors=True)
    with MDSWriter(out=str(output_dir), columns={"tokens": "ndarray", "logprobs": "ndarray"}, compression="zstd") as out:
        for batch in tqdm(dl):
            tokens = batch["tokens"]
            tokens = tokens[..., -seq_len:].cuda()
            if test_inference:
                # TODO
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                inference(
                    local_rank=local_rank,
                    batch=tokens,
                    models=models,
                    forward_backward_func=forward_backward_func,
                    micro_batch_size=micro_batch_size,
                    decode=lambda x: [tokenizer.decode(y) for y in x],
                    use_sp=use_sp,
                    vocab_size=model_args.vocab_size,
                    hidden_dim=model_args.hidden_size   
                )
                return
            logprobs = forward_backward_func(
                cache_logprob,
                tokens,
                models,
                forward_only=True,
                # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
                tensor_shape=(seq_len - 1, micro_batch_size, model_args.hidden_size),
                # T4 doesn't have bfloat16
                dtype=torch.float16,
                async_comm=True,
                sync_batch_comm=False,
                sequence_parallel_enabled=use_sp,
            )[0]["logprobs"]
            # singular, but they are a size-2 batch of sequences.
            for token, logprob in zip(tokens, logprobs):
                out.write({
                    "tokens": token.detach().cpu().numpy(),
                    "logprobs": logprob.detach().cpu().numpy(),
                })


def inference(models, forward_backward_func, batch, decode, vocab_size, hidden_dim,
              local_rank, micro_batch_size, use_sp=False):
    batch = batch.reshape(-1, batch.shape[-1])
    seq_len = batch.shape[1]
    
    src = parallel_state.get_pipeline_model_parallel_last_rank()
    group = parallel_state.get_embedding_group()
    for i in (trange if local_rank == 0 else range)(8, batch.shape[1] - 1):
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
            batch[:, i + 1] = logits[:, i].argmax(dim=-1)
            torch.distributed.broadcast(batch, src, group)
        elif parallel_state.is_pipeline_first_stage():
            torch.distributed.broadcast(batch, src, group)
    if local_rank == 0:
        for i, b in enumerate(batch):
            print(f"Decoded {i}:", decode(batch.cpu().numpy().tolist()))
    return


if __name__ == "__main__":
    fire.Fire(main)
