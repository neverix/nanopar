from loading_utils import main_with_model, load_consolidated_weights

from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer import tensor_parallel

from streaming import StreamingDataset, MDSWriter
from tqdm.auto import tqdm
import torch
import fire

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


@torch.inference_mode()
@main_with_model
def main(models, kwargs, input_dir=Path("data/orig"), output_dir=Path("data/logprob")):
    rank, data_parallel_size, llama_args, model_dir, use_sp, wrap_with_ddp, forward_backward_func = [
        kwargs[k] for k in
        ["rank", "data_parallel_size", "llama_args", "model_dir", "use_sp", "wrap_with_ddp", "forward_backward_func"]]
    
    global_batch_size = 32
    micro_batch_size = 32
    batch_size = global_batch_size // data_parallel_size
    seq_len = 2049
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    load_consolidated_weights(models, model_dir / "consolidated.00.pth", wrap_with_ddp)
    
    # https://github.com/mosaicml/streaming/blob/release/v0.7.1/streaming/multimodal/convert/webvid/extract_webvid_videos.py
    # no special utility for processing StreamingDatasets
    dataset = StreamingDataset(local="local/logprob", remote=input_dir, shuffle=False)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    shutil.rmtree(output_dir, ignore_errors=True)
    with MDSWriter(out=str(output_dir), columns={"tokens": "ndarray", "logprobs": "ndarray"}, compression="zstd") as out:
        for batch in tqdm(dl):
            tokens = batch["tokens"]
            tokens = tokens[..., -seq_len:].cuda()
            logprobs = forward_backward_func(
                cache_logprob,
                tokens,
                models,
                forward_only=True,
                # IO shape? I'm not sure if putting Seq_Len first is used for parallelism
                tensor_shape=(seq_len - 1, micro_batch_size, llama_args.dim),
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
        


if __name__ == "__main__":
    fire.Fire(main)
