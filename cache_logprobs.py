from loading_utils import main_with_model, load_consolidated_weights

from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer import parallel_state, tensor_parallel

from streaming import StreamingDataset
import torch
import fire

from pathlib import Path


def loss_fn(pred, batch):
    targets = batch.view(-1, batch.shape[-1])[:, 1:].transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(pred, targets)
    losses = losses.sum(0).reshape(batch.shape[:-1])
    return losses


def cache_logprob(batch, model):
    inputs = batch.view(-1, batch.shape[-1])[:, :-1].clone()
    inputs[inputs == -100] = 0
    pred = model(inputs)
    return pred, (lambda pred: (0, {"losses": loss_fn(pred, batch)}))


@main_with_model
def main(models, kwargs, data_dir=Path("data")):
    rank, data_parallel_size, llama_args, model_dir, use_sp, forward_backward_func = [
        kwargs[k] for k in
        ["rank", "data_parallel_size", "llama_args", "model_dir", "use_sp", "forward_backward_func"]]
    
    global_batch_size = 1
    micro_batch_size = 1
    batch_size = global_batch_size // data_parallel_size
    seq_len = 2049
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    load_consolidated_weights(models, model_dir / "consolidated.00.pth")
    
    dataset = StreamingDataset(local="local", remote=data_dir, shuffle=False)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    batch = next(iter(dl))["tokens"]
    batch = batch[..., :seq_len]
    losses = forward_backward_func(
        cache_logprob,
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


if __name__ == "__main__":
    fire.Fire(main)
