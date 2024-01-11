from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator, average_losses_across_data_parallel_group
from apex.transformer import parallel_state, tensor_parallel
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
# from apex.optimizers.fused_adam import FusedAdam

from loading_utils import main_with_model, load_consolidated_weights

from streaming import StreamingDataset
from pathlib import Path
from tqdm import tqdm
import torch
import fire


def loss_fn(pred, label, logprobs, beta=0.1):
    losses = tensor_parallel.vocab_parallel_cross_entropy(pred.contiguous(), label.contiguous())
    mask = label >= 0
    losses = (losses * mask).sum(0)
    losses = losses.view(-1, 2).transpose(0, 1)
    losses_chosen, losses_rejected = losses
    logprobs_chosen, logprobs_rejected = logprobs.T
    loss = -torch.nn.functional.logsigmoid(beta * ((losses_chosen - losses_rejected) - (logprobs_chosen - logprobs_rejected)))
    loss = loss.mean()
    return loss, {"loss": average_losses_across_data_parallel_group([loss])}
    
    
def train_step(batch, model):
    tokens, logprobs = batch["tokens"], batch["logprobs"]
    tokens = tokens.long().cuda()
    logprobs = logprobs.cuda()
    inputs = tokens.reshape(-1, tokens.shape[-1])[:, :-1].contiguous()
    inputs[inputs < 0] = 0
    out = model(inputs.contiguous()).contiguous()
    label = tokens[..., 1:]
    label = label.view(-1, label.shape[-1]).transpose(0, 1)
    return out, lambda pred: loss_fn(pred, label, logprobs)


# def inference_step(batch, model):
#     tokens, *kv_cache = batch
#     out = model(tokens[:, -1:], start_pos=0, kv_cache=kv_cache)
#     return (out, *kv_cache), (lambda pred: (0, {"pred": pred}))


def inference_step_dumb(batch, model):
    tokens = batch
    out = model(tokens, start_pos=0, kv_cache=None)
    return out, (lambda pred: (0, {"pred": pred}))


@main_with_model
def main(models, kwargs, data_dir=Path("data/logprob")):
    rank, data_parallel_size, llama_args, model_dir, forward_backward_func, use_sp = [
        kwargs[k] for k in
        ["rank", "data_parallel_size", "llama_args", "model_dir", "forward_backward_func", "use_sp"]]
    
    global_batch_size = 1
    micro_batch_size = 1
    
    setup_microbatch_calculator(
        rank=rank,
        rampup_batch_size=None,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    
    load_consolidated_weights(models, model_dir / "consolidated.00.pth")

    seq_len = llama_args.max_seq_len
    # batch = torch.randint(0, vocab_size, (global_batch_size // data_parallel_size, seq_len), device="cuda")
    batch_size = global_batch_size // data_parallel_size

    dataset = StreamingDataset(local="local/dpo", remote=data_dir, shuffle=True)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)

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
    
    for sample in tqdm(dl):
        torch.distributed.barrier()
        optimizer.zero_grad()
        loss = forward_backward_func(
            train_step,
            sample,
            models,
            forward_only=False,
            tensor_shape=(seq_len, micro_batch_size, llama_args.dim),
            dtype=torch.bfloat16,
            async_comm=True,
            sync_batch_comm=False,
            sequence_parallel_enabled=use_sp,
        )
        optimizer.step()


if __name__ == "__main__":
    fire.Fire(main)
