from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator, average_losses_across_data_parallel_group
from apex.transformer import parallel_state, tensor_parallel
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
from apex.contrib.optimizers.fused_adam import FusedAdam
from models.llama import llama_model_provider, ModelArgs
from models.neox import neox_model_provider, NeoXArgs
# from apex.optimizers.fused_adam import FusedAdam

from loading_utils import main_with_model, load_consolidated_llama_weights, load_consolidated_neox_weights

from streaming import StreamingDataset
from pathlib import Path
from tqdm import tqdm
import wandb
import torch
import fire
import os


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


MODEL_TYPE = os.environ.get("MODEL_TYPE") or "neox"


@main_with_model(
    *((llama_model_provider, ModelArgs) if MODEL_TYPE == "llama" else (neox_model_provider, NeoXArgs))
)
def main(models, kwargs, data_dir=Path("data/logprob"), grad_acc: int = 8):
    rank, data_parallel_size, model_dir, forward_backward_func, use_sp, wrap_with_ddp, model_args = [
        kwargs[k] for k in
        ["rank", "data_parallel_size", "model_dir", "forward_backward_func", "use_sp", "wrap_with_ddp", "model_args"]]
    
    global_batch_size = 1
    micro_batch_size = 1
    
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

    # batch = torch.randint(0, vocab_size, (global_batch_size // data_parallel_size, seq_len), device="cuda")
    batch_size = global_batch_size // data_parallel_size

    dataset = StreamingDataset(local="local/dpo", remote=data_dir, shuffle=True)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)

    lr = 1e-5
    weight_decay = 1e-6
    distributed_adam = False
    if distributed_adam:
        optimizer = DistributedFusedAdam(
            models[0].parameters(),
            lr=lr,
            weight_decay=weight_decay,
            process_group=parallel_state.get_data_parallel_group(),
            dtype=torch.float16,
            # TODO distribute over DP group?
            # distributed_process_group=torch.distributed.new_group(ranks=[torch.distributed.get_rank()]),
            # redundant_process_group=parallel_state.get_data_parallel_group(),
            store_params=False,
        )
        optimizer.init_param_buffer()
    else:
        optimizer = FusedAdam(
            models[0].parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    wandb.init(
        mode="offline",
        project="nanopar"
    )

    total_loss = 0
    for i, sample in enumerate(bar := tqdm(dl)):
        torch.distributed.barrier()
        loss = forward_backward_func(
            train_step,
            sample,
            models,
            forward_only=False,
            tensor_shape=(model_args.max_seq_len, micro_batch_size, model_args.dim),
            dtype=torch.bfloat16,
            async_comm=True,
            sync_batch_comm=False,
            sequence_parallel_enabled=use_sp,
        )[0]["loss"]
        total_loss += loss.item() / grad_acc
        if i % grad_acc == grad_acc - 1:
            torch.nn.utils.clip_grad_norm_(models[0].parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            wandb.log(dict(loss=total_loss))
            bar.set_postfix(loss=total_loss)
            total_loss = 0


if __name__ == "__main__":
    fire.Fire(main)
