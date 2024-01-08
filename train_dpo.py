from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer import parallel_state, tensor_parallel
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
# from apex.optimizers.fused_adam import FusedAdam

from loading_utils import main_with_model, load_consolidated_weights

from streaming import StreamingDataset
from tqdm import trange
from pathlib import Path
import torch
import fire


def loss_fn(pred, label):
    losses = tensor_parallel.vocab_parallel_cross_entropy(pred, label)
    losses_chosen, losses_rejected = losses.chunk(losses, 2)
    loss = losses_chosen
    loss = loss.sum()
    # loss = average_losses_across_data_parallel_group([loss])
    return loss, {}
    
    
def train_step(batch, model):
    out = model(
        batch.transpose(0, 1).reshape(-1, batch.shape[-1])[:, :-1], start_pos=0)
    label = batch[:, 1:].transpose(0, 1)
    return out, lambda pred: loss_fn(pred, label)


# def inference_step(batch, model):
#     tokens, *kv_cache = batch
#     out = model(tokens[:, -1:], start_pos=0, kv_cache=kv_cache)
#     return (out, *kv_cache), (lambda pred: (0, {"pred": pred}))


def inference_step_dumb(batch, model):
    tokens = batch
    out = model(tokens, start_pos=0, kv_cache=None)
    return out, (lambda pred: (0, {"pred": pred}))


@main_with_model
def main(models, kwargs, data_dir=Path("data")):
    rank, local_rank, data_parallel_size, llama_args, model_dir = [
        kwargs[k] for k in
        ["rank", "local_rank", "data_parallel_size", "llama_args", "model_dir"]]
    
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

    seq_len = 128  # llama_args.max_seq_len
    # batch = torch.randint(0, vocab_size, (global_batch_size // data_parallel_size, seq_len), device="cuda")
    batch_size = global_batch_size // data_parallel_size

    dataset = StreamingDataset(local="local", remote=data_dir, shuffle=True)
    dl = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    if test_inference:
        # raise NotImplementedError()
        sample = next(iter(dl))
        batch = sample["tokens"].long().cuda()
        batch = batch.reshape(-1, batch.shape[-1])
        # batch = batch[:, :-1]
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
                batch[:, i + 1] = logits[:, i].argmax(-1)
                torch.distributed.broadcast(batch, src, group)
            elif parallel_state.is_pipeline_first_stage():
                torch.distributed.broadcast(batch, src, group)
            if local_rank == 0:
                for j, b in enumerate(batch):
                    print(f"Step {i}, decoded {j}:", tokenizer.Decode(b[:i].cpu().numpy().tolist()))
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
    
    for sample in dl:
        batch = sample["tokens"].long().cuda()
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
    fire.Fire(main)
