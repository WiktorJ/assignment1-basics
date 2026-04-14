from dataclasses import dataclass
import torch
import numpy as np
import einx
from numpy.lib.stride_tricks import sliding_window_view
import os
import typing
import time
from dataclasses import field
import mlflow
import tyro
from cs336_basics import model as model_lib
from cs336_basics import optimizer as optimizer_lib


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0


@dataclass
class OptimizerConfig:
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-1
    eps: float = 1e-8


@dataclass
class LRConfig:
    min_lr: float = 1e-5
    max_lr: float = 1e-3
    warmup_iters: int = 200
    cosine_cycle_iters: int = 1000


@dataclass
class TrainingConfig:
    train_input_path: str | os.PathLike
    eval_input_path: str | os.PathLike
    output_path: str | os.PathLike
    resume_checkpoint_path: str | os.PathLike | None = None
    save_interval: int = 100
    log_interval: int = 10

    batch_size: int = 16
    max_steps: int = 1000
    device: str = "cpu"

    max_l2_norm = 1.0

    gradient_accumulation_steps: int = 1

    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_config: LRConfig = field(default_factory=LRConfig)


@torch.compile
def cross_entropy_loss(logits, targets, dim=-1):
    logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]
    log_exp_sum = torch.sum(torch.exp(logits), dim=dim, keepdim=True)
    logits = torch.gather(logits, dim=dim, index=targets.unsqueeze(dim))
    return torch.mean(-logits + torch.log(log_exp_sum))


def calculate_perplexity(losses):
    return np.exp(losses.mean())


def lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    elif warmup_iters <= it < cosine_cycle_iters:
        return (max_lr - min_lr) * (
            1 + np.cos(np.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) / 2 + min_lr
    else:
        return min_lr


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    flat_grads = torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None])
    grad_norm = torch.sqrt(einx.sum("... ->", flat_grads**2))
    if grad_norm < max_l2_norm:
        return
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(max_l2_norm / (grad_norm + eps))


def get_batch(dataset, batch_size, context_length, device):
    indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    in_indices = torch.LongTensor(sliding_window_view(dataset, window_shape=context_length)[indices]).to(device)
    out_indices = torch.LongTensor(sliding_window_view(dataset, window_shape=context_length)[indices + 1]).to(device)
    return in_indices, out_indices


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]


def eval(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    step: int,
    eval_iters: int = 1,
):
    model.eval()
    x, y = get_batch(data, batch_size, context_length, device)
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            logits = model(x)
            losses.append(cross_entropy_loss(logits, y))
    val_loss = np.mean(losses)
    perplexity = calculate_perplexity(losses)
    model.train()
    return {
        "step": step,
        "val_loss": val_loss,
        "perplexity": perplexity,
    }


def train(config: TrainingConfig):
    train_data = np.memmap(config.train_input_path, mode="r", dtype=np.uint16)
    eval_data = np.memmap(config.eval_input_path, mode="r", dtype=np.uint16)

    run_name = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(config.output_path, run_name)
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    log_file = os.path.join(output_path, "log.txt")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # str_dtype = str(dtype).split(".")[-1]

    mlflow.set_experiment("cs336_training")
    mlflow_run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            **config.model_config.__dict__,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_l2_norm": config.max_l2_norm,
            "max_lr": config.lr_config.max_lr,
            "min_lr": config.lr_config.min_lr,
            "warmup_iters": config.lr_config.warmup_iters,
            "cosine_cycle_iters": config.lr_config.cosine_cycle_iters,
            "weight_decay": config.optimizer_config.weight_decay,
            "betas": str(config.optimizer_config.betas),
        }
    )

    model = model_lib.Transformer(**config.model_config.__dict__, device=config.device, dtype=dtype)
    model.to(config.device)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optimizer_lib.AdamW(model.parameters(), **config.optimizer_config.__dict__)
    if config.resume_checkpoint_path:
        start_step = load_checkpoint(config.resume_checkpoint_path, model, optimizer) + 1
    else:
        start_step = 0

    if config.device != "mps":
        model = torch.compile(model)
        torch.set_float32_matmul_precision("high")
    elif config.device == "mps":
        model = torch.compile(model, backend="aot_eager")

    lr_schedule = lambda step: lr_cosine_schedule(
        step,
        max_lr=config.lr_config.max_lr,
        min_lr=config.lr_config.min_lr,
        warmup_iters=config.lr_config.warmup_iters,
        cosine_cycle_iters=config.lr_config.cosine_cycle_iters,
    )

    x, y = get_batch(train_data, config.batch_size, config.model_config.context_length, config.device)
    losses = []
    for i in range(start_step, config.max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.gradient_accumulation_steps):
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            loss.backward()
            losses.append(loss.item())
            x, y = get_batch(train_data, config.batch_size, config.model_config.context_length, config.device)

        eval_metrics = eval(
            model,
            eval_data,
            batch_size=config.batch_size,
            context_length=config.model_config.context_length,
            device=config.device,
            step=i,
            eval_iters=1,
        )

        norm = gradient_clipping(model.parameters(), config.max_l2_norm)
        lr = lr_schedule(i)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if config.device == "cuda":
            torch.cuda.synchronize()
        elif config.device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (
            config.model_config.context_length * config.batch_size * config.gradient_accumulation_steps / dt
        )
        train_metrics = {
            "step": i,
            "train_loss": np.mean(losses),
            "train_perplexity": calculate_perplexity(losses),
            "norm": norm,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
        }

        all_metrics = {**train_metrics, **eval_metrics}
        mlflow.log_metrics(
            {k: v for k, v in all_metrics.items() if isinstance(v, (int, float)) and k != "step"},
            step=i,
        )

        if i % config.log_interval == 0:
            print(
                f"step {i:6d} | "
                f"train_loss {train_metrics['train_loss']:.4f} | "
                f"val_loss {eval_metrics['val_loss']:.4f} | "
                f"ppl {eval_metrics['perplexity']:.2f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tokens_per_sec:.0f}"
            )

        if i % config.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{i}.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(model, optimizer, i, ckpt_path)

    mlflow.end_run()


def main():
    config = tyro.cli(TrainingConfig)
    train(config)


if __name__ == "__main__":
    main()
