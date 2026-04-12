models = [
    {
        "name": "GPT-2 Small",
        "config": {
            "vocab_size": 50257,
            "seq_len": 1024,
            "num_layers": 12,
            "d_model": 768,
            "num_heads": 12,
            "d_ff": 3072,
            "bs": 1,
        },
    },
    {
        "name": "GPT-2 Medium",
        "config": {
            "vocab_size": 50257,
            "seq_len": 1024,
            "num_layers": 24,
            "d_model": 1024,
            "num_heads": 16,
            "d_ff": 4096,
            "bs": 1,
        },
    },
    {
        "name": "GPT-2 Large",
        "config": {
            "vocab_size": 50257,
            "seq_len": 1024,
            "num_layers": 36,
            "d_model": 1280,
            "num_heads": 20,
            "d_ff": 5120,
            "bs": 1,
        },
    },
    {
        "name": "GPT-2 XL",
        "config": {
            "vocab_size": 50257,
            "seq_len": 1024,
            "num_layers": 48,
            "d_model": 1600,
            "num_heads": 25,
            "d_ff": 6400,
            "bs": 1,
        },
    },
    {
        "name": "GPT-2 XL Long Context",
        "config": {
            "vocab_size": 50257,
            "seq_len": 16385,
            "num_layers": 48,
            "d_model": 1600,
            "num_heads": 25,
            "d_ff": 6400,
            "bs": 1,
        },
    },
]

models2 = [
    {
        "name": "GPT-2 XL",
        "config": {
            "vocab_size": 50257,
            "seq_len": 1024,
            "num_layers": 48,
            "d_model": 1600,
            "num_heads": 25,
            "d_ff": 6400,
            "bs": 1,
        },
    },
]


def count_params(c: dict) -> dict:
    # counts parameters according to the formula:
    # d_model + (2 * d_model * vocab_size) + (num_layers * ((2 * d_model) + (4 * d_model**2) + (2 * d_ff * d_model)))
    p_embedding = c["d_model"] * c["vocab_size"]
    p_layer = (2 * c["d_model"]) + (4 * c["d_model"] ** 2) + (2 * c["d_ff"] * c["d_model"])
    p_layers = c["num_layers"] * p_layer
    p_norm = c["d_model"]
    p_lm_head = c["d_model"] * c["vocab_size"]
    p_total = p_embedding + p_layers + p_norm + p_lm_head
    p_total_gb = p_total * 4 / 1e9
    return {
        "embedding": f"{p_embedding:,}",
        "layers": f"{p_layers:,}",
        "norm": f"{p_norm:,}",
        "lm_head": f"{p_lm_head:,}",
        "total": f"{p_total:,}",
        "total (fp32 GB)": f"{p_total_gb:.2f} GB",
        "total_int": p_total,
    }


def count_peak_memory_adamw(c: dict) -> dict:
    fp_precision = 4
    # parameters
    parameters = count_params(c)["total_int"]
    # optimizer state
    optimizer_state = 2 * parameters

    # gradients
    gradients = parameters

    # activations
    # - transfomer
    # -- attention block
    rms_norm = 2 * c["d_model"] * c["seq_len"] * c["bs"]
    qvk = 3 * c["d_model"] * c["seq_len"] * c["bs"]
    qk = c["num_heads"] * c["seq_len"] ** 2 * c["bs"]
    qk_softmax = qk
    val_sum = c["d_model"] * c["seq_len"] * c["bs"]
    output_proj = c["d_model"] * c["seq_len"] * c["bs"]
    attention_total = rms_norm + qvk + qk + qk_softmax + val_sum + output_proj
    # -- ff silu
    ff1 = c["d_ff"] * c["seq_len"] * c["bs"]
    ff_silu = c["d_ff"] * c["seq_len"] * c["bs"]
    ff2 = c["d_model"] * c["seq_len"] * c["bs"]
    ff_total = ff1 + ff_silu + ff2

    attention_total_block_total = c["num_layers"] * (attention_total + ff_total)
    # - final rms norm
    rms_norm = c["d_model"] * c["seq_len"] * c["bs"]
    # - lm_head
    lm_head = c["vocab_size"] * c["seq_len"] * c["bs"]
    # - cross entropy
    final_softmax = c["vocab_size"] * c["seq_len"] * c["bs"]
    ce = c["seq_len"] * c["bs"]
    total_ce = final_softmax + ce

    activations = attention_total_block_total + rms_norm + lm_head + total_ce

    # total
    total_count = parameters + optimizer_state + gradients + activations
    peak_memory_gb = total_count * fp_precision / 1e9

    return {
        "parameters": f"{parameters:,}",
        "parameters GB": f"{parameters * fp_precision / 1e9:.2f} GB",
        "optimizer state": f"{optimizer_state:,}",
        "optimizer state GB": f"{optimizer_state * fp_precision / 1e9:.2f} GB",
        "gradients": f"{gradients:,}",
        "gradients GB": f"{gradients * fp_precision / 1e9:.2f} GB",
        "activations": f"{activations:,}",
        "activations GB": f"{activations * fp_precision / 1e9:.2f} GB",
        "total GB": f"{peak_memory_gb:.2f} GB",
    }


def count_flops(c: dict) -> dict:
    # counts flops according to the formula:
    # (num_layers * ((8 * bs * seq_len * d_model) + (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model)
    #   + (4 * bs * seq_len * d_model * d_ff))) + (3 * bs * seq_len * d_model) + (2 * bs * seq_len * d_model * vocab_size)
    f_att_norm = 8 * c["bs"] * c["seq_len"] * c["d_model"]
    f_att_qkv = 6 * c["bs"] * c["seq_len"] * c["d_model"] ** 2
    f_att_dot = 4 * c["bs"] * c["seq_len"] ** 2 * c["d_model"]
    f_att_out = 2 * c["bs"] * c["seq_len"] * c["d_model"] ** 2
    f_att = f_att_norm + f_att_qkv + f_att_dot + f_att_out
    f_ff = 4 * c["bs"] * c["seq_len"] * c["d_model"] * c["d_ff"]
    f_block = f_att + f_ff
    f_att_total = c["num_layers"] * f_att
    f_ff_total = c["num_layers"] * f_ff
    f_layers = f_att_total + f_ff_total
    f_norm = 3 * c["bs"] * c["seq_len"] * c["d_model"]
    f_lm_head = 2 * c["bs"] * c["seq_len"] * c["d_model"] * c["vocab_size"]
    f_total = f_layers + f_norm + f_lm_head
    f_total_tflops = f_total / 1e12
    return {
        "attention": f"{f_att:,}",
        "feed forward": f"{f_ff:,}",
        "attention total": f"{f_att_total:,}",
        "feed forward total": f"{f_ff_total:,}",
        "block": f"{f_block:,}",
        "layers": f"{f_layers:,}",
        "total": f"{f_total:,}",
        "total (TFLOPs)": f"{f_total_tflops:.2f} TFLOPs",
        "total + backwards (TFLOPs)": f"{f_total_tflops * 3:.2f} TFLOPs",
    }


def print_results(title: str, data: dict) -> None:
    width = 44
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    for key, value in data.items():
        print(f"  {key:<20} {value:>20}")
    print(f"{'=' * width}")


if __name__ == "__main__":
    # for model in models:
    #     name = model["name"]
    #     c = model["config"]
    #     print(f"\n\n{'#' * 44}")
    #     print(f"  {name}")
    #     print(f"{'#' * 44}")
    # print_results("Parameter Counts", count_params(c))
    # print_results("FLOPs Counts", count_flops(c))
    for model in models2:
        for batch_size in (1024,):
            c = model["config"]
            c["bs"] = batch_size
            print(f"\n\n{'#' * 44}")
            print(f"  {model['name']} (bs={batch_size})")
            print_results("Memory Counts", count_peak_memory_adamw(c))
            print_results("FLOPs Counts", count_flops(c))

    total_tflops = 10774 * 400_000
    max_nvidia_a100_tflops = 19.5
    mfu = 0.5
    tflops = max_nvidia_a100_tflops * mfu
    seconds = total_tflops / tflops
    days = seconds / (60 * 60 * 24)
    print(f"\n\n{'#' * 44}")
    print(f"  {days:.2f} days")
