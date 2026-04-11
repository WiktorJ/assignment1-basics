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
    for model in models:
        name = model["name"]
        c = model["config"]
        print(f"\n\n{'#' * 44}")
        print(f"  {name}")
        print(f"{'#' * 44}")
        print_results("Parameter Counts", count_params(c))
        print_results("FLOPs Counts", count_flops(c))
