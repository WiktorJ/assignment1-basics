vocab_size = 50257
seq_len = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400
bs = 1


def count_params():
    # counts paramsters according to the formula:
    # d_model + (2 * d_model * vocab_size) + (num_layers * ((2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model)))
    return (
        d_model + (2 * d_model * vocab_size) + (num_layers * ((2 * d_model) + (4 * d_model**2) + (3 * d_ff * d_model)))
    )


def count_flops():
    # counts flops according to the formula:
    # (num_layers * ((8 * bs * seq_len * d_model)  + (8 * bs * seq_len * d_model^2) + (4 * bs * seq_len^2 * d_model)
    #   + (6 * bs * seq_len * d_model * d_ff))) + (3 * bs * seq_len * d_model) + (2 * bs * seq_len * d_model * vocab_size)
    return (
        (
            num_layers
            * (
                (8 * bs * seq_len * d_model)
                + (8 * bs * seq_len * d_model**2)
                + (4 * bs * seq_len**2 * d_model)
                + (6 * bs * seq_len * d_model * d_ff)
            )
        )
        + (3 * bs * seq_len * d_model)
        + (2 * bs * seq_len * d_model * vocab_size)
    )


print(
    f"""
    params: {count_params()}
    flops: {count_flops()}
"""
)
