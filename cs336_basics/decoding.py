from dataclasses import dataclass, field
import torch
import os
import typing
import tyro
import json
import cs336_basics.model as model_lib
import cs336_basics.tokenizer as tokenizer_lib


@dataclass
class TokenizerConfig:
    vocab_path: str
    merges_path: str
    special_tokens: list[str] | None = None


@dataclass
class DecodingConfig:
    run_dir: str
    checkpoint_id: int
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p_threshold: float = 1.0
    device: str = "cpu"
    eos_token_text: str = "<|endoftext|>"

    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)


def load_model(run_dir: str | os.PathLike, checkpoint_id: int, device: str = "cpu") -> model_lib.Transformer:
    with open(os.path.join(run_dir, "model_config.json")) as f:
        model_config = json.load(f)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = model_lib.Transformer(**model_config, device=device, dtype=dtype)
    model.to(device)

    checkpoint_path = os.path.join(run_dir, "checkpoints", f"checkpoint-{checkpoint_id}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def decode(config: DecodingConfig):
    model = load_model(config.run_dir, config.checkpoint_id, config.device)
    tokenizer = tokenizer_lib.Tokenizer.from_files(
        config.tokenizer_config.vocab_path, config.tokenizer_config.merges_path, config.tokenizer_config.special_tokens
    )

    encoded_prompt = tokenizer.encode(config.prompt)
    encoded_prompt = torch.tensor(encoded_prompt, dtype=torch.int64, device=config.device).unsqueeze(0)
    encoded_promtp_len = encoded_prompt.shape[-1]
    all_tokens = torch.zeros((1, encoded_promtp_len + config.max_tokens), dtype=torch.int64, device=config.device)
    all_tokens[:, :encoded_promtp_len] = encoded_prompt
    eos_token = tokenizer.vocab_inv[config.eos_token_text.encode("utf-8")]
    print(tokenizer.decode(encoded_prompt[0].tolist()), end="", flush=True)
    with torch.no_grad():
        tokens_count = encoded_promtp_len
        while tokens_count < encoded_promtp_len + config.max_tokens and all_tokens[:, tokens_count - 1] != eos_token:
            logits = model(all_tokens[:, :tokens_count])
            logits = logits / config.temperature
            probs = torch.softmax(logits[:, -1], dim=-1)
            if config.top_p_threshold < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                probs_cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff_index = torch.searchsorted(probs_cumsum.squeeze(0), config.top_p_threshold, side="left").item()
                sorted_probs[:, cutoff_index + 1:] = 0.0
                sorted_probs /= sorted_probs.sum()
                sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_pos)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            all_tokens[:, tokens_count] = next_token
            tokens_count += 1
            print(tokenizer.decode([next_token.item()]), end="", flush=True)
    print()


def main():
    config = tyro.cli(DecodingConfig)
    decode(config)


if __name__ == "__main__":
    main()
