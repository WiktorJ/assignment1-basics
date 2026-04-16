#!/bin/bash
uv run python cs336_basics/decoding.py \
  --run_dir model_training/20260415-092520 \
  --checkpoint_id 6800 \
  --prompt "Once upon a time" \
  --max_tokens 256 \
  --temperature 1.0 \
  --top_p_threshold 0.9 \
  --device mps \
  --tokenizer_config.vocab_path data/vocab_TinyStoriesV2-GPT4-train.json \
  --tokenizer_config.merges_path data/merges_TinyStoriesV2-GPT4-train.txt \
  --tokenizer_config.special_tokens '["<|endoftext|>"]'
