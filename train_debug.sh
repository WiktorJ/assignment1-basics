#!/bin/bash
uv run python -m cs336_basics/training \
  --train_input_path data/TinyStoriesV2-GPT4-train-encoded.txt \
  --eval_input_path data/TinyStoriesV2-GPT4-eval-encoded.txt \
  --output_path model_training \
  --max_steps 20 \
  --model_config.vocab_size 10000 \
  --model_config.context_length 64 \
  --model_config.d_model 128 \
  --model_config.num_layers 2 \
  --model_config.num_heads 2 \
  --model_config.d_ff 512
