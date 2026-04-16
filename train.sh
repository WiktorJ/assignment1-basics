#!/bin/bash
uv run python cs336_basics/training.py \
  --train_input_path data/TinyStoriesV2-GPT4-train-encoded.txt.npy \
  --eval_input_path data/TinyStoriesV2-GPT4-valid-encoded.txt.npy \
  --output_path model_training \
  --max_steps 20000 \
  --device mps \
  --lr_config.cosine_cycle_iters 20000 \
  --model_config.vocab_size 10000 \
  --model_config.context_length 256 \
  --model_config.d_model 512 \
  --model_config.num_layers 8 \
  --model_config.num_heads 8 \
  --model_config.d_ff 2048 \
  --save_interval 2000 \
  --resume_checkpoint_path model_training/20260415-092520/checkpoints/checkpoint-6800.pt
