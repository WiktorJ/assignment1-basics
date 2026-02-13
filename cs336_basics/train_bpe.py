"""Command-line interface for training a BPE tokenizer."""

import argparse

from cs336_basics.tokenizer import train_bpe


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on a text file."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "vocab_size",
        type=int,
        help="Desired vocabulary size.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens (default: ['<|endoftext|>']).",
    )
    parser.add_argument(
        "--split-special-token",
        type=str,
        default="<|endoftext|>",
        help="Special token used to split chunks (default: '<|endoftext|>').",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel workers for pretokenization (default: 2).",
    )
    parser.add_argument(
        "--save-vocab-path",
        type=str,
        default=None,
        help="Path to save the vocabulary to a JSON file.",
    )
    parser.add_argument(
        "--save-merges-path",
        type=str,
        default=None,
        help="Path to save the merges to a text file.",
    )

    args = parser.parse_args()

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        split_special_token=args.split_special_token,
        num_workers=args.num_workers,
        save_vocab_path=args.save_vocab_path,
        save_merges_path=args.save_merges_path,
    )

    print(f"Training complete. Vocabulary size: {len(vocab)}, Merges: {len(merges)}")


if __name__ == "__main__":
    main()
