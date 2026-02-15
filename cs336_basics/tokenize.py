"""
Encode or decode text using a trained BPE tokenizer.

Encode mode: reads a text file, encodes it, and writes token IDs as a uint16 numpy array.
Decode mode: reads a uint16 numpy array, decodes it, and writes the text to a file.
"""

import argparse
import numpy as np

from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Encode or decode text using a trained BPE tokenizer."
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file.",
    )
    parser.add_argument(
        "--merges-path",
        type=str,
        required=True,
        help="Path to the merges text file.",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens (default: '<|endoftext|>').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["encode", "decode"],
        help="Mode: 'encode' text to tokens or 'decode' tokens to text.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input file (text file for encode, .npy file for decode).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output file (.npy file for encode, text file for decode).",
    )

    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path, special_tokens=args.special_tokens
    )

    if args.mode == "encode":
        with open(args.input_path, "r", encoding="utf-8") as f:
            text = f.read()
        token_ids = tokenizer.encode(text)
        arr = np.array(token_ids, dtype=np.uint16)
        np.save(args.output_path, arr)
        print(f"Encoded {len(text)} characters into {len(token_ids)} tokens -> {args.output_path}")

    elif args.mode == "decode":
        arr = np.load(args.input_path)
        token_ids = arr.astype(np.int64).tolist()
        text = tokenizer.decode(token_ids)
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Decoded {len(token_ids)} tokens into {len(text)} characters -> {args.output_path}")


if __name__ == "__main__":
    main()
