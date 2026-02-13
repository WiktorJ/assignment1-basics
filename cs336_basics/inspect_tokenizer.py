"""Simple script to display BPE vocab and merges in human-readable format."""

import argparse
import json


def _hex_to_readable(hex_str: str) -> str:
    """Convert a hex-encoded byte string to a human-readable representation."""
    raw = bytes.fromhex(hex_str)
    try:
        decoded = raw.decode("utf-8")
        # Replace common whitespace with visible representations
        decoded = decoded.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return decoded
    except UnicodeDecodeError:
        return repr(raw)


def inspect_vocab(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"=== Vocabulary ({len(vocab)} tokens) ===")
    # Sort by token id
    for hex_token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        readable = _hex_to_readable(hex_token)
        print(f"  {idx:>6d}  {readable}")


def inspect_merges(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    print(f"=== Merges ({len(lines)} rules) ===")
    for i, line in enumerate(lines):
        parts = line.strip().split(" ")
        if len(parts) != 2:
            continue
        a_readable = _hex_to_readable(parts[0])
        b_readable = _hex_to_readable(parts[1])
        merged = _hex_to_readable(parts[0] + parts[1])
        print(f"  {i:>6d}  {a_readable!r} + {b_readable!r} -> {merged!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Display BPE vocab and/or merges in human-readable format."
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to the vocabulary JSON file.",
    )
    parser.add_argument(
        "--merges-path",
        type=str,
        default=None,
        help="Path to the merges text file.",
    )

    args = parser.parse_args()

    if args.vocab_path is None and args.merges_path is None:
        parser.error("Provide at least one of --vocab-path or --merges-path.")

    if args.vocab_path is not None:
        inspect_vocab(args.vocab_path)

    if args.vocab_path is not None and args.merges_path is not None:
        print()

    if args.merges_path is not None:
        inspect_merges(args.merges_path)


if __name__ == "__main__":
    main()
