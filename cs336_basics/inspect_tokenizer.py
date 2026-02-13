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


def inspect_vocab(path: str, output_path: str, max_rows: int | None = None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    sorted_items = sorted(vocab.items(), key=lambda x: x[1])
    if max_rows is not None:
        sorted_items = sorted_items[:max_rows]
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"=== Vocabulary ({len(vocab)} tokens) ===\n")
        for hex_token, idx in sorted_items:
            readable = _hex_to_readable(hex_token)
            out.write(f"  {idx:>6d}  {readable}\n")


def inspect_merges(path: str, output_path: str, max_rows: int | None = None) -> None:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"=== Merges ({len(lines)} rules) ===\n")
        for i, line in enumerate(lines):
            if max_rows is not None and i >= max_rows:
                break
            parts = line.strip().split(" ")
            if len(parts) != 2:
                continue
            a_readable = _hex_to_readable(parts[0])
            b_readable = _hex_to_readable(parts[1])
            merged = _hex_to_readable(parts[0] + parts[1])
            out.write(f"  {i:>6d}  {a_readable!r} + {b_readable!r} -> {merged!r}\n")


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
    parser.add_argument(
        "--output-vocab-path",
        type=str,
        default="vocab_readable.txt",
        help="Path to write the human-readable vocabulary (default: vocab_readable.txt).",
    )
    parser.add_argument(
        "--output-merges-path",
        type=str,
        default="merges_readable.txt",
        help="Path to write the human-readable merges (default: merges_readable.txt).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to write (default: all).",
    )

    args = parser.parse_args()

    if args.vocab_path is None and args.merges_path is None:
        parser.error("Provide at least one of --vocab-path or --merges-path.")

    if args.vocab_path is not None:
        inspect_vocab(args.vocab_path, args.output_vocab_path, args.max_rows)

    if args.merges_path is not None:
        inspect_merges(args.merges_path, args.output_merges_path, args.max_rows)


if __name__ == "__main__":
    main()
