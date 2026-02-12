from typing import BinaryIO
import os
import regex as re
from collections import Counter, defaultdict
from sortedcontainers import SortedList


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _pretokenize(
    text: str, pretokenize_regex: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
) -> Counter[tuple[bytes, ...]]:
    return Counter(
        tuple(bytes([c]) for c in match.group().encode("UTF-8")) for match in re.finditer(pretokenize_regex, text)
    )


def merge_subsequences(original, target):
    result = []
    i = 0
    target_len = len(target)
    merged_val = b"".join(target)  # Join the bytes to merge

    while i < len(original):
        # Check if the subsequence starting at 'i' matches our target
        if original[i : i + target_len] == target:
            result.append(merged_val)
            i += target_len  # Skip the elements we just merged
        else:
            result.append(original[i])
            i += 1

    return tuple(result)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str = "<|endoftext|>",
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_workers, split_special_token.encode("utf-8"))
        vocab = {}
        for t in special_tokens:
            vocab[len(vocab)] = t.encode("utf-8")
        for t in range(256):
            vocab[len(vocab)] = bytes([t])
        merges = []
        special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)

        # TODO: Parallelize
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            text_chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pretokens = Counter()
            for chunk in re.split(special_tokens_pattern, text_chunk):
                if chunk:
                    pretokens += _pretokenize(chunk)
            pairs = Counter()
            pretokens_by_pair = defaultdict(set)
            for token, count in pretokens.items():
                for pair in zip(token[:-1], token[1:]):
                    pairs[pair] += count
                    pretokens_by_pair[pair].add(token)

            pair_sl = SortedList(((count, pair) for pair, count in pairs.items()))

            while len(vocab) < vocab_size:
                found_next_pair = False
                # Find pair with highest frequency, filter pairs that have changed
                top_count, top_pair = pair_sl.pop(-1)
                while not found_next_pair:
                    if top_pair not in pairs:
                        top_count, top_pair = pair_sl.pop(-1)
                        continue
                    if pairs[top_pair] != top_count:
                        pair_sl.add((pairs[top_pair], top_pair))
                        top_count, top_pair = pair_sl.pop(-1)
                        continue
                    found_next_pair = True

                merges.append(top_pair)
                vocab[len(vocab)] = b"".join(top_pair)

                new_pairs = set()
                for pretoken in pretokens_by_pair[top_pair].copy():
                    pretoken_count = pretokens[pretoken]
                    new_pretoken = merge_subsequences(pretoken, top_pair)
                    old_pairs = list(zip(pretoken[:-1], pretoken[1:]))
                    current_pairs = list(zip(new_pretoken[:-1], new_pretoken[1:]))
                    for pair in old_pairs:
                        pairs[pair] -= pretoken_count
                        pretokens_by_pair[pair].discard(pretoken)
                    for pair in current_pairs:
                        pairs[pair] += pretoken_count
                        new_pairs.add(pair)
                        pretokens_by_pair[pair].add(new_pretoken)
                    pretokens[new_pretoken] += pretoken_count
                    del pretokens[pretoken]
                del pairs[top_pair]
                for new_pair in new_pairs:
                    pair_sl.add((pairs[new_pair], new_pair))

        return vocab, merges
