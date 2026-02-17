from typing import BinaryIO, Iterable, Iterator
import json
import os
import regex as re
from collections import Counter, defaultdict
from sortedcontainers import SortedList
import multiprocessing as mp


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


def _pretokenize_block(
    text: str, pretokenize_regex: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
) -> list[tuple[bytes, ...]]:
    return [tuple(bytes([c]) for c in match.group().encode("UTF-8")) for match in re.finditer(pretokenize_regex, text)]


def _pretokenize_chunk(special_tokens_pattern: str, text_chunk: str) -> Counter[tuple[bytes, ...]]:
    pretokens = Counter()
    for block in re.split(special_tokens_pattern, text_chunk):
        if block.strip():
            pretokens += Counter(_pretokenize_block(block))
    return pretokens


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


def _save_vocab(vocab: dict[int, bytes], path: str | os.PathLike):
    serializable = {vocab[idx].hex(): idx for idx in sorted(vocab.keys())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
        f.write("\n")


def _save_merges(merges: list[tuple[bytes, bytes]], path: str | os.PathLike):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str = "<|endoftext|>",
    num_workers: int = 2,
    save_vocab_path: str | os.PathLike | None = None,
    save_merges_path: str | os.PathLike | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens_pattern = "|".join(re.escape(token) for token in special_tokens)
    pool = mp.Pool(num_workers)
    chunk_pretokens = []
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, num_workers, split_special_token.encode("utf-8"))
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            text_chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_pretokens.append(pool.apply_async(_pretokenize_chunk, (special_tokens_pattern, text_chunk)))

    pool.close()
    pool.join()
    pretokens = sum((chunk.get() for chunk in chunk_pretokens), Counter())

    vocab = {}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")
    for t in range(256):
        vocab[len(vocab)] = bytes([t])
    merges = []

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

    if save_vocab_path is not None:
        _save_vocab(vocab, save_vocab_path)
    if save_merges_path is not None:
        _save_merges(merges, save_merges_path)

    return vocab, merges


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Initialize a BPE tokenizer.
        Args:
            vocab (dict[int, bytes]): The vocabulary mapping from token IDs to token bytes.
            merges (list[tuple[bytes, bytes]]): The BPE merges, where each tuple represents a merge operation.
            special_tokens (list[str], optional): A list of special tokens to be added to the tokenizer vocabulary.
                These strings will never be split into multiple tokens, and will always be
                kept as a single token.
        """
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        self.pretoken_cache = defaultdict(list)
        if special_tokens:
            self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
            self.special_tokens_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            )
            next_id = max(vocab.keys()) + 1
            for token in self.special_tokens:
                token = token.encode("utf-8")
                if token not in self.vocab_inv:
                    self.vocab[next_id] = token
                    self.vocab_inv[token] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_tokens_pattern = None

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {int(token_id): bytes.fromhex(token_bytes) for token_bytes, token_id in vocab.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [tuple(line.strip().split(" ")) for line in f]
            merges = [(bytes.fromhex(merge[0]), bytes.fromhex(merge[1])) for merge in merges]
        return cls(vocab, merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        Args:
            text (str): The input text to encode.
        Returns:
            list[int]: A list of token IDs representing the input text.
        """
        if not self.special_tokens_pattern:
            return self._encode_block(text)

        tokens = []
        for block in re.split(self.special_tokens_pattern, text):
            if block in self.special_tokens:
                tokens.append(self.vocab_inv[block.encode("utf-8")])
            else:
                tokens.extend(self._encode_block(block))
        return tokens

    def _encode_block(self, block: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        Args:
            text (str): The input text to encode.
        Returns:
            list[int]: A list of token IDs representing the input text.
        """
        tokens = []
        for pretoken in _pretokenize_block(block):
            if pretoken in self.pretoken_cache:
                tokens.extend(self.pretoken_cache[pretoken])
                continue
            while True:
                earliest_merge_index = len(self.merges)
                earliest_pretoken = None
                for p in zip(pretoken[:-1], pretoken[1:]):
                    if p in self.merges_dict and self.merges_dict[p] < earliest_merge_index:
                        earliest_merge_index = self.merges_dict[p]
                        earliest_pretoken = p
                if earliest_pretoken:
                    pretoken = merge_subsequences(pretoken, earliest_pretoken)
                else:
                    break
            for token in pretoken:
                if token in self.vocab_inv:
                    tokens.append(self.vocab_inv[token])
                    self.pretoken_cache[pretoken].append(self.vocab_inv[token])
                else:
                    raise ValueError(f"Token {token} not in vocabulary")
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for element in iterable:
            yield from self.encode(element)

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs into a string.
        Args:
            tokens (list[int]): A list of token IDs to decode.
        Returns:
            str: The decoded string.
        """
        return b"".join([self.vocab[token] for token in tokens]).decode("utf-8", errors="replace")
