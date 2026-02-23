"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator


# GPT-2 pre-tokenization pattern
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# Fast byte lookup table used when converting UTF-8 bytes -> tuple[bytes, ...].
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.

    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []

    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return

    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)

    # Build a pattern that matches special tokens
    import re as std_re

    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"

    # Split text by special tokens
    parts = std_re.split(split_pattern, text)

    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization
            for match in GPT2_PAT.finditer(part):
                yield match.group()


def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.

    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include (e.g., ["<|endoftext|>"])

    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs in order they were learned [(bytes, bytes), ...]

    Algorithm Overview:
        BPE iteratively merges the most frequent pair of adjacent tokens until
        the vocabulary reaches the target size.

    Detailed Steps:

    1. VOCABULARY INITIALIZATION
       The initial vocabulary is built in this exact order:
       - First: Add special tokens (in the order provided)
       - Then: Add all 256 single-byte values (0x00 to 0xFF)

       Example with special_tokens=["<|endoftext|>"]:
         vocab = {
             0: b"<|endoftext|>",   # Special token first
             1: b"\\x00",           # Byte 0
             2: b"\\x01",           # Byte 1
             ...
             256: b"\\xff",         # Byte 255
         }

       So the initial vocab size = len(special_tokens) + 256

    2. WORD FREQUENCY COUNTING
       - Pre-tokenize the corpus using pre_tokenize(text, special_tokens)
       - For each pre-token, convert to bytes and represent as tuple of single bytes
       - Skip any word containing a "forbidden substring" (prefix of a special token)

       Example: "hello" -> (b'h', b'e', b'l', b'l', b'o')

       word_freqs is a Counter mapping: tuple[bytes, ...] -> frequency

    3. PAIR FREQUENCY COUNTING
       Count how often each adjacent pair appears across ALL words, weighted by
       word frequency.

       Example: If word (b'h', b'e', b'l', b'l', b'o') appears 10 times:
         - pair (b'h', b'e') gets +10
         - pair (b'e', b'l') gets +10
         - pair (b'l', b'l') gets +10
         - pair (b'l', b'o') gets +10

    4. MERGE LOOP (repeat until vocab_size is reached)

       a. SELECT BEST PAIR (DETERMINISTIC TIE-BREAKING):
          Find the pair with highest frequency. If multiple pairs have the same
          frequency, select the lexicographically largest pair.

          Lexicographic comparison on (bytes, bytes) tuples:
            - Compare first element as bytes
            - If equal, compare second element as bytes

          Example: If pairs (b'a', b'b') and (b'a', b'c') both have freq=100,
                   select (b'a', b'c') because b'c' > b'b'

          Implementation: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          This sorts by (frequency, pair) and takes the max.
                          Since we want highest freq and highest pair for ties,
                          use: max(pair_counts, key=lambda p: (pair_counts[p], p))

                          Note: Python compares bytes lexicographically by default.

       b. CREATE MERGED TOKEN:
          new_token = first + second  (bytes concatenation)
          Add to vocabulary with next available token_id
          Append (first, second) to merges list

       c. UPDATE WORD REPRESENTATIONS:
          For each word in word_freqs, apply the merge using merge_word()
          This replaces all occurrences of the pair with the merged token

       d. UPDATE PAIR COUNTS:
          Recompute pair frequencies for the updated words
          (Or incrementally update - subtract old pairs, add new pairs)

    5. RETURN
       Return (vocab, merges) where merges is the list of pairs in the order
       they were merged.

    Performance Note:
        A naive implementation recomputing all pair counts each iteration is O(n²).
        For efficiency, incrementally update pair counts by only processing words
        that contained the merged pair.
    """
    special_tokens = special_tokens or []

    # Read the corpus
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Build set of "forbidden" substrings from special tokens
    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])

    # TODO: Implement BPE training
    vocab: dict[int, bytes] = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = BYTE_TOKENS[b]

    # 2. Word frequency counting
    # Cache pre-token conversions since corpora contain many repeated tokens.
    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    token_cache: dict[str, tuple[bytes, ...] | None] = {}

    for token in pre_tokenize(text, special_tokens):
        cached = token_cache.get(token)
        if cached is not None:
            word_freqs[cached] += 1
            continue
        if token in token_cache:  # Cached "skip" marker
            continue

        token_bytes = token.encode("utf-8")

        # Skip words containing forbidden substrings.
        # Keep this check semantically identical to the baseline implementation.
        skip_token = False
        for forbidden in forbidden_substrings:
            if forbidden in token_bytes:
                skip_token = True
                break
        if skip_token:
            token_cache[token] = None
            continue

        word = tuple(BYTE_TOKENS[b] for b in token_bytes)
        if word:
            token_cache[token] = word
            word_freqs[word] += 1
        else:
            token_cache[token] = None

    merges: list[tuple[bytes, bytes]] = []

    # Build pair statistics once, then update incrementally after each merge.
    word_pairs: dict[tuple[bytes, ...], set[tuple[bytes, bytes]]] = {}
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, freq in word_freqs.items():
        pairs = get_pairs(word)
        word_pairs[word] = pairs
        for pair in pairs:
            pair_counts[pair] += freq
            pair_to_words.setdefault(pair, set()).add(word)

    # 3. BPE merge loop
    while len(vocab) < vocab_size and pair_counts:
        # Select best pair: highest frequency, lexicographically largest on ties.
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        # Only words containing best_pair can change.
        affected_words = list(pair_to_words.get(best_pair, ()))
        if not affected_words:
            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            continue

        merged_word_freqs: Counter[tuple[bytes, ...]] = Counter()

        # Remove old pair contributions from affected words.
        for old_word in affected_words:
            freq = word_freqs.pop(old_word, 0)
            if freq == 0:
                continue

            old_pairs = word_pairs.pop(old_word, set())
            for pair in old_pairs:
                new_count = pair_counts.get(pair, 0) - freq
                if new_count > 0:
                    pair_counts[pair] = new_count
                else:
                    pair_counts.pop(pair, None)

                words_for_pair = pair_to_words.get(pair)
                if words_for_pair is not None:
                    words_for_pair.discard(old_word)
                    if not words_for_pair:
                        pair_to_words.pop(pair, None)

            merged_word = merge_word(old_word, best_pair)
            merged_word_freqs[merged_word] += freq

        # Add merged words and their pair contributions.
        for new_word, add_freq in merged_word_freqs.items():
            prev_freq = word_freqs.get(new_word, 0)
            word_freqs[new_word] = prev_freq + add_freq

            if prev_freq == 0:
                pairs = get_pairs(new_word)
                word_pairs[new_word] = pairs
                for pair in pairs:
                    pair_to_words.setdefault(pair, set()).add(new_word)
            else:
                pairs = word_pairs[new_word]

            for pair in pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + add_freq

    return vocab, merges
