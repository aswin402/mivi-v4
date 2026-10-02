"""Bounded GGUF v2/v3 metadata inspection; never load tensor weights.

This preflight validates tokenizer/context metadata, not graph compatibility.
Safety caps are diagnostic parser budgets, not model-family dimensions.
"""

import os
import stat
import struct
from pathlib import Path

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024
MAX_ITEMS = 1024 * 1024
MAX_PROMPT_IDS = 4096
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "B", 10: "Q", 11: "q", 12: "d"}
INTEGERS = {0, 1, 2, 3, 4, 5, 10, 11}


class _Reader:
    def __init__(self, source, file_bytes):
        self.source = source
        self.limit = min(file_bytes, MAX_METADATA_BYTES)

    def span(self, size):
        if size < 0 or self.source.tell() + size > self.limit:
            raise ValueError("GGUF metadata is truncated or exceeds parser budget")

    def read(self, size):
        self.span(size)
        value = self.source.read(size)
        if len(value) != size:
            raise ValueError("truncated GGUF metadata")
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, retain=False):
        size = self.number("Q")
        if size > MAX_STRING_BYTES:
            raise ValueError("GGUF metadata string exceeds parser budget")
        self.span(size)
        if retain:
            try:
                return self.read(size).decode("utf-8", "strict")
            except UnicodeError as exc:
                raise ValueError("invalid GGUF metadata UTF-8") from exc
        self.source.seek(size, os.SEEK_CUR)
        return None

    def value(self, kind, retain=False):
        if kind in SCALARS:
            value = self.number(SCALARS[kind])
            if kind == 7:
                if value not in (0, 1):
                    raise ValueError("invalid GGUF boolean encoding")
                value = bool(value)
            return value if retain else None
        if kind == 8:
            return self.string(retain)
        if kind == 9:
            element = self.number("I")
            count = self.number("Q")
            if count > MAX_ITEMS or element not in (*SCALARS, 8):
                raise ValueError("invalid or oversized GGUF metadata array")
            start = self.source.tell()
            if element == 8:
                for _ in range(count):
                    self.string()
            else:
                size = count * struct.calcsize("<" + SCALARS[element])
                self.span(size)
                if element == 7:
                    remaining = size
                    while remaining:
                        chunk = self.read(min(remaining, 64 * 1024))
                        if any(value not in (0, 1) for value in chunk):
                            raise ValueError("invalid GGUF boolean encoding")
                        remaining -= len(chunk)
                else:
                    self.source.seek(size, os.SEEK_CUR)
            return (element, count, start) if retain else None
        raise ValueError("unsupported GGUF metadata type")


def _inspect(source, file_bytes):
    reader = _Reader(source, file_bytes)
    if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
        raise ValueError("expected GGUF v2/v3")
    reader.number("Q")  # tensor count: tensor descriptors/weights are not read.
    count = reader.number("Q")
    if count > MAX_ITEMS:
        raise ValueError("GGUF metadata key count exceeds parser budget")
    keys = set()
    retained = {}
    for _ in range(count):
        key = reader.string(True)
        if key in keys:
            raise ValueError("duplicate GGUF metadata key")
        keys.add(key)
        kind = reader.number("I")
        wanted = key in {"general.architecture", "tokenizer.ggml.tokens",
                         "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                         "tokenizer.ggml.add_bos_token"} or key.endswith(".context_length")
        value = reader.value(kind, wanted)
        if wanted:
            retained[key] = (kind, value)
    architecture = retained.get("general.architecture")
    if architecture is None or architecture[0] != 8 or not architecture[1]:
        raise ValueError("missing/invalid architecture metadata")
    architecture = architecture[1]
    tokens = retained.get("tokenizer.ggml.tokens")
    if tokens is None or tokens[0] != 9 or tokens[1][0] != 8 or not tokens[1][1]:
        raise ValueError("missing/invalid tokenizer vocabulary")
    vocab_size, tokens_start = tokens[1][1:]

    def token_id(key, fallback):
        kind, value = retained.get(key, (4, fallback))
        if kind not in INTEGERS or not 0 <= value < vocab_size or value > 2**32 - 1:
            raise ValueError("invalid tokenizer special-token ID")
        return value

    bos = token_id("tokenizer.ggml.bos_token_id", 1)
    eos = token_id("tokenizer.ggml.eos_token_id", 2)
    bos_setting = retained.get("tokenizer.ggml.add_bos_token", (7, False))
    if bos_setting[0] != 7:
        raise ValueError("invalid tokenizer BOS policy")
    context = retained.get(architecture + ".context_length")
    if context is not None:
        if context[0] not in INTEGERS or context[1] <= 0:
            raise ValueError("invalid architecture context length")
        context = context[1]
    source.seek(tokens_start)
    for _ in range(eos):
        reader.string()
    eos_text = reader.string(True)
    if len(eos_text.encode()) > 64 * 1024:
        raise ValueError("EOS token string exceeds retained-text budget")
    return {"vocab_size": vocab_size, "bos_id": bos, "eos_id": eos,
            "add_bos": bos_setting[1], "context_length": context,
            "stop_strings": [eos_text], "architecture": architecture}


def inspect_metadata(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(Path(path), flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("GGUF input must be a regular non-symlink file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            return _inspect(source, info.st_size)
    finally:
        os.close(fd)


def normalize_ids(ids, metadata):
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_PROMPT_IDS:
        raise ValueError("invalid prompt ID count")
    if any(isinstance(token, bool) or not isinstance(token, int)
           or not 0 <= token < metadata["vocab_size"] for token in ids):
        raise ValueError("prompt ID is outside model vocabulary")
    if metadata["add_bos"] and ids[0] != metadata["bos_id"]:
        if len(ids) == MAX_PROMPT_IDS:
            raise ValueError("invalid prompt ID count")
        return [metadata["bos_id"], *ids]
    return list(ids)
