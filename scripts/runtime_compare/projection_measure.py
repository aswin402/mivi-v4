#!/usr/bin/env python3
"""Run a private, paired profiled/unprofiled projection measurement session."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from private_io import (PrivateDirectory, open_regular_file, read_bounded,
                        validate_regular_file)
from process_supervisor import run_child

MAX_MANIFEST_BYTES = 64 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_CHANNEL_BYTES = 64 * 1024
MAX_CASES = 16
MAX_MEASURED_CALLS = 32
MAX_WALL_SECONDS = 180
MAX_SESSION_SECONDS = 900
MAX_RSS_BYTES = 2 * 1024**3
MAX_ARTIFACT_BYTES = 64 * 1024**2
REPORT_RESERVE_BYTES = 1024 * 1024
PROFILE_REPORT_BOUND_BYTES = 2048
SAMPLE_REPORT_BOUND_BYTES = 8192
CHILD_CLEANUP_RESERVE_SECONDS = 0.5
MANIFEST_FIELDS = {"schema", "binary", "revision", "repetitions", "wall_seconds",
                   "session_seconds", "rss_bytes", "artifact_bytes", "buffer_limit_bytes",
                   "model_limit_bytes", "cases"}
CASE_FIELDS = {"name", "batch", "source", "warmup_calls", "measured_calls"}
SYNTHETIC_FIELDS = {"kind", "ggml_type", "rows", "cols"}
GGUF_FIELDS = {"kind", "model_path", "tensor"}
RESULT_FIELDS = {"schema", "status", "source_kind", "model_path", "tensor_name",
                 "mapping_bytes", "format", "ggml_type", "rows", "cols", "batch",
                 "branch", "threads", "profile", "activation_source", "setup_ns",
                 "call_wall_ns", "output_bits", "all_calls_bit_identical", "profile_calls",
                 "estimated_heap_bytes", "output_artifact_bound_bytes"}
PROFILE_FIELDS = {"schema", "branch", "call_wall_ns", "validation_ns", "buffer_init_ns",
                  "input_transpose_ns", "rows_wall_ns", "output_layout_ns",
                  "delegated_matvec_ns", "unclassified_wall_ns", "workers"}
WORKER_FIELDS = {"scratch_init_ns", "decode_ns", "accumulate_ns", "zero_copy_ns", "rows"}
SUPPORTED_TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2),
                   8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
                   14: ("Q6_K", 256, 210)}
INTEGER_MAX = (1 << 64) - 1


def _report_reserve_bytes(cases: list[dict]) -> int:
    # Three profiled samples per case; compact report metadata includes bounded
    # log excerpts, errors, summaries, comparison rows, and every profile call.
    return (REPORT_RESERVE_BYTES
            + sum(3 * case["measured_calls"] * PROFILE_REPORT_BOUND_BYTES
                  + 6 * SAMPLE_REPORT_BOUND_BYTES for case in cases))


def _predicted_artifact_bytes(cases: list[dict], report_reserve: int) -> int:
    predicted = report_reserve
    for case in cases:
        rows = case.get("rows")
        output_bound = 0 if rows is None else 11 * case["batch"] * rows + 256 * 1024
        if output_bound > MAX_RESULT_BYTES:
            raise ValueError("predicted child result exceeds the 4 MiB result cap")
        input_size = len(json.dumps(case["input"], separators=(",", ":")).encode())
        # One bounded child result per attempt; compact reports do not duplicate
        # output bits. Logs and profile records are covered by report_reserve.
        predicted += 6 * (output_bound + input_size)
    return predicted


def _output_bits_sha256(bits: list[int]) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(bits), 4096):
        chunk = bits[start:start + 4096]
        digest.update(struct.pack("<" + "I" * len(chunk), *chunk))
    return digest.hexdigest()


def _integer(value: Any, *, minimum: int = 0, maximum: int = INTEGER_MAX) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _duplicate_rejecting_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest path must be absolute and contain no parent traversal")
    value = json.loads(read_bounded(path, MAX_MANIFEST_BYTES),
                       object_pairs_hook=_duplicate_rejecting_pairs)
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be an absolute path")
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be absolute and contain no parent traversal")
    return raw


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject existing symlink components without creating or resolving the path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        if current.is_symlink():
            raise ValueError(f"{label} must not contain symlink components")
        if current != path and not current.is_dir():
            raise ValueError(f"{label} parent components must be directories")


def _case_dimensions(case: dict) -> tuple[int | None, int | None]:
    source = case["source"]
    if source["kind"] == "synthetic":
        return source["rows"], source["cols"]
    return None, None


def validate_manifest(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS:
        raise ValueError("manifest fields do not match schema")
    if not _integer(value["schema"], minimum=1, maximum=1):
        raise ValueError("unsupported manifest schema")
    binary = _absolute_path(value["binary"], "binary")
    revision = value["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be exactly 40 hexadecimal characters")
    limits = (("repetitions", 3, 3), ("wall_seconds", 1, MAX_WALL_SECONDS),
              ("session_seconds", 1, MAX_SESSION_SECONDS), ("rss_bytes", 1, MAX_RSS_BYTES),
              ("buffer_limit_bytes", 1, INTEGER_MAX), ("model_limit_bytes", 1, INTEGER_MAX))
    settings = {}
    for key, low, high in limits:
        if not _integer(value[key], minimum=low, maximum=high):
            raise ValueError(f"{key} is outside its allowed integer range")
        settings[key] = value[key]
    if not _integer(value["artifact_bytes"], minimum=REPORT_RESERVE_BYTES + 1,
                    maximum=MAX_ARTIFACT_BYTES):
        raise ValueError("artifact_bytes is outside its allowed integer range")
    settings["artifact_bytes"] = value["artifact_bytes"]
    if value["session_seconds"] < value["wall_seconds"]:
        raise ValueError("session_seconds must cover at least one child wall limit")
    cases = value["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("cases must contain 1 to 16 explicitly selected cases")
    names = set()
    normalized = []
    for case in cases:
        if not isinstance(case, dict) or not CASE_FIELDS <= set(case):
            raise ValueError("case fields do not match schema")
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("case name must be a unique safe identifier")
        if name in names:
            raise ValueError("case names must be unique")
        names.add(name)
        if not _integer(case["batch"], minimum=1, maximum=65):
            raise ValueError("batch must be an integer from 1 through 65")
        if not _integer(case["warmup_calls"], minimum=0, maximum=1):
            raise ValueError("warmup_calls must be an integer from 0 through 1")
        if not _integer(case["measured_calls"], minimum=1, maximum=MAX_MEASURED_CALLS):
            raise ValueError("measured_calls must be an integer from 1 through 32")
        source = case["source"]
        if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
            raise ValueError("source must be a tagged source object")
        if source["kind"] == "synthetic":
            if set(source) != SYNTHETIC_FIELDS:
                raise ValueError("synthetic source fields do not match child schema")
            if (not _integer(source["ggml_type"], minimum=0)
                    or source["ggml_type"] not in SUPPORTED_TYPES):
                raise ValueError("unsupported synthetic ggml_type")
            if not _integer(source["rows"], minimum=1) or not _integer(source["cols"], minimum=1):
                raise ValueError("synthetic dimensions must be positive integers")
            _format, block, _size = SUPPORTED_TYPES[source["ggml_type"]]
            if source["cols"] % block:
                raise ValueError("synthetic columns do not satisfy format alignment")
            rows = source["rows"]
        elif source["kind"] == "gguf":
            if set(source) != GGUF_FIELDS:
                raise ValueError("GGUF source fields do not match child schema")
            source = dict(source)
            source["model_path"] = str(_absolute_path(source["model_path"], "model_path"))
            if not isinstance(source["tensor"], str) or not source["tensor"] or len(source["tensor"].encode()) > 16 * 1024:
                raise ValueError("GGUF tensor must be a nonempty bounded string")
            rows = None
        else:
            raise ValueError("unsupported source kind")
        # expected_model_sha256 is required only for GGUF and is an outer case field.
        if source["kind"] == "gguf":
            expected = case.get("expected_model_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ValueError("GGUF cases require expected_model_sha256")
            allowed_case_fields = CASE_FIELDS | {"expected_model_sha256"}
        else:
            if "expected_model_sha256" in case:
                raise ValueError("expected_model_sha256 is valid only for GGUF cases")
            allowed_case_fields = CASE_FIELDS
        if set(case) != allowed_case_fields:
            raise ValueError("case fields do not match source kind")
        child_input = {"schema": 1, "source": source, "batch": case["batch"], "threads": 2,
                       "profile": False, "warmup_calls": case["warmup_calls"],
                       "measured_calls": case["measured_calls"],
                       "buffer_limit_bytes": settings["buffer_limit_bytes"],
                       "model_limit_bytes": settings["model_limit_bytes"]}
        # For a tensor-backed source the exact shape is filled by descriptor-only preflight.
        output_bound = 0 if rows is None else 11 * case["batch"] * rows + 256 * 1024
        if output_bound > MAX_RESULT_BYTES:
            raise ValueError("predicted child result exceeds the 4 MiB result cap")
        dimensions = {"rows": rows, "cols": source.get("cols"),
                      "ggml_type": source.get("ggml_type")}
        normalized.append({"name": name, "batch": case["batch"], "source": source,
                           "warmup_calls": case["warmup_calls"],
                           "measured_calls": case["measured_calls"],
                           "expected_model_sha256": case.get("expected_model_sha256"),
                           **dimensions, "input": child_input})
    report_reserve = _report_reserve_bytes(normalized)
    if report_reserve > MAX_RESULT_BYTES:
        raise ValueError("report metadata bound exceeds the 4 MiB report cap")
    if settings["artifact_bytes"] <= report_reserve:
        raise ValueError("artifact budget must exceed the bounded report reserve")
    predicted = _predicted_artifact_bytes(normalized, report_reserve)
    if predicted > settings["artifact_bytes"]:
        raise ValueError("artifact budget cannot reserve bounded outputs and compact reports")
    settings.update({"binary": binary, "revision": revision.lower(), "cases": normalized,
                     "schema": 1, "report_reserve_bytes": report_reserve})
    return settings


def _hash_descriptor(fd: int, limit: int | None = None) -> tuple[str, int]:
    info = os.fstat(fd)
    if limit is not None and info.st_size > limit:
        raise ValueError("file exceeds configured size limit")
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
        if limit is not None and offset > limit:
            raise ValueError("file grew beyond configured size limit while hashing")
    return digest.hexdigest(), offset


class _GGUFDescriptorReader:
    """Read GGUF metadata and tensor descriptors, stopping before tensor data."""

    def __init__(self, fd: int, size: int):
        self.fd = fd
        self.size = size
        self.offset = 0
        self.limit = min(size, 64 * 1024 * 1024)

    def read(self, count: int) -> bytes:
        if count < 0 or self.offset + count > self.limit:
            raise ValueError("GGUF descriptor metadata exceeds the 64 MiB preflight bound")
        data = os.pread(self.fd, count, self.offset)
        if len(data) != count:
            raise ValueError("truncated GGUF descriptor metadata")
        self.offset += count
        return data

    def number(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def skip(self, count: int) -> None:
        if count < 0 or self.offset + count > self.limit:
            raise ValueError("GGUF metadata value exceeds the parser bound")
        self.offset += count

    def string(self, retain: bool = False):
        length = self.number("Q")
        if length > 1024 * 1024:
            raise ValueError("GGUF metadata string exceeds parser bound")
        raw = self.read(length) if retain else None
        if raw is None:
            self.skip(length)
            return None
        return raw.decode("utf-8", "strict")

    def value(self, kind: int, retain: bool = False):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                  7: "B", 10: "Q", 11: "q", 12: "d"}
        if kind in scalar:
            size = struct.calcsize("<" + scalar[kind])
            raw = self.read(size) if retain or kind == 7 else None
            if raw is None:
                self.skip(size)
                return None
            value = struct.unpack("<" + scalar[kind], raw)[0]
            if kind == 7 and value not in (0, 1):
                raise ValueError("invalid GGUF boolean metadata")
            return value if retain else None
        if kind == 8:
            return self.string(retain)
        if kind == 9:
            element, count = self.number("I"), self.number("Q")
            if count > 1_048_576:
                raise ValueError("GGUF metadata array exceeds parser bound")
            if element == 8:
                for _ in range(count):
                    self.string()
                return None
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                     10: 8, 11: 8, 12: 8}
            if element not in sizes:
                raise ValueError("unsupported GGUF metadata array type")
            size = count * sizes[element]
            if size > self.limit - self.offset:
                raise ValueError("GGUF metadata array exceeds parser bound")
            if element == 7:
                raw = self.read(size)
                if any(value not in (0, 1) for value in raw):
                    raise ValueError("invalid GGUF boolean array")
            else:
                self.skip(size)
            return None
        raise ValueError("unsupported GGUF metadata type")


def _gguf_tensor_descriptor(fd: int, size: int, tensor_name: str) -> dict:
    reader = _GGUFDescriptorReader(fd, size)
    if reader.read(4) != b"GGUF" or reader.number("I") not in (2, 3):
        raise ValueError("expected GGUF version 2 or 3")
    tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
    if tensor_count > 1_048_576 or metadata_count > 1_048_576:
        raise ValueError("GGUF descriptor counts exceed parser bound")
    alignment = 32
    seen_metadata = set()
    for _ in range(metadata_count):
        key = reader.string(True)
        if key in seen_metadata:
            raise ValueError("duplicate GGUF metadata key")
        seen_metadata.add(key)
        kind = reader.number("I")
        value = reader.value(kind, retain=(key == "general.alignment"))
        if key == "general.alignment":
            if kind not in {0, 2, 4, 10} or not _integer(value, minimum=1, maximum=1 << 20):
                raise ValueError("invalid GGUF alignment metadata")
            alignment = value
    found = None
    seen_names = set()
    for _ in range(tensor_count):
        name = reader.string(True)
        if name in seen_names:
            raise ValueError("duplicate GGUF tensor descriptor")
        seen_names.add(name)
        rank = reader.number("I")
        if not 1 <= rank <= 4:
            raise ValueError("GGUF tensor rank is outside descriptor bounds")
        dims = [reader.number("Q") for _ in range(rank)]
        ggml_type = reader.number("I")
        offset = reader.number("Q")
        if name == tensor_name:
            found = {"name": name, "rank": rank, "dims": dims, "ggml_type": ggml_type,
                     "offset": offset, "alignment": alignment}
    if found is None:
        raise ValueError("selected GGUF tensor descriptor was not found")
    if found["rank"] != 2:
        raise ValueError("selected GGUF tensor rank must be exactly two")
    cols, rows = found["dims"]
    if not cols or not rows or found["ggml_type"] not in SUPPORTED_TYPES:
        raise ValueError("selected GGUF tensor dimensions or format are unsupported")
    _format, block, block_bytes = SUPPORTED_TYPES[found["ggml_type"]]
    if cols % block:
        raise ValueError("selected GGUF tensor columns violate format alignment")
    tensor_bytes = rows * (cols // block) * block_bytes
    data_start = (reader.offset + alignment - 1) // alignment * alignment
    if found["offset"] % alignment or data_start + found["offset"] + tensor_bytes > size:
        raise ValueError("selected GGUF tensor span exceeds model file")
    return {"rows": rows, "cols": cols, "ggml_type": found["ggml_type"],
            "format": _format, "tensor_bytes": tensor_bytes}


def _metadata_hashes(settings: dict) -> dict:
    binary_fd = open_regular_file(settings["binary"])
    try:
        info = os.fstat(binary_fd)
        if not info.st_mode & 0o111:
            raise ValueError("binary has no execute bits")
        binary_hash, binary_bytes = _hash_descriptor(binary_fd)
    finally:
        os.close(binary_fd)
    models = {}
    for case in settings["cases"]:
        if case["source"]["kind"] != "gguf":
            continue
        path = Path(case["source"]["model_path"])
        fd = open_regular_file(path)
        try:
            actual, size = _hash_descriptor(fd, settings["model_limit_bytes"])
            descriptor = _gguf_tensor_descriptor(fd, size, case["source"]["tensor"])
        finally:
            os.close(fd)
        if actual != case["expected_model_sha256"].lower():
            raise ValueError("GGUF model hash does not match expected_model_sha256")
        output_bound = 11 * case["batch"] * descriptor["rows"] + 256 * 1024
        if output_bound > MAX_RESULT_BYTES:
            raise ValueError("predicted child result exceeds the 4 MiB result cap")
        models[case["name"]] = {"sha256": actual, "bytes": size, **descriptor}
        case.update(descriptor, model_bytes=size)
    predicted = _predicted_artifact_bytes(settings["cases"], settings["report_reserve_bytes"])
    if predicted > settings["artifact_bytes"]:
        raise ValueError("artifact budget cannot reserve descriptor-sized outputs and compact reports")
    return {"binary": {"sha256": binary_hash, "bytes": binary_bytes}, "models": models}


def _paired_order(repetition: int) -> tuple[str, str]:
    return (("unprofiled", "profiled") if repetition in (0, 2)
            else ("profiled", "unprofiled"))


def _validate_profile(profile: Any, branch: str, calls: int, rows: int) -> None:
    if not isinstance(profile, list) or len(profile) != calls:
        raise ValueError("profile_calls length does not match measured calls")
    nullable = {"buffer_init_ns", "input_transpose_ns", "rows_wall_ns", "output_layout_ns",
                "delegated_matvec_ns"}
    for item in profile:
        if not isinstance(item, dict) or set(item) != PROFILE_FIELDS:
            raise ValueError("profile call schema mismatch")
        if not _integer(item["schema"], minimum=1, maximum=1) or item["branch"] != branch:
            raise ValueError("profile call schema or branch mismatch")
        for field in PROFILE_FIELDS - {"schema", "branch", "workers"}:
            value = item[field]
            if field in nullable and value is None:
                continue
            if not _integer(value):
                raise ValueError("profile timing must be a nonnegative integer or allowed null")
        if not isinstance(item["workers"], list):
            raise ValueError("profile workers must be an array")
        for worker in item["workers"]:
            if not isinstance(worker, dict) or set(worker) != WORKER_FIELDS:
                raise ValueError("worker profile schema mismatch")
            if any(not _integer(worker[field]) for field in WORKER_FIELDS):
                raise ValueError("worker profile values must be nonnegative integers")
            if worker["rows"] < 1:
                raise ValueError("worker profile rows must be positive")
        if len(item["workers"]) > 2:
            raise ValueError("worker profile count exceeds the driver-owned thread count")
        if item["workers"] and sum(worker["rows"] for worker in item["workers"]) != rows:
            raise ValueError("worker profile rows do not cover the output rows")


def validate_result(value: Any, case: dict, profile: bool) -> dict:
    if not isinstance(value, dict) or set(value) != RESULT_FIELDS:
        raise ValueError("child result fields do not match schema")
    if not _integer(value["schema"], minimum=1, maximum=1) or value["status"] != "complete":
        raise ValueError("child result status or schema is invalid")
    if value["source_kind"] != case["source"]["kind"] or value["profile"] is not profile:
        raise ValueError("child source or profile mode mismatch")
    if value["activation_source"] != "synthetic_f32":
        raise ValueError("child activation source mismatch")
    if not all(_integer(value[field], minimum=1) for field in ("rows", "cols", "batch", "threads")):
        raise ValueError("child dimensions and thread count must be positive integers")
    if value["batch"] != case["batch"] or value["threads"] != 2:
        raise ValueError("child batch or driver-owned thread count mismatch")
    if case["source"]["kind"] == "synthetic":
        if value["rows"] != case["source"]["rows"] or value["cols"] != case["source"]["cols"]:
            raise ValueError("child shape does not match selected synthetic case")
        if value["ggml_type"] != case["source"]["ggml_type"]:
            raise ValueError("child GGML type does not match selected synthetic case")
        if value["model_path"] is not None or value["tensor_name"] is not None:
            raise ValueError("synthetic child unexpectedly names a model or tensor")
        if value["mapping_bytes"] != 0:
            raise ValueError("synthetic child must report zero mapping bytes")
    else:
        if value["model_path"] != case["source"]["model_path"] or value["tensor_name"] != case["source"]["tensor"]:
            raise ValueError("child model or tensor identity mismatch")
        if (value["rows"] != case.get("rows") or value["cols"] != case.get("cols")
                or value["ggml_type"] != case.get("ggml_type")
                or value["mapping_bytes"] != case.get("model_bytes")):
            raise ValueError("child shape or format differs from selected GGUF descriptor")
    if not isinstance(value["format"], str) or value["format"] not in {entry[0] for entry in SUPPORTED_TYPES.values()}:
        raise ValueError("unsupported child result format")
    if not _integer(value["mapping_bytes"]) or not _integer(value["ggml_type"], maximum=255):
        raise ValueError("invalid mapping length or GGML type")
    expected_type = value["ggml_type"]
    if expected_type not in SUPPORTED_TYPES or value["format"] != SUPPORTED_TYPES[expected_type][0]:
        raise ValueError("format and GGML type are incompatible")
    expected_branch = ("matvec" if value["batch"] == 1 else "per_input_dot" if value["batch"] <= 8
                       else "across_batch" if value["batch"] <= 31 else "across_batch_pair")
    if value["branch"] != expected_branch:
        raise ValueError("child branch does not match batch")
    for field in ("setup_ns", "estimated_heap_bytes", "output_artifact_bound_bytes"):
        if not _integer(value[field]):
            raise ValueError("invalid child numeric metadata")
    if value["estimated_heap_bytes"] == 0:
        raise ValueError("child estimated heap must be positive")
    if value["estimated_heap_bytes"] > case["input"]["buffer_limit_bytes"]:
        raise ValueError("child estimated heap exceeds configured buffer limit")
    if value["mapping_bytes"] > case["input"]["model_limit_bytes"]:
        raise ValueError("child mapping exceeds configured model limit")
    if not _integer(value["cols"], minimum=1) or value["cols"] % SUPPORTED_TYPES[expected_type][1]:
        raise ValueError("child columns violate format alignment")
    output_count = value["batch"] * value["rows"]
    expected_bound = 11 * output_count + 256 * 1024
    if value["output_artifact_bound_bytes"] != expected_bound or expected_bound > MAX_RESULT_BYTES:
        raise ValueError("child output artifact bound is inconsistent or overlarge")
    if value["all_calls_bit_identical"] is not True:
        raise ValueError("child measured calls were not bit-identical")
    times = value["call_wall_ns"]
    bits = value["output_bits"]
    if (not isinstance(times, list) or len(times) != case["measured_calls"]
            or any(not _integer(item) for item in times)):
        raise ValueError("call_wall_ns length or values are invalid")
    if not isinstance(bits, list) or len(bits) != output_count or any(not _integer(item, maximum=0xFFFFFFFF) for item in bits):
        raise ValueError("output_bits shape or values are invalid")
    if any(not math.isfinite(struct.unpack("<f", struct.pack("<I", item))[0]) for item in bits):
        raise ValueError("child output contains a nonfinite float")
    if profile:
        _validate_profile(value["profile_calls"], value["branch"], case["measured_calls"],
                          value["rows"])
        for item, call_wall_ns in zip(value["profile_calls"], times):
            if item["call_wall_ns"] != call_wall_ns:
                raise ValueError("profile and result call wall timings differ")
            if value["batch"] == 1:
                if (item["workers"] or item["delegated_matvec_ns"] is None
                        or any(item[field] is not None for field in
                               ("buffer_init_ns", "input_transpose_ns", "rows_wall_ns", "output_layout_ns"))):
                    raise ValueError("batch-one profile stage availability is invalid")
            elif (item["delegated_matvec_ns"] is not None or item["buffer_init_ns"] is None
                  or item["rows_wall_ns"] is None or item["output_layout_ns"] is None
                  or not item["workers"]):
                raise ValueError("batched profile stage availability is invalid")
            if value["batch"] < 32 and item["input_transpose_ns"] is not None:
                raise ValueError("input transpose stage is unavailable below batch 32")
            if value["batch"] >= 32 and item["input_transpose_ns"] is None:
                raise ValueError("large batch profile lacks input transpose stage")
    elif value["profile_calls"] != []:
        raise ValueError("unprofiled result must not contain profile calls")
    return value


def summarize_samples(samples: list[dict]) -> dict:
    grouped = {}
    for sample in samples:
        key = (sample["case"], sample["mode"])
        group = grouped.setdefault(key, {"status_counts": Counter(), "repetitions": [],
                                         "by_repetition": {}})
        group["status_counts"][sample["status"]] += 1
        total = (sum(sample["call_wall_ns"])
                 if sample["status"] == "complete" and sample.get("call_wall_ns") else None)
        group["by_repetition"][sample["repetition"]] = {
            "status": sample["status"], "total_call_ns": total}
        if total is not None:
            group["repetitions"].append(total)
    output = {}
    for (case, mode), group in grouped.items():
        values = group["repetitions"]
        output.setdefault(case, {})[mode] = {
            "status_counts": dict(group["status_counts"]),
            "repetition_call_totals": [
                {"repetition": repetition, **group["by_repetition"].get(
                    repetition, {"status": "not_started", "total_call_ns": None})}
                for repetition in range(3)],
            "total_call_ns": None if not values else {
                "median": statistics.median(values), "min": min(values), "max": max(values)},
        }
    return output


def compare_pairs(samples: list[dict], exact_bit_matches: dict | None = None) -> dict:
    paired = {}
    for sample in samples:
        paired.setdefault((sample["case"], sample["repetition"]), {})[sample["mode"]] = sample
    comparisons = []
    for (case, repetition), pair in sorted(paired.items()):
        left, right = pair.get("profiled"), pair.get("unprofiled")
        pair_key = (case, repetition)
        if exact_bit_matches is None:
            bit_match = bool(left and right and "output_bits" in left and "output_bits" in right
                             and left["output_bits"] == right["output_bits"])
        else:
            bit_match = exact_bit_matches.get(pair_key)
        compatible = bool(left and right and left["status"] == right["status"] == "complete"
                          and bit_match is True
                          and left.get("rows") == right.get("rows") and left.get("cols") == right.get("cols")
                          and left.get("branch") == right.get("branch")
                          and left.get("threads") == right.get("threads")
                          and left.get("format") == right.get("format")
                          and left.get("timer_boundary") == right.get("timer_boundary"))
        comparisons.append({"case": case, "repetition": repetition, "compatible": compatible,
                            "sample_statuses": {
                                "profiled": None if left is None else left["status"],
                                "unprofiled": None if right is None else right["status"]},
                            "profiled_total_call_ns": sum(left["call_wall_ns"]) if compatible else None,
                            "unprofiled_total_call_ns": sum(right["call_wall_ns"]) if compatible else None,
                            "output_bits_match": bit_match})
    valid = [entry for entry in comparisons if entry["compatible"]]
    by_case = {}
    for entry in valid:
        by_case.setdefault(entry["case"], []).append(entry)
    case_results = {}
    for case, case_pairs in by_case.items():
        ratios = [entry["profiled_total_call_ns"] / entry["unprofiled_total_call_ns"]
                  for entry in case_pairs if entry["unprofiled_total_call_ns"] > 0]
        deltas = [entry["profiled_total_call_ns"] - entry["unprofiled_total_call_ns"]
                  for entry in case_pairs]
        median_ratio = statistics.median(ratios) if ratios else None
        if median_ratio is None:
            attribution = "unavailable"
        elif abs(median_ratio - 1.0) > 0.05 or (min(deltas) <= 0 <= max(deltas)):
            attribution = "instrumentation_perturbed_or_uncertain"
        else:
            attribution = "uncertain_within_five_percent_reporting_threshold"
        case_results[case] = {"profiled_to_unprofiled_total_call_ratio_median": median_ratio,
                              "overhead_attribution": attribution,
                              "matched_repetitions": len(case_pairs)}
    return {"pairs": comparisons, "cases": case_results,
            "ratio_policy": "outer call wall totals; matched outputs, shape, branch, threads, and timer boundary only; faster profiled calls do not establish negative overhead"}


def run_session(settings: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    started = time.monotonic()
    deadline = started + settings["session_seconds"]
    provenance = _metadata_hashes(settings)
    root = PrivateDirectory.create(output_dir, repo_root=Path(__file__).resolve().parents[2])
    samples = []
    pending_pair_bits = {}
    exact_bit_matches = {}
    stop_launching = False
    try:
        for case_index, case in enumerate(settings["cases"]):
            for repetition in range(3):
                for mode in _paired_order(repetition):
                    if stop_launching:
                        break
                    sample_index = len(samples)
                    pair_key = (case["name"], repetition)
                    mode_order = _paired_order(repetition)
                    sample = {"case": case["name"], "repetition": repetition, "mode": mode,
                              "status": "not_started", "runner_status": None,
                              "comparison_status": "not_comparable",
                              "cleanup": None, "rss_scope": None, "call_wall_ns": None,
                              "setup_ns": None, "profile_calls": None,
                              "output_bit_count": None, "output_bits_sha256": None,
                              "result_ref": None, "log_excerpt": ""}
                    samples.append(sample)
                    now = time.monotonic()
                    allowance = min(settings["wall_seconds"], deadline - now)
                    if allowance <= CHILD_CLEANUP_RESERVE_SECONDS:
                        sample["status"] = "session_timeout"
                        sample["runner_status"] = "session_timeout"
                        continue
                    folder = None
                    try:
                        folder = root.mkdir(f"sample-{sample_index:04d}")
                        child_input = dict(case["input"])
                        child_input["profile"] = mode == "profiled"
                        input_data = json.dumps(child_input, separators=(",", ":"), allow_nan=False).encode()
                        folder.write("input.json", input_data, MAX_MANIFEST_BYTES)
                        result_name = "result.json"
                        # Leave one MiB for the final JSON/Markdown reports; every child shares
                        # the remaining artifact allowance through the supervisor callback.
                        child_cap = settings["artifact_bytes"] - settings["report_reserve_bytes"]
                        result = run_child(
                            [str(settings["binary"]), "--input", str(folder.path / "input.json"),
                             "--output", str(folder.path / result_name)],
                            allowance - CHILD_CLEANUP_RESERVE_SECONDS, settings["rss_bytes"],
                            MAX_CHANNEL_BYTES,
                            artifact_size=lambda: root.size(child_cap),
                            artifact_limit_bytes=child_cap)
                        sample["runner_status"] = result.get("status")
                        sample["rss_scope"] = result.get("rss_scope")
                        cleanup = result.get("cleanup")
                        if isinstance(cleanup, dict):
                            cleanup_error = cleanup.get("error")
                            sample["cleanup"] = {
                                "success": cleanup.get("success") is True,
                                "reaped": cleanup.get("reaped") is True,
                                "error": None if cleanup_error is None else str(cleanup_error)[:256],
                            }
                        else:
                            sample["cleanup"] = None
                        sample["log_excerpt"] = (result.get("stdout", "") + result.get("stderr", ""))[-512:]
                        sample["status"] = result.get("status", "supervisor_error")
                        if not isinstance(sample["cleanup"], dict) or not sample["cleanup"].get("success"):
                            sample["status"] = "cleanup_error"
                            stop_launching = True
                        elif sample["status"] == "complete":
                            raw = folder.read(result_name, MAX_RESULT_BYTES)
                            parsed = json.loads(raw, object_pairs_hook=_duplicate_rejecting_pairs)
                            validated = validate_result(parsed, case, mode == "profiled")
                            bits = validated["output_bits"]
                            sample["result_ref"] = f"sample-{sample_index:04d}/{result_name}"
                            sample["output_bit_count"] = len(bits)
                            sample["output_bits_sha256"] = _output_bits_sha256(bits)
                            previous = pending_pair_bits.pop(pair_key, None)
                            if previous is None:
                                pending_pair_bits[pair_key] = (mode, bits)
                            else:
                                _previous_mode, previous_bits = previous
                                exact_bit_matches[pair_key] = previous_bits == bits
                            sample.update({"status": "complete", "call_wall_ns": validated["call_wall_ns"],
                                           "setup_ns": validated["setup_ns"],
                                           "profile_calls": validated["profile_calls"],
                                           "rows": validated["rows"], "cols": validated["cols"],
                                           "format": validated["format"], "branch": validated["branch"],
                                           "threads": validated["threads"],
                                           "timer_boundary": "outer_call_wall_ns"})
                            del parsed, validated, raw, bits
                    except Exception as exc:
                        sample["status"] = "result_error" if sample["runner_status"] == "complete" else "artifact_error"
                        sample["error"] = str(exc)[:256]
                        cleanup = sample.get("cleanup")
                        if not isinstance(cleanup, dict) or cleanup.get("success") is not True:
                            stop_launching = True
                    finally:
                        if folder is not None:
                            folder.close()
                        if mode_order.index(mode) == 1:
                            pending_pair_bits.pop(pair_key, None)
        cleanup_unverified = any(sample.get("runner_status") is not None and
                                 (not isinstance(sample.get("cleanup"), dict)
                                  or sample["cleanup"].get("success") is not True)
                                 for sample in samples)
        pairs = {}
        for sample in samples:
            pairs.setdefault((sample["case"], sample["repetition"]), {})[sample["mode"]] = sample
        for pair_key, pair in pairs.items():
            profiled, unprofiled = pair.get("profiled"), pair.get("unprofiled")
            if (profiled is not None and unprofiled is not None
                    and profiled["status"] == unprofiled["status"] == "complete"):
                same_work = (exact_bit_matches.get(pair_key) is True and all(
                    profiled.get(key) == unprofiled.get(key)
                    for key in ("rows", "cols", "format", "branch", "threads")))
                if not same_work:
                    profiled["status"] = unprofiled["status"] = "comparison_mismatch"
                    profiled["comparison_error"] = unprofiled["comparison_error"] = "paired output bits or work metadata differ"
                    profiled["comparison_status"] = unprofiled["comparison_status"] = "mismatch"
                else:
                    profiled["comparison_status"] = unprofiled["comparison_status"] = "matched"
        retention = None
        if not cleanup_unverified and root.size(settings["artifact_bytes"]) > settings["artifact_bytes"] - settings["report_reserve_bytes"]:
            retention = root.limit_retained_artifacts(settings["artifact_bytes"] - settings["report_reserve_bytes"])
        report = {"schema": 1, "status": "complete" if samples and all(s["status"] == "complete" for s in samples) else "partial",
                  "settings": {"revision": settings["revision"], "binary_sha256": provenance["binary"]["sha256"],
                               "binary_bytes": provenance["binary"]["bytes"], "models": provenance["models"],
                               "repetitions": 3, "wall_seconds": settings["wall_seconds"],
                               "session_seconds": settings["session_seconds"], "rss_bytes": settings["rss_bytes"],
                               "artifact_bytes": settings["artifact_bytes"], "buffer_limit_bytes": settings["buffer_limit_bytes"],
                               "model_limit_bytes": settings["model_limit_bytes"], "threads": 2,
                               "report_reserve_bytes": settings["report_reserve_bytes"],
                               "build_command_example": "Example only; not verified for this binary: CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo build --offline --release -j 1 -p mivi-model --features projection-diagnostics --example projection_measure"},
                  "samples": samples, "summary": summarize_samples(samples),
                  "comparison": compare_pairs(samples, exact_bit_matches),
                  "elapsed_wall_ns": round((time.monotonic() - started) * 1_000_000_000),
                  "content_quiescence_assumption": "The selected model and executable are assumed quiescent during measurement; preflight hashes do not freeze in-place content."}
        if retention is not None:
            report["artifact_retention"] = retention
        _persist_report(report, root, settings["artifact_bytes"])
        return report
    finally:
        root.close()


def _persist_report(report: dict, output: PrivateDirectory, artifact_cap: int) -> None:
    report_json = json.dumps(report, indent=2, allow_nan=False).encode()
    report_text = ("# Projection measurement\n\n" + f"Status: {report['status']}\n\n" +
                   "Samples:\n" + "\n".join(
                       f"- {item['case']} repetition {item['repetition']} {item['mode']}: {item['status']}"
                       for item in report["samples"]) + "\n").encode()
    if len(report_json) > MAX_RESULT_BYTES or len(report_text) > MAX_RESULT_BYTES:
        raise ValueError("report exceeds bounded report size")
    if output.size(artifact_cap) + len(report_json) + len(report_text) > artifact_cap:
        if any(isinstance(item.get("cleanup"), dict) and not item["cleanup"].get("success")
               for item in report["samples"]):
            raise ValueError("report reserve exhausted with unverified child cleanup")
        retained_budget = artifact_cap - len(report_json) - len(report_text)
        if retained_budget < 0:
            raise ValueError("artifact budget cannot hold final reports")
        retention = output.limit_retained_artifacts(retained_budget)
        report["artifact_retention"] = retention
        report_json = json.dumps(report, indent=2, allow_nan=False).encode()
        if output.size(artifact_cap) + len(report_json) + len(report_text) > artifact_cap:
            raise ValueError("artifact budget cannot retain final reports")
    output.write("report.json", report_json, MAX_RESULT_BYTES)
    output.write("report.md", report_text, MAX_RESULT_BYTES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = validate_manifest(load_manifest(args.manifest))
        validate_regular_file(settings["binary"], executable=True)
        for case in settings["cases"]:
            if case["source"]["kind"] == "gguf":
                _reject_symlink_components(Path(case["source"]["model_path"]), "model_path")
        output_path = _absolute_path(str(args.output_dir), "output-dir")
        _reject_symlink_components(output_path, "output-dir")
        if args.validate_only:
            print("manifest valid")
            return 0
        report = run_session(settings, args.output_dir)
        return 0 if report["status"] == "complete" else 2
    except Exception as exc:
        print(f"projection measurement failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
