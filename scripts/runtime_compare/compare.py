#!/usr/bin/env python3
"""Bounded, private paired comparison for runtime_replay and local llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import socket
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from gguf_metadata import inspect_metadata, normalize_ids
from private_io import (PrivateDirectory, open_regular_file, read_bounded_json,
                        validate_regular_file)
from process_supervisor import OwnedProcess, run_child, stop_child


MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CHANNEL_BYTES = 64 * 1024
MAX_CONTEXT = 4096
MAX_TILE = 128
MAX_OUTPUT = 64
MAX_CASES = 256
MAX_REPETITIONS = 100
MAX_WALL_SECONDS = 180.0
MAX_SESSION_SECONDS = 2700.0
MIN_WALL_SECONDS = 1.0
MAX_RSS_BYTES = 2 * 1024**3
MAX_ARTIFACT_SETTING = MAX_ARTIFACT_BYTES
POLL_SECONDS = 0.1
RUNNER_INPUT_FIELDS = {
    "prompt_ids", "context", "tile", "max_tokens", "profile", "split_prefill",
    "teacher_forced_ids", "logit_ids",
}
MANIFEST_FIELDS = {
    "schema", "model_path", "model_sha256", "mivi_binary", "reference_binary",
    "reference_revision", "context", "tile", "max_tokens", "repetitions",
    "wall_seconds", "session_seconds", "rss_bytes", "artifact_bytes", "cases",
}


class CleanupFailure(RuntimeError):
    """A supervised probe failed to verify cleanup; callers must stop launching children."""

    def __init__(self, stage: str, result: dict):
        cleanup = result.get("cleanup") if isinstance(result, dict) else None
        self.stage = stage
        self.cleanup = cleanup if isinstance(cleanup, dict) else {
            "success": False, "reaped": False, "error": "cleanup outcome missing"}
        self.result = {"status": result.get("status", "cleanup_error"),
                       "returncode": result.get("returncode"),
                       "log_excerpt": (result.get("stdout", "") + result.get("stderr", ""))[-2048:],
                       "cleanup": self.cleanup} if isinstance(result, dict) else {
                           "status": "cleanup_error", "cleanup": self.cleanup}
        super().__init__(f"{stage} cleanup failed: {self.cleanup.get('error') or 'not verified'}")


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def summarize(samples: list[dict]) -> dict:
    counts = Counter(sample["status"] for sample in samples)
    values = [sample["prefill_ms"] for sample in samples if sample["status"] == "complete"]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("invalid completed prefill duration")
    summary = None if not values else {
        "median": statistics.median(values), "min": min(values), "max": max(values),
    }
    return {"status_counts": dict(counts), "prefill_ms": summary}


def _paired_order(repetition: int) -> tuple[str, str]:
    return ("mivi", "reference") if repetition % 2 == 0 else ("reference", "mivi")


def _run_deadline(started: float, wall_seconds: float, session_deadline: float) -> float:
    return min(started + wall_seconds, session_deadline)


def _diagnosis_prefix(prompt: list[int], left: list[int], right: list[int], index: int):
    if not _is_int(index) or index < 0 or index > min(len(left), len(right)):
        raise ValueError("invalid first-divergence position")
    shared = min(index, len(left), len(right))
    if left[:shared] != right[:shared]:
        raise ValueError("candidate outputs do not share the supplied prefix")
    candidates = []
    for output in (left, right):
        if index < len(output) and output[index] not in candidates:
            candidates.append(output[index])
    return [*prompt, *left[:shared]], candidates


def _reference_probe_scores(record: dict, candidates: list[int]) -> dict[int, float] | None:
    if not isinstance(record, dict):
        return None
    rows = _reference_probability_rows(record)
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    top = rows[0].get("top_logprobs")
    if not isinstance(top, list):
        return None
    scores = {}
    for item in top:
        if not isinstance(item, dict):
            continue
        token = item.get("id")
        score = item.get("logprob")
        if (_is_int(token) and token in candidates and isinstance(score, (int, float))
                and not isinstance(score, bool) and math.isfinite(score)):
            scores[token] = float(score)
    return scores if scores else None


def _reference_probability_rows(record: dict) -> list | None:
    # The pinned serializer emits completion_probabilities; accept README's
    # documented `probs` alias only as a typed list of top_logprobs rows.
    rows = record.get("completion_probabilities")
    if rows is None:
        rows = record.get("probs")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    top = rows[0].get("top_logprobs")
    if not isinstance(top, list):
        return None
    for item in top:
        score = item.get("logprob") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or not _is_int(item.get("id"))
                or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score)):
            return None
    return rows


def _revision_matches(expected: str, version_text: str) -> bool:
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected)
            or not isinstance(version_text, str)):
        return False
    pin = expected.lower()
    actual_tokens = re.findall(r"(?<![A-Za-z0-9])[0-9a-fA-F]{7,40}(?![A-Za-z0-9])", version_text)
    return any(pin.startswith(token.lower()) or token.lower().startswith(pin)
               for token in actual_tokens)


def _effective_reference_context(props: dict, slots: object = None) -> int | None:
    if isinstance(props, dict):
        defaults = props.get("default_generation_settings")
        if isinstance(defaults, dict) and _is_int(defaults.get("n_ctx")):
            return defaults["n_ctx"]
    if isinstance(slots, list) and slots:
        contexts = [slot.get("n_ctx") for slot in slots if isinstance(slot, dict)]
        if len(contexts) == len(slots) and contexts and all(_is_int(value) for value in contexts):
            if len(set(contexts)) == 1:
                return contexts[0]
    return None


def _validate_reference_record(record: dict, props: dict, settings: dict,
                               prompt_ids: list[int], n_predict: int, n_probs: int,
                               slots: object = None) -> dict:
    metadata = settings["model_metadata"]
    effective_context = _effective_reference_context(props, slots)
    if (effective_context != settings["context"] or not isinstance(props, dict)
            or not _is_int(props.get("total_slots")) or props.get("total_slots") != 1
            or props.get("model_path") != str(settings["model_path"])):
        raise ValueError("reference effective context, slot count, or model path mismatched")
    if not isinstance(record, dict) or record.get("stop") is not True:
        raise ValueError("reference completion stop/EOF protocol is incomplete")
    tokens = record.get("tokens")
    timing = record.get("timings")
    generation = record.get("generation_settings")
    if (not isinstance(tokens, list) or len(tokens) > n_predict
            or any(not _is_int(token) or not 0 <= token < metadata["vocab_size"] for token in tokens)):
        raise ValueError("reference output token list is invalid")
    prompt_ms = timing.get("prompt_ms") if isinstance(timing, dict) else None
    processed = timing.get("prompt_n") if isinstance(timing, dict) else None
    reused = timing.get("cache_n") if isinstance(timing, dict) else None
    if (isinstance(prompt_ms, bool) or not isinstance(prompt_ms, (int, float))
            or not math.isfinite(prompt_ms) or prompt_ms < 0
            or not _is_int(processed) or processed != len(prompt_ids)
            or not _is_int(reused) or reused != 0):
        raise ValueError("reference timing or cold-prompt counts are invalid")
    requested = {"temperature": 0, "seed": 7, "repeat_penalty": 1.0,
                 "presence_penalty": 0.0, "frequency_penalty": 0.0,
                 "n_predict": n_predict, "n_probs": n_probs,
                 "post_sampling_probs": False,
                 "stop": settings["reference_stop_strings"]}
    if (not isinstance(generation, dict)
            or any(generation.get(key) != value for key, value in requested.items())
            or any(not _is_int(generation.get(key)) for key in ("seed", "n_predict", "n_probs"))):
        raise ValueError("reference did not confirm serialized generation settings")
    stop_type = record.get("stop_type")
    content_ids = list(tokens)
    terminal_id = None
    if stop_type == "eos":
        eos_id = metadata.get("eos_id")
        if not _is_int(eos_id) or not tokens or tokens[-1] != eos_id:
            raise ValueError("reference EOS stop lacks its serialized terminal token")
        terminal_id = eos_id
        content_ids.pop()
    return {"prefill_ms": float(prompt_ms), "output_ids": content_ids,
            "raw_output_ids": list(tokens), "terminal_token_id": terminal_id,
            "processed_prompt_tokens": processed, "cached_prompt_tokens": reused,
            "timings": timing, "stop_type": stop_type,
            "settings": requested, "effective_context": effective_context}


def _validate_native_record(record: dict, settings: dict) -> dict:
    """Reject incomplete or semantically mismatched Task2 output before comparison."""
    if not isinstance(record, dict) or record.get("schema") != 1 or record.get("status") != "complete":
        raise ValueError("native runner did not report schema-1 completion")
    effective = record.get("effective")
    metadata = settings["model_metadata"]
    exact = {"context": settings["context"], "tile": settings["tile"],
             "kv_precision": "F32", "seed": 7, "worker_threads": 2}
    if (not isinstance(effective, dict)
            or any(effective.get(key) != value for key, value in exact.items())
            or any(not _is_int(effective.get(key)) for key in ("context", "tile", "seed", "worker_threads"))):
        raise ValueError("native effective context/tile/KV/seed/thread settings mismatch")
    for key, expected in (("temperature", 0.0), ("repetition_penalty", 1.0),
                          ("presence_penalty", 0.0), ("frequency_penalty", 0.0)):
        value = effective.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
            raise ValueError(f"native effective {key} mismatch")
    stops = effective.get("stop_tokens")
    if (not isinstance(stops, list) or any(not isinstance(item, str) for item in stops)
            or not set(metadata["stop_strings"]).issubset(stops)):
        raise ValueError("native effective EOS stop strings missing or invalid")
    terminal = effective.get("terminal_policy")
    if (not isinstance(terminal, dict) or not _is_int(terminal.get("eos_id"))
            or terminal.get("eos_id") != metadata["eos_id"]):
        raise ValueError("native terminal EOS policy mismatch")
    if terminal.get("suppress_first_step") is not True:
        raise ValueError("native first-step suppression policy was not confirmed")
    input_record = record.get("input")
    normalized = input_record.get("normalized_prompt_ids") if isinstance(input_record, dict) else None
    if normalized != settings["normalized_prompt_ids"]:
        raise ValueError("native normalized prompt does not match GGUF metadata")
    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("native generation record is missing")
    output_ids = generation.get("content_ids")
    if (not isinstance(output_ids, list) or len(output_ids) > settings["max_tokens"]
            or any(not _is_int(token) or not 0 <= token < metadata["vocab_size"] for token in output_ids)):
        raise ValueError("native output token IDs are invalid or outside vocabulary")
    progress = generation.get("prefill_progress")
    expected_count = len(normalized)
    if (not isinstance(progress, dict)
            or any(not _is_int(progress.get(key)) for key in ("prompt_tokens", "reused_tokens", "processed_tokens"))
            or progress.get("prompt_tokens") != expected_count
            or progress.get("reused_tokens") != 0 or progress.get("processed_tokens") != expected_count
            or progress.get("outcome") != "Complete"):
        raise ValueError("native prefill was not a complete cold prompt evaluation")
    for key in ("raw_capture", "delivered_capture", "returned_text_capture"):
        capture = generation.get(key)
        if (not isinstance(capture, dict) or capture.get("truncated") is not False
                or capture.get("counter_overflow") is not False):
            raise ValueError(f"native {key} is missing, truncated, or overflowed")
    captured_ids = generation.get("captured_content_ids")
    if captured_ids is not None:
        ids = captured_ids.get("ids") if isinstance(captured_ids, dict) else None
        if (not isinstance(ids, list) or len(ids) > MAX_OUTPUT
                or any(not _is_int(token) or not 0 <= token < metadata["vocab_size"] for token in ids)
                or captured_ids.get("truncated") is not False
                or captured_ids.get("counter_overflow") is not False):
            raise ValueError("native captured token IDs are invalid or truncated")
    timing = record.get("timing_us")
    prefill = timing.get("prefill") if isinstance(timing, dict) else None
    if not _is_int(prefill) or prefill < 0:
        raise ValueError("native prefill timing is missing or invalid")
    first_delivered = timing.get("first_delivered")
    if first_delivered is not None and (not _is_int(first_delivered) or first_delivered < 0):
        raise ValueError("native callback TTFT is invalid")
    return {
        "status": "complete", "prefill_ms": prefill / 1000.0,
        "ttft": {"value_ms": None if first_delivered is None else first_delivered / 1000.0,
                 "boundary": timing.get("first_delivered_boundary")},
        "normalized_prompt_ids": list(normalized), "output_ids": list(output_ids),
        "processed_prompt_tokens": expected_count, "cached_prompt_tokens": 0,
        "stop_strings": list(stops), "stopping_reason": generation.get("stopping_reason"),
        "first_step_suppression": True,
        "effective": {key: effective[key] for key in exact},
    }


def _read_json_bounded(path: Path, cap: int) -> Any:
    return read_bounded_json(path, cap)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_int(value: Any, low: int, high: int, field: str) -> int:
    if not _is_int(value) or not low <= value <= high:
        raise ValueError(f"invalid {field} bound")
    return value


def _bounded_seconds(value: Any, ceiling: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {field} bound")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > ceiling:
        raise ValueError(f"invalid {field} bound")
    return number


def _regular_file(path: Path, executable: bool = False) -> Path:
    try:
        return validate_regular_file(path, executable)
    except (OSError, ValueError) as exc:
        raise ValueError(f"not a regular non-symlink file: {path.name}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    fd = open_regular_file(path)
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def validate_manifest(manifest_path: Path) -> dict:
    manifest_path = _regular_file(manifest_path)
    raw = _read_json_bounded(manifest_path, MAX_MANIFEST_BYTES)
    if not isinstance(raw, dict) or set(raw) != MANIFEST_FIELDS:
        raise ValueError("manifest fields must exactly match schema 1")
    if raw["schema"] != 1 or not _is_int(raw["schema"]):
        raise ValueError("manifest schema must equal 1")
    base = manifest_path.parent

    def path_field(name: str, executable: bool = False, optional: bool = False):
        value = raw[name]
        if optional and value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid {name}")
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        return _regular_file(path, executable)

    model = path_field("model_path")
    mivi = path_field("mivi_binary", executable=True)
    reference = path_field("reference_binary", executable=True, optional=True)
    revision = raw["reference_revision"]
    if reference is not None and (not isinstance(revision, str)
                                  or not re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
        raise ValueError("reference_revision must be a 40-character commit pin")
    if reference is None and revision is not None and not isinstance(revision, str):
        raise ValueError("invalid reference_revision")
    sha = raw["model_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha):
        raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
    if _sha256(model).lower() != sha.lower():
        raise ValueError("model_sha256 does not match model_path")
    context = _bounded_int(raw["context"], 1, MAX_CONTEXT, "context")
    tile = _bounded_int(raw["tile"], 1, MAX_TILE, "tile")
    max_tokens = _bounded_int(raw["max_tokens"], 0, MAX_OUTPUT, "max_tokens")
    repetitions = _bounded_int(raw["repetitions"], 1, MAX_REPETITIONS, "repetitions")
    wall = _bounded_seconds(raw["wall_seconds"], MAX_WALL_SECONDS, "wall_seconds")
    session = _bounded_seconds(raw["session_seconds"], MAX_SESSION_SECONDS, "session_seconds")
    if wall < MIN_WALL_SECONDS or session < MIN_WALL_SECONDS:
        raise ValueError("wall/session budgets must meet the 1-second cleanup minimum")
    rss = _bounded_int(raw["rss_bytes"], 1, MAX_RSS_BYTES, "rss_bytes")
    artifacts = _bounded_int(raw["artifact_bytes"], 1, MAX_ARTIFACT_SETTING, "artifact_bytes")
    cases = raw["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise ValueError("cases must be a nonempty bounded array")
    checked_cases = []
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"name", "prompt_ids"}:
            raise ValueError("each case must contain only name and prompt_ids")
        name = case["name"]
        ids = case["prompt_ids"]
        if not isinstance(name, str) or not name or len(name) > 128 or name in names:
            raise ValueError("case names must be unique nonempty strings up to 128 characters")
        names.add(name)
        if not isinstance(ids, list) or not 1 <= len(ids) <= context:
            raise ValueError("prompt_ids must fit the configured context")
        if any(not _is_int(token) or token < 0 or token > 2**32 - 1 for token in ids):
            raise ValueError("prompt_ids must contain nonnegative u32 IDs")
        if len(ids) + max_tokens > context:
            raise ValueError("prompt and generation exceed configured context")
        checked_cases.append({"name": name, "prompt_ids": list(ids)})
    if session < wall:
        raise ValueError("session_seconds must be at least wall_seconds")
    report_reserve = (64 * 1024 + sum(4096 for _case in checked_cases) + repetitions * sum(
        8192 + len(case["name"].encode("utf-8")) for case in checked_cases))
    if report_reserve > MAX_RESULT_BYTES or report_reserve >= artifacts:
        raise ValueError("sample matrix leaves insufficient bounded space for private reports")
    # All manifest shape and numeric budgets are checked before parsing model metadata.
    metadata = inspect_metadata(model)
    normalized_cases = []
    for case in checked_cases:
        normalized = normalize_ids(case["prompt_ids"], metadata)
        if len(normalized) + max_tokens > context:
            raise ValueError(f"normalized prompt for {case['name']} exceeds configured context")
        if metadata["context_length"] is not None and context > metadata["context_length"]:
            raise ValueError("configured context exceeds model metadata context length")
        normalized_cases.append({**case, "normalized_prompt_ids": normalized})
    return {
        "schema": 1, "model_path": model, "model_sha256": sha.lower(),
        "mivi_binary": mivi, "reference_binary": reference,
        "reference_revision": revision, "context": context, "tile": tile,
        "max_tokens": max_tokens, "repetitions": repetitions,
        "wall_seconds": wall, "session_seconds": session, "rss_bytes": rss,
        "artifact_bytes": artifacts, "cases": normalized_cases, "model_metadata": metadata,
        "report_reserve_bytes": report_reserve,
    }


def private_output_directory(path: Path) -> PrivateDirectory:
    return PrivateDirectory.create(path, repo_root=Path(__file__).resolve().parents[2])


def parse_sse(data: bytes) -> dict:
    token_ids: list[int] = []
    saw_done = False
    saw_payload = False
    try:
        for line in data.decode("utf-8").splitlines():
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                saw_done = True
                continue
            if not value:
                continue
            event = json.loads(value)
            saw_payload = True
            tokens = event.get("tokens", [])
            if not isinstance(tokens, list) or any(not _is_int(item) or item < 0 for item in tokens):
                return {"status": "protocol_error", "token_ids": token_ids}
            token_ids.extend(tokens)
        if saw_done and saw_payload:
            return {"status": "complete", "token_ids": token_ids}
        return {"status": "protocol_error", "token_ids": token_ids}
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, AttributeError):
        return {"status": "protocol_error", "token_ids": token_ids}


def parse_completion_body(data: bytes) -> dict:
    """Native non-streaming /completion uses its terminal flag and complete HTTP body."""
    if len(data) > MAX_CHANNEL_BYTES:
        return {"status": "response_limit"}
    try:
        response = json.loads(data)
        if not isinstance(response, dict) or response.get("stop") is not True:
            return {"status": "protocol_error"}
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(not _is_int(token) or token < 0 for token in tokens):
            return {"status": "protocol_error"}
        return {"status": "complete", "response": response}
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return {"status": "protocol_error"}


def post_json(url: str, payload: dict, timeout_seconds: float = 10.0) -> dict:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.port is None):
        raise ValueError("HTTP requests must target an explicit loopback address")
    body = json.dumps(payload, separators=(",", ":")).encode()
    if len(body) > MAX_CHANNEL_BYTES:
        return {"status": "request_limit"}
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_seconds)
    try:
        connection.request("POST", parsed.path or "/", body=body,
                           headers={"Content-Type": "application/json", "Connection": "close"})
        response = connection.getresponse()
        data = response.read(MAX_CHANNEL_BYTES + 1)
        if response.status != 200:
            return {"status": "http_error", "http_status": response.status}
        if len(data) > MAX_CHANNEL_BYTES:
            return {"status": "response_limit"}
        if response.getheader("Content-Type", "").lower().startswith("text/event-stream"):
            return parse_sse(data)
        return parse_completion_body(data)
    except (OSError, http.client.HTTPException, json.JSONDecodeError):
        return {"status": "http_error"}
    finally:
        connection.close()


def _get_json(url: str, timeout_seconds: float = 10.0) -> dict:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.port is None):
        raise ValueError("HTTP requests must target an explicit loopback address")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout_seconds)
    try:
        connection.request("GET", parsed.path or "/", headers={"Connection": "close"})
        response = connection.getresponse()
        data = response.read(MAX_CHANNEL_BYTES + 1)
        if response.status != 200 or len(data) > MAX_CHANNEL_BYTES:
            raise ValueError("reference metadata response invalid or oversized")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("reference metadata response must be an object")
        return value
    finally:
        connection.close()


def _runner_input(ids: list[int], settings: dict, teacher_ids: list[int] | None = None,
                  logit_ids: list[int] | None = None) -> bytes:
    body = {"prompt_ids": ids, "context": settings["context"], "tile": settings["tile"],
            "max_tokens": settings["max_tokens"], "profile": False, "split_prefill": False,
            "teacher_forced_ids": teacher_ids or [], "logit_ids": logit_ids or []}
    return json.dumps(body, separators=(",", ":")).encode()


def _load_result(directory: PrivateDirectory, filename: str) -> dict:
    value = json.loads(directory.read(filename, MAX_RESULT_BYTES))
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("runner returned invalid schema")
    return value


def _native_sample(settings: dict, case: dict, directory: PrivateDirectory,
                   artifact_root: PrivateDirectory, index: int, session_deadline: float) -> dict:
    prompt_ids = case["prompt_ids"]
    filename = f"input-{index}.json"
    input_data = _runner_input(prompt_ids, settings)
    child_artifact_limit = settings["artifact_bytes"] - settings["report_reserve_bytes"]
    if artifact_root.size(child_artifact_limit) + len(input_data) > child_artifact_limit:
        return {"engine": "mivi", "status": "artifact_limit", "prefill_ms": None,
                "error": "sample input would exceed reserved session artifact budget"}
    directory.write(filename, input_data, MAX_CHANNEL_BYTES)
    input_path = directory.path / filename
    result_name = f"mivi-{index}.json"
    result_path = directory.path / result_name
    argv = [str(settings["mivi_binary"]), "--model", str(settings["model_path"]),
            "--input", str(input_path), "--output", str(result_path)]
    allowance = min(settings["wall_seconds"], session_deadline-time.monotonic())
    if allowance < MIN_WALL_SECONDS:
        return {"engine": "mivi", "status": "session_timeout", "prefill_ms": None}
    result = run_child(argv, allowance, settings["rss_bytes"], MAX_CHANNEL_BYTES,
                       artifact_size=lambda: artifact_root.size(settings["artifact_bytes"]),
                       artifact_limit_bytes=child_artifact_limit)
    sample = {"engine": "mivi", "status": result["status"], "prefill_ms": None,
              "elapsed_seconds": result["elapsed_seconds"], "cleanup": result["cleanup"],
              "rss_scope": result["rss_scope"]}
    if result["status"] == "cleanup_error" or not result["cleanup"]["success"]:
        sample["status"] = "cleanup_error"
        sample["error"] = result["cleanup"]["error"]
        return sample
    if result["status"] != "complete":
        sample["error"] = result.get("error") or result["stderr"][:2048]
        sample["log_excerpt"] = (result["stdout"] + result["stderr"])[-2048:]
        try:
            failed_record = _load_result(directory, result_name)
            sample["runner_status"] = failed_record.get("status")
            timing = failed_record.get("timing_us", {})
            value = timing.get("prefill") if isinstance(timing, dict) else None
            if _is_int(value) and value >= 0:
                sample["observed_prefill_ms"] = value / 1000.0
            sample["error"] = failed_record.get("error") or sample["error"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return sample
    try:
        record = _load_result(directory, result_name)
        validated = _validate_native_record(record, settings)
        sample.update(validated)
        sample["elapsed_seconds"] = result["elapsed_seconds"]
        sample["cleanup"] = result["cleanup"]
        sample["rss_scope"] = result["rss_scope"]
        sample["boundary_note"] = "Mivi prefill is model_prefill_only; callback TTFT unavailable when absent"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sample["status"] = "result_error"
        sample["error"] = str(exc)
    return sample


def _native_divergence_probe(settings: dict, prefix: list[int], candidates: list[int],
                             directory: PrivateDirectory, artifact_root: PrivateDirectory,
                             index: int, session_deadline: float) -> dict:
    if not candidates:
        return {"status": "unavailable", "reason": "no divergent generated token to probe"}
    if len(prefix) + 1 > settings["context"]:
        return {"status": "unavailable", "reason": "shared prefix leaves no context for a teacher-forced token"}
    probe_settings = dict(settings, max_tokens=0, normalized_prompt_ids=list(prefix))
    input_data = _runner_input(prefix, probe_settings, [candidates[0]], candidates)
    child_artifact_limit = settings["artifact_bytes"] - settings["report_reserve_bytes"]
    if artifact_root.size(child_artifact_limit) + len(input_data) > child_artifact_limit:
        return {"status": "artifact_limit", "reason": "diagnosis input exceeds artifact budget"}
    input_name = f"diagnosis-mivi-{index}.json"
    result_name = f"diagnosis-mivi-{index}-result.json"
    directory.write(input_name, input_data, MAX_CHANNEL_BYTES)
    started = time.monotonic()
    allowance = min(settings["wall_seconds"], session_deadline - started)
    if allowance < MIN_WALL_SECONDS:
        return {"status": "session_timeout"}
    result = run_child([str(settings["mivi_binary"]), "--model", str(settings["model_path"]),
                        "--input", str(directory.path / input_name),
                        "--output", str(directory.path / result_name)],
                       allowance, settings["rss_bytes"], MAX_CHANNEL_BYTES,
                       artifact_size=lambda: artifact_root.size(child_artifact_limit),
                       artifact_limit_bytes=child_artifact_limit)
    if not result["cleanup"]["success"]:
        return {"status": "cleanup_error", "cleanup": result["cleanup"]}
    if result["status"] != "complete":
        return {"status": result["status"], "error": result.get("error")}
    try:
        record = _load_result(directory, result_name)
        _validate_native_record(record, probe_settings)
        rows = record.get("teacher_forced")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("native teacher-forced score row unavailable")
        row = rows[0]
        selected = row.get("selected_logits")
        if not isinstance(selected, list):
            raise ValueError("native selected logits unavailable")
        scores = {}
        for item in selected:
            if not isinstance(item, dict):
                continue
            token, score = item.get("id"), item.get("logit")
            if (_is_int(token) and token in candidates and isinstance(score, (int, float))
                    and not isinstance(score, bool) and math.isfinite(score)):
                scores[token] = float(score)
        margin = row.get("top_margin")
        margin = float(margin) if isinstance(margin, (int, float)) and not isinstance(margin, bool) and math.isfinite(margin) else None
        return {"status": "complete", "selected_logits": scores,
                "top_token_margin": margin, "position": row.get("position"),
                "scored_token": row.get("token_id"), "elapsed_seconds": result["elapsed_seconds"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"status": "diagnosis_error", "error": str(exc)[:512]}


def _diagnose_divergence(settings: dict, case: dict, native: dict, reference: dict,
                         directory: PrivateDirectory, artifact_root: PrivateDirectory,
                         index: int, session_deadline: float, reference_version: str) -> dict | None:
    if native.get("status") != "complete" or reference.get("status") != "complete":
        return None
    divergence = first_difference(native.get("output_ids", []), reference.get("output_ids", []))
    if divergence is None:
        return None
    prefix, candidates = _diagnosis_prefix(case["normalized_prompt_ids"],
                                           native["output_ids"], reference["output_ids"], divergence)
    probe_dir = directory.mkdir("divergence-probe")
    local = dict(settings, normalized_prompt_ids=prefix,
                 reference_stop_strings=native["stop_strings"])
    mivi_probe = _native_divergence_probe(local, prefix, candidates, probe_dir, artifact_root,
                                          index, session_deadline)
    if mivi_probe["status"] == "cleanup_error":
        probe_dir.close()
        return {"status": "cleanup_error", "position": divergence,
                "mivi": mivi_probe, "reference": {"status": "not_run_cleanup_failure"}}
    reference_probe = _reference_sample(local, prefix, probe_dir, artifact_root, index,
                                        session_deadline, reference_version,
                                        probe_candidates=candidates)
    if reference_probe.get("status") == "cleanup_error":
        probe_dir.close()
        return {"status": "cleanup_error", "position": divergence,
                "mivi": mivi_probe, "reference": reference_probe,
                "raw_reference_logits": "unavailable",
                "intermediate_states": "unavailable"}
    scores = reference_probe.get("probe_scores") if reference_probe.get("status") == "complete" else None
    mivi_scores = mivi_probe.get("selected_logits") if mivi_probe.get("status") == "complete" else None
    matched = None
    if isinstance(scores, dict) and isinstance(mivi_scores, dict):
        matched = {str(token): {"mivi_logit": mivi_scores[token], "reference_logprob": scores[token]}
                   for token in candidates if token in scores and token in mivi_scores}
    reference_margin = None
    top = reference_probe.get("top_logprobs")
    if isinstance(top, list) and len(top) >= 2:
        a, b = top[0].get("logprob"), top[1].get("logprob")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in (a, b)):
            reference_margin = float(a) - float(b)
    probe_dir.close()
    return {
        "status": "complete" if mivi_probe["status"] == reference_probe.get("status") == "complete" else "partial",
        "position": divergence, "shared_prefix_token_count": len(prefix),
        "candidate_ids": candidates, "mivi": mivi_probe,
        "reference": {"status": reference_probe.get("status"), "selected_logprobs": scores,
                      "top_token_margin_logprob": reference_margin,
                      "timing_boundary": "pre_sampling_n_probs_logprob"},
        "matched_selected_scores": matched or "unavailable",
        "raw_reference_logits": "unavailable", "intermediate_states": "unavailable",
        "activation_quantization": "unavailable_no_quantized_activation_probe",
        "stop_policy": "first-step suppression differs; replayed positions follow shared prefix",
    }


def _reference_flags(help_text: str) -> list[str]:
    required = ["--threads", "--threads-batch", "--ctx-size", "--batch-size", "--ubatch-size",
                "--flash-attn", "--cache-type-k", "--cache-type-v", "--fit", "--warmup",
                "--parallel", "--device", "--host", "--port"]
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise ValueError("reference lacks required pinned settings: " + ", ".join(missing))
    return required


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference_sample(settings: dict, prompt_ids: list[int], directory: PrivateDirectory,
                      artifact_root: PrivateDirectory, index: int, session_deadline: float,
                      version_output: str, probe_candidates: list[int] | None = None) -> dict:
    sample = {"engine": "llama.cpp", "status": "startup_error", "prefill_ms": None,
              "reference_version": version_output,
              "ttft": {"value_ms": None, "boundary": "unavailable_nonstreaming_completion"},
              "boundary_note": "nonstreaming /completion does not expose network-visible first-token time"}
    server = None
    started = time.monotonic()
    allowance = min(settings["wall_seconds"], session_deadline - started)
    if allowance < MIN_WALL_SECONDS:
        sample["status"] = "session_timeout"
        return sample
    port = None
    try:
        child_artifact_limit = settings["artifact_bytes"] - settings["report_reserve_bytes"]
        if artifact_root.size(child_artifact_limit) > child_artifact_limit:
            sample["status"] = "artifact_limit"
            return sample
        port = _free_loopback_port()
        argv = [str(settings["reference_binary"]), "-m", str(settings["model_path"]),
                "--host", "127.0.0.1", "--port", str(port), "--device", "none",
                "--threads", "2", "--threads-batch", "2", "--ctx-size", str(settings["context"]),
                "--batch-size", str(settings["tile"]), "--ubatch-size", str(settings["tile"]),
                "--parallel", "1", "--cache-type-k", "f32", "--cache-type-v", "f32",
                "--flash-attn", "off", "--fit", "off", "--no-warmup"]
        server = OwnedProcess(argv, allowance, settings["rss_bytes"], MAX_CHANNEL_BYTES,
                              cleanup_reserve_seconds=min(0.4, max(0.03, allowance * 0.05)),
                              artifact_size=lambda: artifact_root.size(child_artifact_limit),
                              artifact_limit_bytes=child_artifact_limit,
                              merge_stderr=True, require_running=True)
        endpoint = f"http://127.0.0.1:{port}"
        ready = False
        while server.check() is None and time.monotonic() < server.run_deadline:
            conn = http.client.HTTPConnection("127.0.0.1", port,
                                              timeout=max(0.05, min(0.2, server.run_deadline-time.monotonic())))
            try:
                conn.request("GET", "/health")
                response = conn.getresponse()
                health_body = response.read(MAX_CHANNEL_BYTES + 1)
                if response.status == 200 and len(health_body) <= MAX_CHANNEL_BYTES:
                    ready = True
                    break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                conn.close()
            time.sleep(0.05)
        if not ready:
            sample["status"] = server.check() or "startup_error"
            sample["error"] = "reference server did not become healthy"
            return sample
        props = server.run_monitored(
            lambda: _get_json(endpoint + "/props", timeout_seconds=max(
                0.05, server.run_deadline - time.monotonic())))
        if isinstance(props, str):
            sample["status"] = props
            return sample
        defaults = props.get("default_generation_settings") if isinstance(props, dict) else None
        if (not isinstance(defaults, dict) or not isinstance(defaults.get("params"), dict)
                or not _is_int(defaults.get("n_ctx"))
                or not _is_int(props.get("total_slots")) or props.get("total_slots") != 1
                or props.get("model_path") != str(settings["model_path"])):
            sample["status"] = "settings_mismatch"
            sample["error"] = "reference /props context/slot contract invalid"
            return sample
        request_n_predict = 1 if probe_candidates is not None else settings["max_tokens"]
        request_n_probs = 64 if probe_candidates is not None else 0
        payload = {"prompt": prompt_ids, "n_predict": request_n_predict,
                   "temperature": 0, "seed": 7, "repeat_penalty": 1.0,
                   "frequency_penalty": 0.0, "presence_penalty": 0.0,
                   "stop": settings["reference_stop_strings"],
                   "cache_prompt": False, "return_tokens": True,
                   "timings_per_token": True, "n_probs": request_n_probs,
                   "post_sampling_probs": False, "stream": False}
        http_timeout = max(0.05, server.run_deadline - time.monotonic())
        response = server.run_monitored(
            lambda: post_json(endpoint + "/completion", payload, timeout_seconds=http_timeout))
        if isinstance(response, str):
            sample["status"] = response
            return sample
        if not isinstance(response, dict) or response.get("status") != "complete":
            sample["status"] = response.get("status", "protocol_error") if isinstance(response, dict) else "protocol_error"
            sample["error"] = "reference completion response invalid"
            return sample
        record = response["response"]
        try:
            validated = _validate_reference_record(record, props, settings, prompt_ids,
                                                    request_n_predict, request_n_probs)
        except ValueError as exc:
            sample["status"] = "settings_mismatch" if "context" in str(exc) or "settings" in str(exc) else "protocol_error"
            sample["error"] = str(exc)[:512]
            return sample
        sample.update({"status": "complete", **validated,
                       "first_step_suppression": "unmatchable_mivi_suppresses_first_step"})
        if probe_candidates is not None:
            sample["probe_scores"] = _reference_probe_scores(record, probe_candidates)
            probabilities = _reference_probability_rows(record)
            top = probabilities[0].get("top_logprobs") if probabilities else None
            sample["top_logprobs"] = top if isinstance(top, list) else None
        return sample
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        sample["status"] = "startup_error" if server is None else (server.check() or "reference_error")
        sample["error"] = str(exc)[:2048]
        return sample
    finally:
        if server is not None:
            cleanup = server.close()
            sample["cleanup"] = cleanup
            if not cleanup["success"]:
                sample["status"] = "cleanup_error"
            elif server.status is not None:
                sample["status"] = server.status
                if sample.get("prefill_ms") is not None:
                    sample["observed_prefill_ms"] = sample["prefill_ms"]
                    sample["prefill_ms"] = None
            if sample["status"] != "complete":
                sample["log_excerpt"] = server.stdout[-2048:]
            sample["server_elapsed_seconds"] = time.monotonic() - started


def _version_and_help(settings: dict, deadline: float) -> tuple[str, str]:
    ref = str(settings["reference_binary"])
    version_allowance = min(settings["wall_seconds"], deadline-time.monotonic())
    if version_allowance < MIN_WALL_SECONDS:
        raise ValueError("session budget exhausted before reference version probe")
    version = run_child([ref, "--version"], version_allowance,
                        settings["rss_bytes"], MAX_CHANNEL_BYTES,
                        artifact_size=lambda: settings["artifact_counter"](),
                        artifact_limit_bytes=settings["artifact_bytes"] - settings["report_reserve_bytes"])
    _require_probe_cleanup("--version", version)
    if version["status"] != "complete" or version["returncode"] != 0:
        raise ValueError("reference version probe failed: " + version["status"])
    version_text = (version["stdout"] + version["stderr"]).strip()
    if not _revision_matches(settings["reference_revision"], version_text):
        raise ValueError("reference --version output does not contain pinned revision")
    help_allowance = min(settings["wall_seconds"], deadline-time.monotonic())
    if help_allowance < MIN_WALL_SECONDS:
        raise ValueError("session budget exhausted before reference help probe")
    help_result = run_child([ref, "--help"], help_allowance, settings["rss_bytes"],
                            MAX_CHANNEL_BYTES, artifact_size=lambda: settings["artifact_counter"](),
                            artifact_limit_bytes=settings["artifact_bytes"] - settings["report_reserve_bytes"])
    _require_probe_cleanup("--help", help_result)
    if help_result["status"] != "complete" or help_result["returncode"] != 0:
        raise ValueError("reference help probe failed: " + help_result["status"])
    help_text = help_result["stdout"] + help_result["stderr"]
    _reference_flags(help_text)
    return version_text[:MAX_CHANNEL_BYTES], help_text


def _require_probe_cleanup(stage: str, result: dict) -> None:
    cleanup = result.get("cleanup") if isinstance(result, dict) else None
    if (not isinstance(cleanup, dict) or cleanup.get("success") is not True
            or result.get("status") == "cleanup_error"):
        raise CleanupFailure(stage, result if isinstance(result, dict) else {})


def _compare_sample(mivi: dict, reference: dict) -> dict:
    result = {"mivi_status": mivi["status"], "reference_status": reference["status"],
              "first_difference": None, "tokens_equal": None,
              "raw_logits": "unavailable", "intermediate_states": "unavailable",
              "selected_scores": "unavailable", "top_token_margin": "unavailable",
              "stop_policy": {"mivi": "suppress_first_step=true", "reference": "not_equivalent"}}
    if mivi["status"] == "complete" and reference["status"] == "complete":
        left, right = mivi.get("output_ids", []), reference.get("output_ids", [])
        result["first_difference"] = first_difference(left, right)
        result["tokens_equal"] = result["first_difference"] is None
    return result


def _compact_engine(sample: dict) -> dict:
    keys = ("status", "prefill_ms", "observed_prefill_ms", "ttft", "elapsed_seconds",
            "processed_prompt_tokens", "cached_prompt_tokens", "effective_context",
            "cleanup", "rss_scope", "error", "log_excerpt", "runner_status",
            "stopping_reason", "first_step_suppression", "output_ids", "raw_output_ids",
            "terminal_token_id")
    compact = {key: sample[key] for key in keys if key in sample}
    for key in ("error", "log_excerpt"):
        if isinstance(compact.get(key), str):
            compact[key] = compact[key][-512:]
    return compact


def run_session(settings: dict, output_dir: PrivateDirectory) -> dict:
    session_started = time.monotonic()
    session_deadline = session_started + settings["session_seconds"]
    reference_version = None
    reference_error = None
    probe_cleanup_failure = None
    detected_flags = []
    if settings["reference_binary"] is not None:
        probe_settings = dict(settings)
        probe_settings["artifact_counter"] = lambda: output_dir.size(settings["artifact_bytes"])
        try:
            reference_version, reference_help = _version_and_help(probe_settings, session_deadline)
            detected_flags = _reference_flags(reference_help)
        except CleanupFailure as exc:
            probe_cleanup_failure = {"stage": exc.stage, "cleanup": exc.cleanup,
                                     "error": str(exc), "log_excerpt": exc.result.get("log_excerpt", "")}
            reference_error = str(exc)[:2048]
        except Exception as exc:
            reference_error = str(exc)[:2048]
    results = []
    reference_available = settings["reference_binary"] is not None and reference_error is None
    stop_session = False
    verified_native_stops = None
    diagnosed_cases = set()
    sequence = [(case, rep) for rep in range(settings["repetitions"]) for case in settings["cases"]]
    for sample_index, (case, repetition) in enumerate(sequence):
        if stop_session:
            break
        if probe_cleanup_failure is not None:
            results.append({"case": case["name"], "repetition": repetition,
                            "engine_order": [],
                            "mivi": {"status": "not_run_cleanup_failure", "prefill_ms": None},
                            "reference": {"status": "cleanup_error", "prefill_ms": None,
                                          "error": probe_cleanup_failure["error"],
                                          "probe_stage": probe_cleanup_failure["stage"],
                                          "cleanup": probe_cleanup_failure["cleanup"]},
                            "comparison": None})
            continue
        try:
            folder = output_dir.mkdir(f"sample-{sample_index:04d}")
        except Exception as exc:
            results.append({"case": case["name"], "repetition": repetition,
                            "engine_order": [],
                            "mivi": {"status": "artifact_error", "error": str(exc)[:512], "prefill_ms": None},
                            "reference": {"status": "not_run_artifact_error", "prefill_ms": None}})
            break
        local_settings = dict(settings)
        local_settings["normalized_prompt_ids"] = case["normalized_prompt_ids"]
        # Read the complete runtime policy from validated native output rather
        # than duplicate GenerationConfig defaults or guess model-family stops.
        local_settings["reference_stop_strings"] = verified_native_stops
        local_settings["artifact_counter"] = lambda: output_dir.size(settings["artifact_bytes"])
        order = _paired_order(repetition)
        attempted_order = []
        mivi = None
        reference = None
        for engine in order:
            if stop_session:
                break
            if time.monotonic() >= session_deadline:
                if engine == "mivi":
                    mivi = {"engine": "mivi", "status": "session_timeout", "prefill_ms": None}
                else:
                    reference = {"engine": "llama.cpp", "status": "session_timeout", "prefill_ms": None}
                continue
            if engine == "mivi":
                attempted_order.append("mivi")
                try:
                    mivi = _native_sample(local_settings, case, folder, output_dir,
                                          sample_index, session_deadline)
                except Exception as exc:
                    mivi = {"engine": "mivi", "status": "artifact_error", "prefill_ms": None,
                            "error": str(exc)[:512]}
                if mivi["status"] == "cleanup_error":
                    stop_session = True
                elif mivi["status"] == "complete":
                    verified_native_stops = list(mivi["stop_strings"])
                    local_settings["reference_stop_strings"] = verified_native_stops
            elif settings["reference_binary"] is None:
                reference = {"engine": "llama.cpp", "status": "unavailable", "prefill_ms": None,
                             "reason": "no local reference binary supplied"}
            elif reference_error is not None:
                reference = {"engine": "llama.cpp", "status": "probe_error", "prefill_ms": None,
                             "error": reference_error}
            elif verified_native_stops is None:
                reference = {"engine": "llama.cpp", "status": "not_run_unverified_stop_policy",
                             "prefill_ms": None,
                             "reason": "no complete native result confirmed effective stop strings"}
            else:
                attempted_order.append("reference")
                reference = _reference_sample(local_settings, case["normalized_prompt_ids"], folder,
                                              output_dir, sample_index, session_deadline, reference_version)
                if reference["status"] == "cleanup_error":
                    stop_session = True
        if mivi is None:
            mivi = {"engine": "mivi", "status": "not_run_cleanup_failure", "prefill_ms": None}
        if reference is None:
            reference = {"engine": "llama.cpp", "status": "not_run_cleanup_failure", "prefill_ms": None}
        assert mivi is not None and reference is not None
        if reference["status"] == "complete" and mivi.get("status") == "complete":
            if reference.get("settings", {}).get("stop") != mivi.get("stop_strings"):
                reference["observed_prefill_ms"] = reference.pop("prefill_ms")
                reference["prefill_ms"] = None
                reference["status"] = "settings_mismatch"
                reference["error"] = "reference stop strings did not match native effective stop strings"
        comparison = _compare_sample(mivi, reference) if reference_available else None
        if (reference_available and case["name"] not in diagnosed_cases
                and comparison is not None and comparison["first_difference"] is not None):
            diagnosed_cases.add(case["name"])
            try:
                diagnosis = _diagnose_divergence(local_settings, case, mivi, reference,
                                                 folder, output_dir, sample_index,
                                                 session_deadline, reference_version)
            except Exception as exc:
                diagnosis = {"status": "diagnosis_error", "error": str(exc)[:512],
                             "raw_reference_logits": "unavailable",
                             "intermediate_states": "unavailable"}
            comparison["diagnosis"] = diagnosis
            if diagnosis is not None and diagnosis.get("status") == "cleanup_error":
                stop_session = True
        record = {"case": case["name"], "repetition": repetition,
                  "engine_order": attempted_order,
                  "mivi": _compact_engine(mivi), "reference": _compact_engine(reference),
                  "comparison": comparison}
        results.append(record)
        folder.close()
    native_samples = [r["mivi"] for r in results]
    reference_samples = [r["reference"] for r in results]
    report = {
        "schema": 1, "status": "cleanup_error" if probe_cleanup_failure is not None else
                  ("complete" if reference_available and results and all(
            r["mivi"]["status"] == "complete" and r["reference"]["status"] == "complete" for r in results)
                  else "partial"),
        "settings": {"model_sha256": settings["model_sha256"], "context": settings["context"],
                     "tile": settings["tile"], "max_tokens": settings["max_tokens"],
                     "repetitions": settings["repetitions"], "wall_seconds": settings["wall_seconds"],
                     "session_seconds": settings["session_seconds"], "rss_bytes": settings["rss_bytes"],
                     "artifact_bytes": settings["artifact_bytes"], "threads": 2,
                     "reference_revision": settings["reference_revision"],
                     "reference_version_output": reference_version, "reference_flags": detected_flags,
                     "reference_settings": {"device": "none", "threads": 2, "threads_batch": 2,
                         "parallel": 1, "kv_k": "f32", "kv_v": "f32", "flash_attention": "off",
                         "fit": "off", "warmup": False, "batch": settings["tile"],
                         "context": settings["context"], "stop_strings": verified_native_stops,
                         "stop_policy_source": ("validated_native_effective_settings"
                                                if verified_native_stops is not None else "unavailable")}},
        "reference": {"status": "cleanup_error" if probe_cleanup_failure is not None else
                      ("available" if reference_available else
                      ("probe_error" if reference_error else "unavailable")),
                      "reason": reference_error or (None if reference_available else "no local reference binary supplied")},
        "timing_boundaries": {"mivi_prefill": "model_prefill_only",
                              "mivi_ttft": "model_callback_first_delivered_or_unavailable",
                              "reference_ttft": "unavailable_nonstreaming_completion",
                              "ratio_valid": False},
        "divergence_diagnosis": {"matched_selected_scores": "not_probed",
                                 "top_token_margin": "not_probed",
                                 "raw_logits": "unavailable_from_reference",
                                 "intermediate_states": "unavailable",
                                 "activation_quantization": "unavailable",
                                 "stop_policy": "Mivi first-step suppression cannot be matched by server request"},
        "summary": {"mivi": summarize(native_samples), "reference": summarize(reference_samples)},
        "samples": results, "elapsed_seconds": time.monotonic() - session_started,
    }
    if probe_cleanup_failure is not None:
        report["cleanup_failure"] = probe_cleanup_failure
    return report


def _markdown(report: dict) -> bytes:
    lines = ["# Runtime comparison", "", f"Status: {report['status']}",
             f"Reference: {report['reference']['status']}", "",
             "Mivi prefill is model-only. Reference TTFT is unavailable from the nonstreaming completion endpoint; no ratio is computed.", "",
             "Reference cannot expose raw logits or intermediate states; Mivi first-step suppression is unmatchable.", ""]
    for sample in report["samples"]:
        lines.append(f"- {sample['case']} repetition {sample['repetition']}: Mivi {sample['mivi']['status']}; "
                     f"reference {sample['reference']['status']}")
    lines.append("")
    return "\n".join(lines).encode()


def _persist_report(report: dict, output: PrivateDirectory, artifact_cap: int) -> None:
    def render():
        json_bytes = json.dumps(report, indent=2, allow_nan=False).encode()
        markdown_bytes = _markdown(report)
        if len(json_bytes) > MAX_RESULT_BYTES or len(markdown_bytes) > MAX_RESULT_BYTES:
            raise ValueError("report exceeds output byte limit")
        return json_bytes, markdown_bytes

    json_bytes, markdown_bytes = render()
    if output.size(artifact_cap) + len(json_bytes) + len(markdown_bytes) > artifact_cap:
        # Never trim while cleanup is unverified: a live writer could race the
        # private retention pass and its file-size accounting.
        unsafe_cleanup = report.get("cleanup_failure") is not None or any(
            sample.get(engine, {}).get("cleanup", {}).get("success") is False
            for sample in report["samples"] for engine in ("mivi", "reference"))
        unsafe_cleanup = unsafe_cleanup or any(
            isinstance(sample.get("comparison"), dict)
            and isinstance(sample["comparison"].get("diagnosis"), dict)
            and sample["comparison"]["diagnosis"].get("status") == "cleanup_error"
            for sample in report["samples"])
        if unsafe_cleanup:
            raise ValueError("artifact overrun with unverified cleanup; retention refused")
        report["status"] = "artifact_limit"
        # Reserve numeric accounting space before trimming, then render actual
        # counters. No paths, raw text, IDs or logs need to enter this summary.
        report["artifact_retention"] = {"truncated_files": 4096,
                                        "discarded_bytes": 2**64 - 1}
        json_bytes, markdown_bytes = render()
        retained_budget = artifact_cap - len(json_bytes) - len(markdown_bytes)
        if retained_budget < 0:
            raise ValueError("artifact budget cannot hold the bounded failure report")
        report["artifact_retention"] = output.limit_retained_artifacts(retained_budget)
        json_bytes, markdown_bytes = render()
    if output.size(artifact_cap) + len(json_bytes) + len(markdown_bytes) > artifact_cap:
        raise ValueError("report would exceed session artifact budget")
    output.write("report.json", json_bytes, MAX_RESULT_BYTES)
    output.write("report.md", markdown_bytes, MAX_RESULT_BYTES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = validate_manifest(args.manifest)
        if args.validate_only:
            print("manifest valid")
            return 0
        output = private_output_directory(args.output_dir)
        try:
            report = run_session(settings, output)
            _persist_report(report, output, settings["artifact_bytes"])
            return 0 if report["status"] == "complete" else 2
        finally:
            output.close()
    except Exception as exc:
        print(f"runtime comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
