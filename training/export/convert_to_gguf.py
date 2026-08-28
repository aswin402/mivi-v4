#!/usr/bin/env python3
"""
GGUF v3 binary file exporter for Mivi-v4 / LFM2.5 models.
Converts model weights and tokenizers into quantized GGUF format.
"""

import struct
import json
import argparse
from typing import Dict, List, Any, Tuple


GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF Value Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12


class GgufWriter:
    def __init__(self, filename: str):
        self.filename = filename
        self.metadata: List[Tuple[str, int, bytes]] = []
        self.tensors: List[Dict[str, Any]] = []

    def add_uint32(self, key: str, val: int):
        data = struct.pack("<I", val)
        self.metadata.append((key, GGUF_TYPE_UINT32, data))

    def add_uint64(self, key: str, val: int):
        data = struct.pack("<Q", val)
        self.metadata.append((key, GGUF_TYPE_UINT64, data))

    def add_float32(self, key: str, val: float):
        data = struct.pack("<f", val)
        self.metadata.append((key, GGUF_TYPE_FLOAT32, data))

    def add_string(self, key: str, val: str):
        encoded = val.encode("utf-8")
        data = struct.pack("<Q", len(encoded)) + encoded
        self.metadata.append((key, GGUF_TYPE_STRING, data))

    def add_string_array(self, key: str, arr: List[str]):
        body = struct.pack("<I", GGUF_TYPE_STRING)  # Element type
        body += struct.pack("<Q", len(arr))          # Array length
        for s in arr:
            encoded = s.encode("utf-8")
            body += struct.pack("<Q", len(encoded)) + encoded
        self.metadata.append((key, GGUF_TYPE_ARRAY, body))

    def add_tensor(self, name: str, dims: List[int], ggml_type: int, raw_bytes: bytes):
        self.tensors.append({
            "name": name,
            "dims": dims,
            "type": ggml_type,
            "data": raw_bytes,
        })

    def write(self):
        with open(self.filename, "wb") as f:
            # 1. Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # 2. Metadata KV pairs
            for key, val_type, raw_val in self.metadata:
                k_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(k_bytes)))
                f.write(k_bytes)
                f.write(struct.pack("<I", val_type))
                f.write(raw_val)

            # 3. Calculate tensor offsets and write tensor infos
            alignment = 32
            current_tensor_offset = 0

            # Temporary buffer for header + tensor info
            tensor_info_bytes = bytearray()
            for t in self.tensors:
                name_bytes = t["name"].encode("utf-8")
                tensor_info_bytes += struct.pack("<Q", len(name_bytes))
                tensor_info_bytes += name_bytes
                tensor_info_bytes += struct.pack("<I", len(t["dims"]))
                for d in t["dims"]:
                    tensor_info_bytes += struct.pack("<Q", d)
                tensor_info_bytes += struct.pack("<I", t["type"])
                tensor_info_bytes += struct.pack("<Q", current_tensor_offset)

                # Compute byte size
                t_bytes_len = len(t["data"])
                # Align tensor data offset to 32 bytes
                aligned_len = (t_bytes_len + alignment - 1) & ~(alignment - 1)
                t["offset"] = current_tensor_offset
                current_tensor_offset += aligned_len

            f.write(tensor_info_bytes)

            # 4. Pad before data start
            pos = f.tell()
            pad = ((pos + alignment - 1) & ~(alignment - 1)) - pos
            if pad > 0:
                f.write(b"\x00" * pad)

            # 5. Write tensor raw data buffers
            for t in self.tensors:
                f.write(t["data"])
                pad = ((len(t["data"]) + alignment - 1) & ~(alignment - 1)) - len(t["data"])
                if pad > 0:
                    f.write(b"\x00" * pad)

        print(f"✅ Successfully wrote {len(self.tensors)} tensors to {self.filename}")


def quantize_f32_to_q8_0(float_data: List[float]) -> bytes:
    """Quantize flat f32 array into GGML Q8_0 blocks (32 floats -> 34 bytes)."""
    assert len(float_data) % 32 == 0
    out = bytearray()

    import numpy as np

    for i in range(0, len(float_data), 32):
        chunk = float_data[i : i + 32]
        max_abs = max(abs(v) for v in chunk) or 1e-6
        d = max_abs / 127.0
        # Convert scale to float16 bytes
        d_f16 = np.float16(d).tobytes()
        out.extend(d_f16)

        inv_d = 1.0 / d
        for v in chunk:
            q = int(round(v * inv_d))
            q = max(-128, min(127, q))
            out.extend(struct.pack("b", q))

    return bytes(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mivi-v4 GGUF Exporter")
    parser.add_argument("--output", type=str, default="models/mivi-model.gguf")
    args = parser.parse_args()
    print(f"Mivi GGUF Converter initialized.")
