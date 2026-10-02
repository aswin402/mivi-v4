import os
import struct
import tempfile
import unittest
from pathlib import Path

from gguf_metadata import inspect_metadata, normalize_ids


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def entry(key, kind, data):
    return string(key) + struct.pack("<I", kind) + data


def fixture(extra=(), token_kind=8, bos_kind=7, bos_data=b"\x01"):
    tokens = struct.pack("<IQ", token_kind, 4) + b"".join(string(v) for v in ("zero", "bos", "eos", "three"))
    fields = [
        entry("general.architecture", 8, string("synthetic")),
        entry("tokenizer.ggml.tokens", 9, tokens),
        entry("tokenizer.ggml.add_bos_token", bos_kind, bos_data),
        entry("synthetic.context_length", 4, struct.pack("<I", 16)),
        *extra,
    ]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields)) + b"".join(fields)


class MetadataTests(unittest.TestCase):
    def read(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.gguf"
            path.write_bytes(data)
            return inspect_metadata(path)

    def test_reads_vocabulary_bos_eos_context_without_tensors(self):
        self.assertEqual(self.read(fixture()), {
            "vocab_size": 4, "bos_id": 1, "eos_id": 2, "add_bos": True,
            "context_length": 16, "stop_strings": ["eos"], "architecture": "synthetic",
        })

    def test_metadata_ids_resolved_after_tokens(self):
        meta = self.read(fixture([entry("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 3))]))
        self.assertEqual(meta.get("stop_strings"), ["three"])

    def test_duplicate_keys_and_nonboolean_bos_fail_closed(self):
        for extra in ([entry("tokenizer.ggml.add_bos_token", 4, struct.pack("<I", 1))],
                      [entry("general.architecture", 8, string("other"))]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.read(fixture(extra))

    def test_present_nonboolean_bos_without_duplicate_key_is_rejected(self):
        with self.assertRaises(ValueError):
            self.read(fixture(bos_kind=4, bos_data=struct.pack("<I", 1)))
        with self.assertRaises(ValueError):
            self.read(fixture(bos_data=b"\x02"))

    def test_false_bos_and_unsigned_wide_token_metadata(self):
        meta = self.read(fixture(
            [entry("tokenizer.ggml.eos_token_id", 10, struct.pack("<Q", 3))],
            bos_data=b"\x00",
        ))
        self.assertIs(meta["add_bos"], False)
        self.assertEqual(meta["stop_strings"], ["three"])

    def test_invalid_or_outside_vocabulary_metadata_ids(self):
        for kind, data in ((4, struct.pack("<I", 4)), (5, struct.pack("<i", -1)), (8, string("2"))):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read(fixture([entry("tokenizer.ggml.eos_token_id", kind, data)]))

    def test_wrong_magic_truncation_and_array_type(self):
        for data in (b"bad!" + fixture()[4:], fixture()[:-1], fixture(token_kind=4)):
            with self.subTest(data_len=len(data)), self.assertRaises(ValueError):
                self.read(data)

    def test_boolean_array_values_are_validated(self):
        def boolean_array(values):
            return entry("synthetic.flags", 9,
                         struct.pack("<IQ", 7, len(values)) + bytes(values))

        self.read(fixture([boolean_array([0, 1] * 32769)]))
        with self.assertRaises(ValueError):
            self.read(fixture([boolean_array([0, 1, 2])]))

    def test_oversized_declared_string_rejected_before_allocation(self):
        data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**63)
        with self.assertRaises(ValueError):
            self.read(data)

    def test_normalization_validates_ids_and_respects_bos(self):
        meta = {"vocab_size": 4, "bos_id": 1, "add_bos": True}
        self.assertEqual(normalize_ids([3], meta), [1, 3])
        self.assertEqual(normalize_ids([1, 3], meta), [1, 3])
        self.assertEqual(normalize_ids([3], dict(meta, add_bos=False)), [3])
        for ids in ([4], [-1], [True], [1.5], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                normalize_ids(ids, meta)

    def test_normalization_output_respects_prompt_id_limit(self):
        ids_without_bos = [3] * 4095
        ids_with_bos = [1] + [3] * 4095
        add_bos = {"vocab_size": 4, "bos_id": 1, "add_bos": True}
        no_bos = dict(add_bos, add_bos=False)

        self.assertEqual(len(normalize_ids(ids_without_bos, add_bos)), 4096)
        self.assertEqual(len(normalize_ids(ids_with_bos, add_bos)), 4096)
        with self.assertRaises(ValueError):
            normalize_ids([3] * 4096, add_bos)
        self.assertEqual(len(normalize_ids([1] + [3] * 4095, no_bos)), 4096)
        self.assertEqual(len(normalize_ids([3] * 4096, no_bos)), 4096)

    @unittest.skipUnless(os.name == "posix", "Unix non-symlink read")
    def test_symlink_and_nonregular_file_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target"
            path.write_bytes(fixture())
            link = Path(folder) / "link"
            link.symlink_to(path)
            with self.assertRaises((OSError, ValueError)):
                inspect_metadata(link)
            with self.assertRaises((OSError, ValueError)):
                inspect_metadata(Path(folder))
