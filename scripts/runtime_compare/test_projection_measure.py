"""Strict manifest and paired projection-driver coverage."""

import hashlib
import gc
import json
import os
import struct
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import projection_measure


class ProjectionManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.binary = self.root / "projection_measure"
        self.binary.write_bytes(b"synthetic executable placeholder")
        self.binary.chmod(0o700)
        self.output_dir = self.root / "private-output"

    def tearDown(self):
        self.temp.cleanup()

    def valid_manifest(self):
        return {
            "schema": 3,
            "binary": str(self.binary),
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "repetitions": 3,
            "wall_seconds": 20,
            "session_seconds": 180,
            "rss_bytes": 512 * 1024 * 1024,
            "artifact_bytes": 8 * 1024 * 1024,
            "buffer_limit_bytes": 8 * 1024 * 1024,
            "model_limit_bytes": 1024 * 1024 * 1024,
            "cases": [
                {"name": name, "comparison_group": "synthetic-group", "column_tile": tile,
                 "batch": 33,
                 "source": {"kind": "synthetic", "ggml_type": 0, "rows": 2, "cols": 256},
                 "warmup_calls": 0, "measured_calls": 1}
                for name, tile in (("synthetic-small", None), ("synthetic-tile-32", 32),
                                   ("synthetic-tile-64", 64), ("synthetic-tile-128", 128))
            ],
        }

    def paired_manifest(self):
        return self.valid_manifest()

    def column_manifest(self):
        return self.valid_manifest()

    def test_schema_three_accepts_baseline_and_each_column_tile(self):
        settings = projection_measure.validate_manifest(self.column_manifest())
        self.assertEqual([case["column_tile"] for case in settings["cases"]],
                         [None, 32, 64, 128])
        self.assertEqual(settings["schema"], 3)
        self.assertTrue(all(case["input"]["schema"] == 3 for case in settings["cases"]))

    def test_schema_two_and_legacy_token_tile_are_rejected(self):
        manifest = self.valid_manifest()
        manifest["schema"] = 2
        with self.assertRaisesRegex(ValueError, "schema"):
            projection_measure.validate_manifest(manifest)
        manifest = self.column_manifest()
        manifest["cases"][0]["token_tile"] = None
        with self.assertRaisesRegex(ValueError, "fields"):
            projection_measure.validate_manifest(manifest)

    def test_unknown_column_tile_is_rejected(self):
        manifest = self.column_manifest()
        manifest["cases"][1]["column_tile"] = 16
        with self.assertRaisesRegex(ValueError, "column_tile"):
            projection_measure.validate_manifest(manifest)

    def test_valid_baseline_and_explicit_column_tile_selectors_are_accepted(self):
        settings = projection_measure.validate_manifest(self.paired_manifest())
        self.assertEqual([case["column_tile"] for case in settings["cases"]],
                         [None, 32, 64, 128])

    def test_explicit_panel_group_rejects_batch_below_pair_kernel_threshold(self):
        manifest = self.paired_manifest()
        for case in manifest["cases"]:
            case["batch"] = 31
        with self.assertRaisesRegex(ValueError, "panel.*batch|batch.*panel"):
            projection_measure.validate_manifest(manifest)

    def test_explicit_panel_group_rejects_single_row(self):
        manifest = self.paired_manifest()
        for case in manifest["cases"]:
            case["source"]["rows"] = 1
        with self.assertRaisesRegex(ValueError, "panel.*rows|rows.*panel"):
            projection_measure.validate_manifest(manifest)

    def test_explicit_panel_group_rejects_columns_without_distinct_panel_boundary(self):
        manifest = self.paired_manifest()
        for case in manifest["cases"]:
            case["source"]["cols"] = 128
        with self.assertRaisesRegex(ValueError, "panel.*columns|columns.*panel"):
            projection_measure.validate_manifest(manifest)

    def test_unknown_column_tile_selector_is_rejected(self):
        manifest = self.paired_manifest()
        manifest["cases"][1]["column_tile"] = 16
        with self.assertRaisesRegex(ValueError, "column_tile"):
            projection_measure.validate_manifest(manifest)

    def test_missing_or_duplicate_baseline_and_variant_members_are_rejected(self):
        for remove in (0, 1):
            manifest = self.paired_manifest()
            manifest["cases"].pop(remove)
            with self.subTest(remove=remove), self.assertRaises(ValueError):
                projection_measure.validate_manifest(manifest)
        manifest = self.paired_manifest()
        manifest["cases"].append(dict(manifest["cases"][0], name="another-baseline"))
        with self.assertRaisesRegex(ValueError, "member|selector|group"):
            projection_measure.validate_manifest(manifest)

    def test_duplicate_variant_selectors_are_rejected(self):
        manifest = self.paired_manifest()
        manifest["cases"][2]["column_tile"] = 32
        with self.assertRaisesRegex(ValueError, "member|selector|group"):
            projection_measure.validate_manifest(manifest)

    def test_workload_mismatch_inside_group_is_rejected(self):
        for field, value in (("batch", 8), ("warmup_calls", 1), ("measured_calls", 2)):
            manifest = self.paired_manifest()
            manifest["cases"][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "match|group|settings|panel"):
                projection_measure.validate_manifest(manifest)
        manifest = self.paired_manifest()
        manifest["cases"][1]["source"]["rows"] = 4
        with self.assertRaisesRegex(ValueError, "match|group|settings"):
            projection_measure.validate_manifest(manifest)

    def test_variant_execution_alternates_around_baseline_each_repetition(self):
        settings = projection_measure.validate_manifest(self.paired_manifest())
        launches = []
        child_input_fields = []

        def child(argv, *_args, **_kwargs):
            child_input = json.loads(Path(argv[2]).read_bytes())
            child_input_fields.append(set(child_input))
            launches.append((child_input["column_tile"], child_input["profile"],
                             len(launches) // 8))
            source = child_input["source"]
            record = self.child_record(child_input["profile"], ggml_type=source["ggml_type"],
                                       rows=source["rows"], cols=source["cols"],
                                       batch=child_input["batch"])
            record.update(column_tile=child_input["column_tile"],
                          comparison_group=child_input["comparison_group"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=child):
            report = projection_measure.run_session(settings, self.output_dir)
        expected = []
        for repetition in range(3):
            tiles = ((None, 32, 64, 128) if repetition % 2 == 0
                     else (32, 64, 128, None))
            for tile in tiles:
                for mode in projection_measure._paired_order(repetition):
                    expected.append((tile, mode == "profiled", repetition))
        self.assertEqual(launches, expected)
        self.assertTrue(all(fields == {
            "schema", "comparison_group", "column_tile", "source", "batch", "threads",
            "profile", "warmup_calls", "measured_calls", "buffer_limit_bytes",
            "model_limit_bytes",
        } for fields in child_input_fields))
        self.assertEqual(report["schema"], 3)

    def test_failed_and_timed_out_group_members_are_retained_and_not_compared(self):
        settings = projection_measure.validate_manifest(self.paired_manifest())
        calls = 0

        def failed_variant(argv, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            child_input = json.loads(Path(argv[2]).read_bytes())
            if child_input["column_tile"] == 32:
                return {"status": "timeout", "stdout": "bounded", "stderr": "",
                        "returncode": -15, "elapsed_seconds": 1.0, "rss_scope": "sampled",
                        "cleanup": {"success": True, "reaped": True, "error": None}}
            source = child_input["source"]
            record = self.child_record(child_input["profile"], ggml_type=source["ggml_type"],
                                       rows=source["rows"], cols=source["cols"],
                                       batch=child_input["batch"])
            record.update(column_tile=child_input["column_tile"],
                          comparison_group=child_input["comparison_group"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=failed_variant):
            report = projection_measure.run_session(settings, self.output_dir)
        failed = [sample for sample in report["samples"]
                  if sample["case"] == "synthetic-tile-32"]
        self.assertTrue(failed)
        self.assertTrue(all(sample["status"] == "timeout" for sample in failed))
        self.assertTrue(all(not pair["compatible"] for pair in report["comparison"]["groups"]))

    def test_group_comparison_requires_every_output_bit_to_match(self):
        settings = projection_measure.validate_manifest(self.paired_manifest())
        calls = 0

        def differing_tile(argv, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            child_input = json.loads(Path(argv[2]).read_bytes())
            source = child_input["source"]
            bits = [0x40000000] * 66 if child_input["column_tile"] == 64 else None
            record = self.child_record(child_input["profile"], bits=bits,
                                       ggml_type=source["ggml_type"], rows=source["rows"],
                                       cols=source["cols"], batch=child_input["batch"])
            record.update(column_tile=child_input["column_tile"],
                          comparison_group=child_input["comparison_group"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=differing_tile):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertTrue(report["comparison"]["groups"])
        self.assertTrue(any(any(value is False for value in row["output_bits_match"].values())
                            for row in report["comparison"]["groups"]))

    def invoke_validate_only(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.valid_manifest()))
        return projection_measure.main([
            "--manifest", str(manifest), "--output-dir", str(self.output_dir), "--validate-only"
        ])

    def test_boolean_batch_rejected(self):
        manifest = self.valid_manifest()
        manifest["cases"][0]["batch"] = True
        with self.assertRaises(ValueError):
            projection_measure.validate_manifest(manifest)

    def test_bf16_manifest_result_and_session_are_supported(self):
        manifest = self.valid_manifest()
        for case in manifest["cases"]:
            case["source"]["ggml_type"] = 30
        settings = projection_measure.validate_manifest(manifest)
        record = self.child_record(False, ggml_type=30, fmt="BF16")
        projection_measure.validate_result(record, settings["cases"][0], False)
        with mock.patch.object(projection_measure, "run_child", side_effect=self.complete_child):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(all(sample["format"] == "BF16" for sample in report["samples"]))

    def test_manifest_still_rejects_boolean_and_unsupported_ggml_types(self):
        for ggml_type in (True, 15):
            manifest = self.valid_manifest()
            manifest["cases"][0]["source"]["ggml_type"] = ggml_type
            with self.subTest(ggml_type=ggml_type), self.assertRaises(ValueError):
                projection_measure.validate_manifest(manifest)

    def test_report_reserve_rejects_unbounded_call_profile_combination_prelaunch(self):
        manifest = self.valid_manifest()
        manifest["artifact_bytes"] = 64 * 1024 * 1024
        template = manifest["cases"][0]
        template["measured_calls"] = 32
        manifest["cases"] = [
            dict(template, name=f"case-{group}-{tile}", comparison_group=f"group-{group}",
                 column_tile=selector)
            for group in range(4)
            for tile, selector in (("base", None), ("32", 32), ("64", 64), ("128", 128))
        ]
        with self.assertRaisesRegex(ValueError, "report metadata bound"):
            projection_measure.validate_manifest(manifest)

    def test_validate_only_creates_no_output(self):
        with mock.patch.object(projection_measure, "run_child") as child, \
                mock.patch.object(projection_measure, "validate_regular_file", wraps=projection_measure.validate_regular_file) as validate:
            self.assertEqual(self.invoke_validate_only(), 0)
            child.assert_not_called()
            validate.assert_called_once_with(self.binary, executable=True)
            self.assertFalse(self.output_dir.exists())

    def test_validate_only_rejects_symlink_output_target(self):
        target = self.root / "real-output"
        target.mkdir(mode=0o700)
        link = self.root / "output-link"
        link.symlink_to(target, target_is_directory=True)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.valid_manifest()))
        self.assertEqual(projection_measure.main([
            "--manifest", str(manifest), "--output-dir", str(link), "--validate-only"
        ]), 2)

    def test_validate_only_rejects_symlink_model_path(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"not loaded during validation")
        link = self.root / "model-link.gguf"
        link.symlink_to(model)
        manifest_value = self.valid_manifest()
        manifest_value["cases"][0]["source"] = {
            "kind": "gguf", "model_path": str(link), "tensor": "fixture.tensor"}
        manifest_value["cases"][0]["expected_model_sha256"] = "0" * 64
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(manifest_value))
        self.assertEqual(projection_measure.main([
            "--manifest", str(manifest), "--output-dir", str(self.output_dir), "--validate-only"
        ]), 2)

    def test_profile_worker_rows_must_match_output_rows(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        case = settings["cases"][0]
        record = self.child_record(True)
        record["profile_calls"][0]["workers"][0]["rows"] = 1
        with self.assertRaises(ValueError):
            projection_measure.validate_result(record, case, True)

    def test_profile_inner_timer_may_differ_from_outer_timer(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        record = self.child_record(True)
        record["call_wall_ns"][0] = 2000

        validated = projection_measure.validate_result(record, settings["cases"][0], True)

        self.assertEqual(validated["call_wall_ns"], [2000])
        self.assertEqual(validated["profile_calls"][0]["call_wall_ns"], 1000)

    def test_profile_inner_timer_cannot_exceed_outer_timer(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        record = self.child_record(True)
        record["call_wall_ns"][0] = 999

        with self.assertRaisesRegex(ValueError, "inner profile call wall exceeds outer call wall"):
            projection_measure.validate_result(record, settings["cases"][0], True)

    def test_across_batch_profile_requires_input_transpose_at_batch_nine(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        record = self.child_record(True)
        record["profile_calls"][0]["input_transpose_ns"] = 10

        validated = projection_measure.validate_result(record, settings["cases"][0], True)

        self.assertEqual(validated["profile_calls"][0]["input_transpose_ns"], 10)

    def test_per_input_dot_profile_rejects_input_transpose(self):
        manifest = self.valid_manifest()
        settings = projection_measure.validate_manifest(manifest)
        case = dict(settings["cases"][0], batch=2, column_tile=None)
        record = self.child_record(True, batch=2)
        record["profile_calls"][0]["input_transpose_ns"] = 10

        with self.assertRaisesRegex(ValueError, "input transpose stage is"):
            projection_measure.validate_result(record, case, True)

    def test_across_batch_profile_rejects_missing_input_transpose(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        record = self.child_record(True)
        record["profile_calls"][0]["input_transpose_ns"] = None

        with self.assertRaisesRegex(ValueError, "input transpose stage is"):
            projection_measure.validate_result(record, settings["cases"][0], True)

    def test_synthetic_result_must_report_zero_mapping_bytes(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        record = self.child_record(False)
        record["mapping_bytes"] = 1
        with self.assertRaises(ValueError):
            projection_measure.validate_result(record, settings["cases"][0], False)

    def test_result_requires_actual_kernel_route_matching_selector(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        case = settings["cases"][1]
        record = self.child_record(False)
        record.update(column_tile=32, comparison_group=case["comparison_group"],
                      kernel_route="ordinary")
        with self.assertRaisesRegex(ValueError, "route"):
            projection_measure.validate_result(record, case, False)

        record["kernel_route"] = "avx2_column_panel"
        self.assertEqual(projection_measure.validate_result(record, case, False)["kernel_route"],
                         "avx2_column_panel")

    def test_gguf_descriptor_preflight_rejects_non_panel_shape_before_launch(self):
        model = self.root / "tiny.gguf"
        header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
        name = b"fixture.tensor"
        header += struct.pack("<Q", len(name)) + name
        header += struct.pack("<IQQIQ", 2, 64, 3, 0, 0)
        header += b"\0" * ((32 - len(header) % 32) % 32)
        model.write_bytes(header + b"\0" * (64 * 3 * 4))

        manifest = self.valid_manifest()
        for case in manifest["cases"]:
            case["source"] = {"kind": "gguf", "model_path": str(model),
                               "tensor": "fixture.tensor"}
            case["expected_model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        settings = projection_measure.validate_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "panel.*columns|columns.*panel"):
            projection_measure._metadata_hashes(settings)

    def test_baseline_result_requires_ordinary_kernel_route(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        case = settings["cases"][0]
        record = self.child_record(False)
        record.update(column_tile=None, comparison_group=case["comparison_group"],
                      kernel_route="avx2_column_panel")
        with self.assertRaisesRegex(ValueError, "route"):
            projection_measure.validate_result(record, case, False)

    def test_duplicate_json_keys_rejected(self):
        manifest = self.root / "duplicate.json"
        manifest.write_text('{"schema":1,"schema":1}')
        with self.assertRaises(ValueError):
            projection_measure.load_manifest(manifest)

    def test_unknown_fields_and_non_three_repetitions_rejected(self):
        for mutate in (
            lambda value: value.update(extra=1),
            lambda value: value.update(repetitions=2),
            lambda value: value["cases"][0].update(mode="profiled"),
        ):
            value = self.valid_manifest()
            mutate(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                projection_measure.validate_manifest(value)

    def test_paired_order_and_report_schema(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        with mock.patch.object(projection_measure, "run_child", side_effect=self.complete_child):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(report["schema"], 3)
        self.assertEqual([sample["mode"] for sample in report["samples"][:6]],
                         ["unprofiled", "profiled", "unprofiled", "profiled",
                          "unprofiled", "profiled"])
        self.assertTrue(all(sample["status"] == "complete" for sample in report["samples"]))
        self.assertEqual(len(report["comparison"]["pairs"]), 12)
        self.assertTrue(all(pair["compatible"] for pair in report["comparison"]["pairs"]))
        profiled = next(sample for sample in report["samples"] if sample["mode"] == "profiled")
        self.assertEqual(profiled["profile_calls"][0]["workers"][0]["decode_ns"], 2)
        self.assertEqual(profiled["setup_ns"], 12)
        self.assertIn("build_command_example", report["settings"])
        self.assertNotIn("build_command", report["settings"])

    def child_record(self, profile, *, bits=None, ggml_type=0, fmt=None,
                     rows=2, cols=256, batch=33, heap_bytes=4096):
        branch = ("matvec" if batch == 1 else "per_input_dot" if batch <= 8
                  else "across_batch" if batch <= 31 else "across_batch_pair")
        output_count = batch * rows
        if fmt is None:
            fmt = projection_measure.SUPPORTED_TYPES[ggml_type][0]
        record = {
            "schema": 3, "status": "complete", "source_kind": "synthetic",
            "model_path": None, "tensor_name": None, "mapping_bytes": 0,
            "format": fmt, "ggml_type": ggml_type, "rows": rows, "cols": cols,
            "batch": batch, "branch": branch, "threads": 2,
            "profile": profile, "comparison_group": "synthetic-group", "column_tile": None,
            "kernel_route": "ordinary",
            "activation_source": "synthetic_f32",
            "setup_ns": 12, "call_wall_ns": [1000],
            "output_bits": [0x3F800000] * output_count if bits is None else bits,
            "all_calls_bit_identical": True,
            "profile_calls": [], "estimated_heap_bytes": heap_bytes,
            "output_artifact_bound_bytes": 11 * output_count + 256 * 1024,
        }
        if profile:
            record["profile_calls"] = [{
                "schema": 1, "branch": branch, "call_wall_ns": 1000,
                "validation_ns": 20, "buffer_init_ns": 10,
                "input_transpose_ns": 10 if batch >= 9 else None, "rows_wall_ns": 800,
                "output_layout_ns": 30, "delegated_matvec_ns": None,
                "unclassified_wall_ns": 140,
                "workers": [{"scratch_init_ns": 1, "decode_ns": 2,
                             "accumulate_ns": 3, "zero_copy_ns": 4, "rows": rows}],
            }]
        return record

    def complete_child(self, argv, timeout, rss, log_cap, **kwargs):
        input_path = Path(argv[2])
        result_path = Path(argv[4])
        child_input = json.loads(input_path.read_bytes())
        source = child_input["source"]
        record = self.child_record(child_input["profile"], ggml_type=source["ggml_type"],
                                   rows=source["rows"], cols=source["cols"],
                                   batch=child_input["batch"])
        record.update(comparison_group=child_input["comparison_group"],
                      column_tile=child_input["column_tile"],
                      kernel_route=("ordinary" if child_input["column_tile"] is None
                                    else "avx2_column_panel"))
        result_path.write_text(json.dumps(record))
        result_path.chmod(0o600)
        return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                "elapsed_seconds": 0.01, "rss_scope": "linux_owned_process_tree_sampled_100ms",
                "cleanup": {"success": True, "reaped": True, "error": None}, "error": None}

    def test_large_output_session_keeps_bits_private_and_persists_compact_report(self):
        manifest = self.valid_manifest()
        manifest["artifact_bytes"] = 32 * 1024 * 1024
        manifest["buffer_limit_bytes"] = 256 * 1024 * 1024
        for case in manifest["cases"]:
            case.update(batch=32)
            case["source"].update(rows=2048)
        settings = projection_measure.validate_manifest(manifest)
        launches = []

        def large_child(argv, timeout, rss, log_cap, **kwargs):
            launches.append(argv)
            child_input = json.loads(Path(argv[2]).read_bytes())
            record = self.child_record(
                child_input["profile"], ggml_type=0, rows=2048, cols=256, batch=32,
                heap_bytes=32 * 1024 * 1024)
            record.update(comparison_group=child_input["comparison_group"],
                          column_tile=child_input["column_tile"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            result_path = Path(argv[4])
            result_path.write_text(json.dumps(record))
            result_path.chmod(0o600)
            return {"status": "complete", "stdout": "bounded log", "stderr": "",
                    "returncode": 0, "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=large_child):
            report = projection_measure.run_session(settings, self.output_dir)

        self.assertEqual(len(launches), 24)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(all(pair["compatible"] and pair["output_bits_match"]
                            for pair in report["comparison"]["pairs"]))
        self.assertEqual(report["comparison"]["cases"]["synthetic-small"]["matched_repetitions"], 3)
        expected_values = 32 * 2048
        for sample in report["samples"]:
            self.assertNotIn("output_bits", sample)
            self.assertEqual(sample["output_bit_count"], expected_values)
            self.assertEqual(len(sample["output_bits_sha256"]), 64)
            self.assertFalse(sample["result_ref"].startswith("/"))
        report_bytes = (self.output_dir / "report.json").read_bytes()
        self.assertLess(len(report_bytes), projection_measure.MAX_RESULT_BYTES)
        self.assertNotIn(b'"output_bits"', report_bytes)
        results = list(self.output_dir.glob("sample-*/result.json"))
        self.assertEqual(len(results), 24)
        self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in results))
        self.assertTrue(all(path.stat().st_size < projection_measure.MAX_RESULT_BYTES for path in results))
        total_artifacts = sum(path.stat().st_size for path in self.output_dir.rglob("*") if path.is_file())
        self.assertLessEqual(total_artifacts, manifest["artifact_bytes"])

    def test_paired_bit_mismatch_excludes_both_samples_from_summary(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        calls = 0

        def mismatch(argv, *args, **kwargs):
            nonlocal calls
            calls += 1
            child_input = json.loads(Path(argv[2]).read_bytes())
            bits = [0x40000000] * 66 if calls == 2 else None
            record = self.child_record(child_input["profile"], bits=bits)
            record.update(comparison_group=child_input["comparison_group"],
                          column_tile=child_input["column_tile"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled", "cleanup":
                    {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=mismatch):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(report["samples"][0]["status"], "comparison_mismatch")
        self.assertEqual(report["samples"][1]["status"], "comparison_mismatch")
        self.assertEqual(report["summary"]["synthetic-small"]["unprofiled"]["status_counts"],
                         {"comparison_mismatch": 1, "complete": 2})
        self.assertEqual(report["summary"]["synthetic-small"]["profiled"]["status_counts"],
                         {"comparison_mismatch": 1, "complete": 2})

    def test_unmatched_complete_samples_are_counted_but_excluded_from_medians(self):
        samples = [
            {"case": "case", "repetition": 0, "mode": "profiled",
             "status": "complete", "comparison_status": "mismatch",
             "call_wall_ns": [10, 20]},
            {"case": "case", "repetition": 0, "mode": "unprofiled",
             "status": "complete", "comparison_status": "mismatch",
             "call_wall_ns": [100, 200]},
            {"case": "case", "repetition": 1, "mode": "profiled",
             "status": "complete", "comparison_status": "matched",
             "call_wall_ns": [30]},
        ]

        summary = projection_measure.summarize_samples(samples)

        profiled = summary["case"]["profiled"]
        self.assertEqual(profiled["status_counts"], {"complete": 2})
        self.assertEqual(profiled["repetition_call_totals"][0], {
            "repetition": 0, "status": "complete", "total_call_ns": None})
        self.assertEqual(profiled["total_call_ns"], {"median": 30, "min": 30, "max": 30})
        self.assertIsNone(summary["case"]["unprofiled"]["total_call_ns"])

    def test_output_bit_arrays_are_released_at_each_pair_boundary(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        bit_refs = []
        retained_at_group_boundary = None
        retained_indexes = None
        child_calls = 0
        original_validate = projection_measure.validate_result

        class TrackedBits(list):
            pass

        def track_validated_bits(*args, **kwargs):
            validated = original_validate(*args, **kwargs)
            tracked = TrackedBits(validated["output_bits"])
            bit_refs.append(weakref.ref(tracked))
            validated["output_bits"] = tracked
            return validated

        def child(argv, *_args, **_kwargs):
            nonlocal child_calls, retained_at_group_boundary, retained_indexes
            if child_calls == 8:
                gc.collect()
                retained_at_group_boundary = sum(ref() is not None for ref in bit_refs)
                retained_indexes = [index for index, ref in enumerate(bit_refs)
                                    if ref() is not None]
            child_calls += 1
            child_input = json.loads(Path(argv[2]).read_bytes())
            record = self.child_record(child_input["profile"])
            record.update(comparison_group=child_input["comparison_group"],
                          column_tile=child_input["column_tile"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "validate_result", new=track_validated_bits), \
                mock.patch.object(projection_measure, "run_child", side_effect=child):
            report = projection_measure.run_session(settings, self.output_dir)

        self.assertEqual(child_calls, 24)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(retained_at_group_boundary, 0, retained_indexes)

    def test_nonfinite_bits_and_malformed_result_are_rejected(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        case = settings["cases"][0]
        for record in (self.child_record(False, bits=[0x7F800000] * 27),
                       dict(self.child_record(False), unexpected=1),
                       dict(self.child_record(False), schema=True)):
            with self.subTest(record=record), self.assertRaises(ValueError):
                projection_measure.validate_result(record, case, False)

    def test_supervisor_timeout_is_retained_and_reported(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        def timed_out(*_args, **_kwargs):
            return {"status": "timeout", "stdout": "bounded", "stderr": "", "returncode": -15,
                    "elapsed_seconds": 1.0, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}
        with mock.patch.object(projection_measure, "run_child", side_effect=timed_out):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(report["samples"][0]["status"], "timeout")
        self.assertEqual(report["summary"]["synthetic-small"]["unprofiled"]["status_counts"],
                         {"timeout": 3})

    def test_complete_sample_with_timed_out_partner_is_not_summarized(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        calls = 0

        def complete_then_timeout(argv, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                return {"status": "timeout", "stdout": "", "stderr": "", "returncode": -15,
                        "elapsed_seconds": 0.1, "rss_scope": "sampled",
                        "cleanup": {"success": True, "reaped": True, "error": None}}
            child_input = json.loads(Path(argv[2]).read_bytes())
            record = self.child_record(child_input["profile"])
            record.update(comparison_group=child_input["comparison_group"],
                          column_tile=child_input["column_tile"],
                          kernel_route=("ordinary" if child_input["column_tile"] is None
                                        else "avx2_column_panel"))
            Path(argv[4]).write_text(json.dumps(record))
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.01, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}

        with mock.patch.object(projection_measure, "run_child", side_effect=complete_then_timeout):
            report = projection_measure.run_session(settings, self.output_dir)

        unprofiled = report["samples"][0]
        self.assertEqual(unprofiled["status"], "complete")
        self.assertEqual(unprofiled["comparison_status"], "not_comparable")
        summary = report["summary"]["synthetic-small"]["unprofiled"]
        self.assertEqual(summary["status_counts"], {"complete": 1, "timeout": 2})
        self.assertEqual(summary["repetition_call_totals"][0], {
            "repetition": 0, "status": "complete", "total_call_ns": None})
        self.assertIsNone(summary["total_call_ns"])

    def test_comparison_report_includes_attempted_failed_pairs(self):
        samples = [
            {"case": "case", "repetition": 0, "mode": "unprofiled", "status": "timeout"},
            {"case": "case", "repetition": 0, "mode": "profiled", "status": "complete",
             "call_wall_ns": [10], "output_bits": [1], "rows": 1, "cols": 1,
             "format": "F32", "branch": "across_batch", "threads": 2,
             "timer_boundary": "outer_call_wall_ns"},
        ]
        comparison = projection_measure.compare_pairs(samples)
        self.assertEqual(len(comparison["pairs"]), 1)
        self.assertFalse(comparison["pairs"][0]["compatible"])
        self.assertEqual(comparison["pairs"][0]["sample_statuses"],
                         {"unprofiled": "timeout", "profiled": "complete"})

    def test_cleanup_failure_halts_new_child_launches(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        failed = {"status": "cleanup_error", "stdout": "", "stderr": "", "returncode": None,
                  "elapsed_seconds": 1.0, "rss_scope": "sampled",
                  "cleanup": {"success": False, "reaped": False, "error": "child remains"}}
        with mock.patch.object(projection_measure, "run_child", return_value=failed) as child:
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(report["samples"][0]["status"], "cleanup_error")

    def test_cleanup_report_error_text_is_bounded(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        result = {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                  "elapsed_seconds": 0.01, "rss_scope": "sampled",
                  "cleanup": {"success": True, "reaped": True, "error": "e" * 10000}}
        with mock.patch.object(projection_measure, "run_child", return_value=result):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(len(report["samples"][0]["cleanup"]["error"]), 256)

    def test_rss_and_artifact_supervisor_failures_are_retained(self):
        for failure in ("rss_limit", "artifact_limit"):
            with self.subTest(failure=failure):
                settings = projection_measure.validate_manifest(self.valid_manifest())
                result = {"status": failure, "stdout": "bounded", "stderr": "",
                          "returncode": -15, "elapsed_seconds": 0.2, "rss_scope": "sampled",
                          "cleanup": {"success": True, "reaped": True, "error": None}}
                with mock.patch.object(projection_measure, "run_child", return_value=result):
                    report = projection_measure.run_session(settings, self.root / f"out-{failure}")
                self.assertEqual(report["samples"][0]["status"], failure)
                self.assertEqual(report["samples"][0]["runner_status"], failure)

    def test_overlarge_child_result_is_rejected_and_reported(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        def oversized(argv, *_args, **_kwargs):
            Path(argv[4]).write_bytes(b"x" * (projection_measure.MAX_RESULT_BYTES + 1))
            Path(argv[4]).chmod(0o600)
            return {"status": "complete", "stdout": "", "stderr": "", "returncode": 0,
                    "elapsed_seconds": 0.1, "rss_scope": "sampled",
                    "cleanup": {"success": True, "reaped": True, "error": None}}
        with mock.patch.object(projection_measure, "run_child", side_effect=oversized):
            report = projection_measure.run_session(settings, self.output_dir)
        self.assertEqual(report["samples"][0]["status"], "result_error")

    def test_symlink_binary_and_output_collision_are_refused(self):
        link = self.root / "binary-link"
        link.symlink_to(self.binary)
        manifest = self.valid_manifest()
        manifest["binary"] = str(link)
        with self.assertRaises((OSError, ValueError)):
            projection_measure.validate_regular_file(Path(manifest["binary"]), executable=True)
        settings = projection_measure.validate_manifest(self.valid_manifest())
        self.output_dir.mkdir(mode=0o700)
        with mock.patch.object(projection_measure, "run_child", side_effect=self.complete_child):
            with self.assertRaises(FileExistsError):
                projection_measure.run_session(settings, self.output_dir)

    def test_report_reserve_exhaustion_is_explicit(self):
        output = projection_measure.PrivateDirectory.create(self.root / "reserve",
                                                            repo_root=Path.cwd())
        try:
            report = {"status": "partial", "samples": [], "padding": "x" * (1024 * 1024 + 32)}
            with self.assertRaisesRegex(ValueError, "cannot hold final reports"):
                projection_measure._persist_report(report, output, 1024 * 1024)
            self.assertEqual(output.size(), 0)
        finally:
            output.close()

    def test_private_session_files_have_restricted_modes(self):
        settings = projection_measure.validate_manifest(self.valid_manifest())
        with mock.patch.object(projection_measure, "run_child", side_effect=self.complete_child):
            projection_measure.run_session(settings, self.output_dir)
        self.assertEqual((self.output_dir.stat().st_mode & 0o777), 0o700)
        self.assertEqual((self.output_dir / "report.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.output_dir / "sample-0000").stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.output_dir / "sample-0000" / "input.json").stat().st_mode & 0o777, 0o600)

    def test_gguf_descriptor_preflight_uses_real_shape_without_mapping_weights(self):
        model = self.root / "tiny.gguf"
        header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
        name = b"fixture.tensor"
        header += struct.pack("<Q", len(name)) + name
        header += struct.pack("<IQQIQ", 2, 64, 3, 0, 0)
        header += b"\0" * ((32 - len(header) % 32) % 32)
        model.write_bytes(header + b"\0" * (64 * 3 * 4))
        fd = projection_measure.open_regular_file(model)
        try:
            desc = projection_measure._gguf_tensor_descriptor(fd, model.stat().st_size, "fixture.tensor")
        finally:
            os.close(fd)
        self.assertEqual((desc["rows"], desc["cols"], desc["format"]), (3, 64, "F32"))


if __name__ == "__main__":
    unittest.main()
