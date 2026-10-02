"""Regression coverage for post-exit bounds and private failure retention."""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare
from compare import main, _persist_report
from private_io import PrivateDirectory
from process_supervisor import run_child
import test_compare


class PostExitLimitsTests(unittest.TestCase):
    def test_reference_probes_never_launch_after_session_budget_exhaustion(self):
        settings = {"reference_binary": "/unused/reference",
                    "reference_revision": "7fe450e19305b828c199d602c23a8337aaa1f03b",
                    "wall_seconds": 10, "rss_bytes": 2 * 1024**3,
                    "artifact_bytes": 1048576, "report_reserve_bytes": 65536,
                    "artifact_counter": lambda: 0}
        completed = {"status": "complete", "returncode": 0,
                     "stdout": settings["reference_revision"], "stderr": "",
                     "cleanup": {"success": True, "reaped": True, "error": None}}
        for readings, expected_calls in (([105.0, 105.0], 0), ([100.0, 105.0], 1)):
            with self.subTest(expected_calls=expected_calls), \
                 patch("compare.time.monotonic", side_effect=readings), \
                 patch("compare.run_child", return_value=completed) as child:
                with self.assertRaisesRegex(ValueError, "session budget"):
                    compare._version_and_help(settings, 105.0)
                self.assertEqual(child.call_count, expected_calls)

    def test_retention_refuses_unverified_divergence_probe_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            try:
                output.write("probe.json", b"x" * 4096, 4096)
                report = {"schema": 1, "status": "partial",
                          "reference": {"status": "available"},
                          "samples": [{"case": "fixture", "repetition": 0,
                            "mivi": {"status": "complete"},
                            "reference": {"status": "complete"},
                            "comparison": {"diagnosis": {"status": "cleanup_error"}}}]}
                with self.assertRaisesRegex(ValueError, "unverified cleanup"):
                    _persist_report(report, output, 2048)
                self.assertEqual(output.size(), 4096)
            finally:
                output.close()

    def test_cli_retains_a_bounded_failure_report_after_artifact_overrun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            manifest = test_compare.ManifestTests().make_manifest(root)
            budget = json.loads(manifest.read_bytes())["artifact_bytes"]

            def failed_session(_settings, output):
                output.write("oversized.json", b"x" * (budget + 1), budget + 1)
                return {"schema": 1, "status": "partial",
                        "reference": {"status": "unavailable"}, "samples": [],
                        "summary": {"mivi": {"status_counts": {"artifact_limit": 1},
                                              "prefill_ms": None}}}

            with patch("compare.run_session", side_effect=failed_session):
                self.assertEqual(main(["--manifest", str(manifest),
                                       "--output-dir", str(root / "out")]), 2)
            report_path = root / "out" / "report.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_bytes())
            self.assertEqual(report["status"], "artifact_limit")
            self.assertGreater(report["artifact_retention"]["discarded_bytes"], 0)
            self.assertLessEqual(sum(p.stat().st_size for p in (root / "out").rglob("*")
                                     if p.is_file()), budget)

    def test_fast_exiting_child_is_checked_for_artifact_overrun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "large.bin"
            ready = root / "ready"
            first_check = True

            def count_artifacts():
                nonlocal first_check
                if first_check:
                    first_check = False
                    ready.touch()
                    return 0
                return artifact.stat().st_size if artifact.exists() else 0

            child = (
                "import pathlib,sys,time; "
                "ready=pathlib.Path(sys.argv[1]); "
                "exec('while not ready.exists(): time.sleep(0.001)'); "
                "pathlib.Path(sys.argv[2]).write_bytes(b'x'*131072)"
            )
            result = run_child([sys.executable, "-c", child, str(ready), str(artifact)],
                               2, 2 * 1024**3, 1024, sample_interval=0.005,
                               artifact_size=count_artifacts, artifact_limit_bytes=32)
            self.assertEqual(result["status"], "artifact_limit")
            self.assertTrue(result["cleanup"]["success"])

    def test_failure_retention_truncates_only_private_regular_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            untouched = root / "manifest.json"
            untouched.write_bytes(b"external input")
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            try:
                folder = output.mkdir("sample")
                folder.write("large.json", b"x" * 128, 128)
                folder.close()
                retention = output.limit_retained_artifacts(32)
                self.assertEqual(output.size(), 32)
                self.assertEqual(untouched.read_bytes(), b"external input")
                self.assertEqual(retention["truncated_files"], 1)
                self.assertEqual(retention["discarded_bytes"], 96)
            finally:
                output.close()

    def test_failure_retention_refuses_hardlinked_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            outside = root / "keep.bin"
            outside.write_bytes(b"x" * 128)
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            try:
                os.link(outside, output.path / "linked.bin")
                with self.assertRaises((ValueError, PermissionError)):
                    output.limit_retained_artifacts(32)
                self.assertEqual(outside.stat().st_size, 128)
            finally:
                output.close()
