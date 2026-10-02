import json
import hashlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from compare import (
    first_difference,
    _validate_native_record,
    _paired_order,
    _run_deadline,
    _diagnosis_prefix,
    _reference_probe_scores,
    _version_and_help,
    _validate_reference_record,
    _revision_matches,
    parse_sse,
    parse_completion_body,
    private_output_directory,
    run_child,
    run_session,
    stop_child,
    summarize,
    validate_manifest,
    CleanupFailure,
    main,
)
from test_gguf_metadata import fixture as synthetic_gguf


def native_record():
    return {
        "schema": 1,
        "status": "complete",
        "effective": {"context": 128, "tile": 64, "kv_precision": "F32",
                      "temperature": 0.0, "seed": 7, "repetition_penalty": 1.0,
                      "presence_penalty": 0.0, "frequency_penalty": 0.0,
                      "stop_tokens": ["<eos>"], "worker_threads": 2,
                      "terminal_policy": {"eos_id": 2, "suppress_first_step": True}},
        "input": {"normalized_prompt_ids": [1, 3]},
        "generation": {
            "content_ids": [4],
            "prefill_progress": {"prompt_tokens": 2, "reused_tokens": 0,
                                 "processed_tokens": 2, "outcome": "Complete"},
            "raw_capture": {"truncated": False, "counter_overflow": False},
            "delivered_capture": {"truncated": False, "counter_overflow": False},
            "returned_text_capture": {"truncated": False, "counter_overflow": False},
        },
        "timing_us": {"prefill": 2500, "first_delivered": None,
                      "first_delivered_boundary": "first_nonempty_filtered_delivery_callback"},
    }


class ReportTests(unittest.TestCase):
    def test_first_divergence_and_shortened_output(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)

    def test_timeouts_stay_in_report(self):
        result = summarize([
            {"status": "complete", "prefill_ms": 10.0},
            {"status": "complete", "prefill_ms": 20.0},
            {"status": "timeout", "prefill_ms": None},
        ])
        self.assertEqual(result["status_counts"], {"complete": 2, "timeout": 1})
        self.assertEqual(result["prefill_ms"], {"median": 15.0, "min": 10.0, "max": 20.0})

    def test_empty_successes_have_no_zero_summary(self):
        self.assertEqual(
            summarize([{"status": "startup_error", "prefill_ms": None}]),
            {"status_counts": {"startup_error": 1}, "prefill_ms": None},
        )

    def test_invalid_completed_duration_is_rejected(self):
        for value in (True, float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize([{"status": "complete", "prefill_ms": value}])

    def test_pair_order_alternates_per_case_and_deadline_is_per_engine(self):
        self.assertEqual(_paired_order(0), ("mivi", "reference"))
        self.assertEqual(_paired_order(1), ("reference", "mivi"))
        self.assertEqual(_run_deadline(100.0, 30.0, 300.0), 130.0)
        self.assertEqual(_run_deadline(290.0, 30.0, 300.0), 300.0)

    def test_diagnosis_replays_shared_prefix_and_keeps_missing_scores_unavailable(self):
        prompt, candidates = _diagnosis_prefix([1, 3], [4, 5], [4, 6], 1)
        self.assertEqual(prompt, [1, 3, 4])
        self.assertEqual(candidates, [5, 6])
        probe = {"completion_probabilities": [{"top_logprobs": [
            {"id": 5, "logprob": -0.2}, {"id": 6, "logprob": -1.3}]}]}
        self.assertEqual(_reference_probe_scores(probe, candidates), {5: -0.2, 6: -1.3})
        self.assertIsNone(_reference_probe_scores({}, candidates))
        alternate = {"probs": [{"top_logprobs": [
            {"id": 5, "logprob": -0.2}, {"id": 6, "logprob": -1.3}]}]}
        self.assertEqual(_reference_probe_scores(alternate, candidates), {5: -0.2, 6: -1.3})
        self.assertIsNone(_reference_probe_scores({"probs": [{"top_probs": []}]}, candidates))
        self.assertIsNone(_reference_probe_scores(
            {"completion_probabilities": [{"top_logprobs": [{"id": 5, "logprob": "bad"}]}]},
            candidates))

    def test_reference_revision_accepts_only_exact_or_commit_prefix(self):
        pin = "7fe450e19305b828c199d602c23a8337aaa1f03b"
        self.assertTrue(_revision_matches(pin, "llama.cpp commit: 7fe450e"))
        self.assertTrue(_revision_matches("7fe450e", "llama.cpp commit: 7fe450e19305b828"))
        self.assertFalse(_revision_matches(pin, "build id: x7fe450e19305b828"))
        self.assertFalse(_revision_matches(pin, "commit: 7fe450ff"))

    def test_pinned_completion_serializer_schema_and_props_context(self):
        settings = {"context": 128, "max_tokens": 4, "model_path": "/tmp/fixture.gguf",
                    "reference_stop_strings": ["<eos>"],
                    "model_metadata": {"vocab_size": 10}}
        # Pinned task_params::to_json fields: no cache_prompt or n_ctx here.
        record = {"stop": True, "stop_type": "limit", "tokens": [4],
                  "tokens_cached": 2,
                  "timings": {"prompt_ms": 2.5, "prompt_n": 2, "cache_n": 0},
                  "generation_settings": {"temperature": 0, "seed": 7,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "n_predict": 4,
                    "n_probs": 0, "post_sampling_probs": False,
                    "stop": ["<eos>"]}}
        props = {"default_generation_settings": {"params": {}, "n_ctx": 128},
                 "total_slots": 1, "model_path": "/tmp/fixture.gguf"}
        sample = _validate_reference_record(record, props, settings, [1, 3], 4, 0)
        self.assertEqual(sample["cached_prompt_tokens"], 0)
        self.assertEqual(sample["processed_prompt_tokens"], 2)
        self.assertEqual(sample["output_ids"], [4])
        self.assertEqual(sample["effective_context"], 128)

        warm = json.loads(json.dumps(record))
        warm["timings"]["cache_n"] = 1
        with self.assertRaises(ValueError):
            _validate_reference_record(warm, props, settings, [1, 3], 4, 0)
        wrong_context = {"default_generation_settings": {"params": {}, "n_ctx": 64},
                         "total_slots": 1, "model_path": "/tmp/fixture.gguf"}
        with self.assertRaises(ValueError):
            _validate_reference_record(record, wrong_context, settings, [1, 3], 4, 0)
        wrong_model = dict(props, model_path="/tmp/other.gguf")
        with self.assertRaises(ValueError):
            _validate_reference_record(record, wrong_model, settings, [1, 3], 4, 0)
        post_sampling = json.loads(json.dumps(record))
        post_sampling["generation_settings"]["post_sampling_probs"] = True
        with self.assertRaises(ValueError):
            _validate_reference_record(post_sampling, props, settings, [1, 3], 4, 0)

        eos = json.loads(json.dumps(record))
        eos["tokens"] = [4, 2]
        eos["stop_type"] = "eos"
        eos_settings = dict(settings, model_metadata={"vocab_size": 10, "eos_id": 2})
        eos_sample = _validate_reference_record(eos, props, eos_settings, [1, 3], 4, 0)
        self.assertEqual(eos_sample["raw_output_ids"], [4, 2])
        self.assertEqual(eos_sample["output_ids"], [4])
        self.assertEqual(eos_sample["terminal_token_id"], 2)

    def test_native_record_rejects_warm_reused_truncated_or_mismatched_result(self):
        metadata = {"vocab_size": 10, "bos_id": 1, "eos_id": 2,
                    "add_bos": True, "stop_strings": ["<eos>"]}
        settings = {"context": 128, "tile": 64, "max_tokens": 4,
                    "normalized_prompt_ids": [1, 3], "model_metadata": metadata}
        record = native_record()
        sample = _validate_native_record(record, settings)
        self.assertIsNone(sample["ttft"]["value_ms"])
        self.assertEqual(sample["prefill_ms"], 2.5)
        self.assertEqual(sample["processed_prompt_tokens"], 2)
        self.assertEqual(sample["cached_prompt_tokens"], 0)
        for mutate in (
            lambda r: r["generation"]["prefill_progress"].update(reused_tokens=1),
            lambda r: r["generation"]["prefill_progress"].update(processed_tokens=1),
            lambda r: r["generation"]["raw_capture"].update(truncated=True),
            lambda r: r["generation"].update(content_ids=[10]),
            lambda r: r["effective"].update(worker_threads=1),
            lambda r: r["effective"].update(kv_precision="F16"),
            lambda r: r["effective"].update(temperature=0.2),
            lambda r: r["input"].update(normalized_prompt_ids=[3]),
        ):
            bad = native_record()
            mutate(bad)
            with self.subTest(record=bad), self.assertRaises(ValueError):
                _validate_native_record(bad, settings)


class ProcessTests(unittest.TestCase):
    def test_timeout_retains_bounded_log_and_reaps_child(self):
        result = run_child(
            [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(5)"],
            timeout_seconds=0.15,
            rss_limit_bytes=2 * 1024**3,
            log_limit_bytes=128,
            sample_interval=0.01,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertIn("started", result["stdout"])
        self.assertTrue(result["cleanup"]["reaped"])

    def test_oversized_child_logs_are_capped_without_pipe_deadlock(self):
        result = run_child(
            [sys.executable, "-c", "import os; os.write(1, b'x' * 1000000)"],
            timeout_seconds=2,
            rss_limit_bytes=2 * 1024**3,
            log_limit_bytes=128,
            sample_interval=0.01,
        )
        self.assertEqual(result["status"], "log_limit")
        self.assertLessEqual(len(result["stdout"].encode()), 128)
        self.assertTrue(result["cleanup"]["reaped"])

    def test_rss_limit_applies_to_owned_child_tree(self):
        result = run_child(
            [sys.executable, "-c", "x=bytearray(64*1024*1024); import time; time.sleep(3)"],
            timeout_seconds=2,
            rss_limit_bytes=24 * 1024**2,
            log_limit_bytes=128,
            sample_interval=0.01,
        )
        self.assertEqual(result["status"], "rss_limit")
        self.assertIn("owned_process_tree", result["rss_scope"])

    def test_startup_failure_is_a_result_not_a_success(self):
        result = run_child(
            ["/no/such/runtime-compare-executable"],
            timeout_seconds=1,
            rss_limit_bytes=2 * 1024**3,
            log_limit_bytes=128,
        )
        self.assertEqual(result["status"], "startup_error")
        self.assertIsNone(result["elapsed_seconds"])

    def test_timeout_escalates_to_kill_and_returns_bounded_result(self):
        started = time.monotonic()
        result = run_child(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(5)"],
            timeout_seconds=0.3,
            rss_limit_bytes=2 * 1024**3,
            log_limit_bytes=128,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["cleanup"]["success"], result["cleanup"])
        self.assertLess(time.monotonic() - started, 2)

    def test_fast_child_exit_still_checks_artifact_overrun(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oversized.bin"
            result = run_child(
                [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'x'*4096)", str(path)],
                timeout_seconds=2, rss_limit_bytes=2 * 1024**3,
                log_limit_bytes=128, artifact_size=lambda: path.stat().st_size if path.exists() else 0,
                artifact_limit_bytes=64,
            )
        self.assertEqual(result["status"], "artifact_limit")
        self.assertTrue(result["cleanup"]["success"])


class OutputTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "private output contract is Unix-only")
    def test_private_output_rejects_collision_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            out = root / "result"
            out.mkdir(mode=0o700)
            with self.assertRaises((FileExistsError, ValueError)):
                private_output_directory(out)
            link = root / "link"
            link.symlink_to(out, target_is_directory=True)
            with self.assertRaises((FileExistsError, ValueError, OSError)):
                private_output_directory(link)

    @unittest.skipUnless(os.name == "posix", "descriptor-relative output is Unix-only")
    def test_private_directory_pins_parents_and_uses_private_files(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            try:
                output.write("record.json", b'{"schema":1}', 32)
                self.assertEqual(output.read("record.json", 32), b'{"schema":1}')
                with self.assertRaises(FileExistsError):
                    output.write("record.json", b"overwrite", 32)
                self.assertEqual((root / "out" / "record.json").stat().st_mode & 0o777, 0o600)
                with self.assertRaises(ValueError):
                    output.write("../escape", b"x", 8)
            finally:
                output.close()

    @unittest.skipUnless(os.name == "posix", "descriptor-relative output is Unix-only")
    def test_private_directory_rejects_symlinked_ancestor(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            real = root / "real"
            real.mkdir(mode=0o700)
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises((OSError, ValueError)):
                PrivateDirectory.create(alias / "out", repo_root=Path.cwd())

    @unittest.skipUnless(os.name == "posix", "descriptor-relative output is Unix-only")
    def test_private_directory_rejects_raw_parent_traversal(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            with self.assertRaises(ValueError):
                PrivateDirectory.create(root / "unused" / ".." / "out", repo_root=Path.cwd())

    @unittest.skipUnless(os.name == "posix", "FIFO flags are Unix-only")
    def test_bounded_manifest_reader_refuses_fifo_without_blocking(self):
        from private_io import read_bounded

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            fifo = root / "input.fifo"
            os.mkfifo(fifo, 0o600)
            hold = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
            try:
                with self.assertRaises(ValueError):
                    read_bounded(fifo, 128)
            finally:
                os.close(hold)

    @unittest.skipUnless(os.name == "posix", "private output enforcement is Unix-only")
    def test_oversized_owned_artifact_is_reclaimed_for_bounded_failure_report(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            model = root / "input.gguf"
            model.write_bytes(b"user-model")
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            output.write("owned-runaway.bin", b"x" * 4096, 8192)
            report = {"schema": 1, "status": "partial",
                      "reference": {"status": "unavailable"},
                      "samples": [{"case": "fixture", "repetition": 0,
                                   "engine_order": ["mivi"],
                                   "mivi": {"status": "artifact_limit"},
                                   "reference": {"status": "unavailable"}}]}
            try:
                from compare import _persist_report

                _persist_report(report, output, artifact_cap=2048)
                rendered = json.loads(output.read("report.json", 4096))
                self.assertEqual(rendered["status"], "artifact_limit")
                self.assertGreater(rendered["artifact_retention"]["discarded_bytes"], 0)
                self.assertEqual(rendered["samples"][0]["mivi"]["status"], "artifact_limit")
                self.assertLessEqual(output.size(), 2048)
                self.assertTrue((root / "out" / "owned-runaway.bin").exists())
                self.assertLess((root / "out" / "owned-runaway.bin").stat().st_size, 4096)
                self.assertEqual(model.read_bytes(), b"user-model")
            finally:
                output.close()


class ManifestTests(unittest.TestCase):
    def make_manifest(self, root, prompt_ids=None):
        model = root / "synthetic.gguf"
        model.write_bytes(synthetic_gguf())
        binary = Path(os.path.realpath(sys.executable))
        content = {
            "schema": 1, "model_path": str(model),
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "mivi_binary": str(binary), "reference_binary": None,
            "reference_revision": None, "context": 8, "tile": 4,
            "max_tokens": 2, "repetitions": 2, "wall_seconds": 1,
            "session_seconds": 10, "rss_bytes": 64 * 1024 * 1024,
            "artifact_bytes": 1024 * 1024,
            "cases": [{"name": "fixture", "prompt_ids": [3] if prompt_ids is None else prompt_ids}],
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(content))
        os.chmod(manifest, 0o600)
        return manifest

    def test_manifest_uses_gguf_vocabulary_and_metadata_bos_before_children(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            manifest = self.make_manifest(root)
            settings = validate_manifest(manifest)
            self.assertEqual(settings["cases"][0]["normalized_prompt_ids"], [1, 3])
            output = root / "result"
            captured = io.StringIO()
            with redirect_stdout(captured):
                self.assertEqual(main(["--manifest", str(manifest), "--output-dir", str(output),
                                       "--validate-only"]), 0)
            self.assertFalse(output.exists())
            self.assertEqual(captured.getvalue().strip(), "manifest valid")

    def test_manifest_rejects_out_of_vocab_and_bos_context_overrun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            manifest = self.make_manifest(root, [4])
            with self.assertRaises(ValueError):
                validate_manifest(manifest)
            manifest = self.make_manifest(root, [3])
            data = json.loads(manifest.read_text())
            data["context"] = 3
            data["max_tokens"] = 2
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "normalized prompt"):
                validate_manifest(manifest)

    def test_reference_pin_must_be_full_commit_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            manifest = self.make_manifest(root)
            data = json.loads(manifest.read_text())
            data["reference_binary"] = str(Path(os.path.realpath(sys.executable)))
            data["reference_revision"] = "7fe450e"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "40-character"):
                validate_manifest(manifest)

    def test_wall_and_session_budgets_meet_cleanup_minimum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            for field in ("wall_seconds", "session_seconds"):
                manifest = self.make_manifest(root)
                data = json.loads(manifest.read_text())
                data[field] = 0.5
                manifest.write_text(json.dumps(data))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "minimum"):
                    validate_manifest(manifest)

    def reference_settings(self, root):
        manifest = self.make_manifest(root)
        data = json.loads(manifest.read_text())
        data["reference_binary"] = str(Path(os.path.realpath(sys.executable)))
        data["reference_revision"] = "7fe450e19305b828c199d602c23a8337aaa1f03b"
        manifest.write_text(json.dumps(data))
        return validate_manifest(manifest)

    def test_version_and_help_cleanup_failures_stop_later_launches(self):
        from private_io import PrivateDirectory

        revision = "7fe450e19305b828c199d602c23a8337aaa1f03b"
        flags = "--threads --threads-batch --ctx-size --batch-size --ubatch-size --flash-attn --cache-type-k --cache-type-v --fit --warmup --parallel --device --host --port"
        good = {"status": "complete", "returncode": 0,
                "stdout": f"llama.cpp commit {revision}", "stderr": "",
                "cleanup": {"success": True, "reaped": True, "error": None}}
        failed = {"status": "cleanup_error", "returncode": 0, "stdout": flags,
                  "stderr": "", "cleanup": {"success": False, "reaped": True,
                                                "error": "owned group remains"}}
        for stage, outcomes, expected_calls in (("--version", [failed], 1),
                                                 ("--help", [good, failed], 2)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                os.chmod(root, 0o700)
                settings = self.reference_settings(root)
                calls = []

                def child(argv, *_args, **_kwargs):
                    calls.append(argv[1])
                    return outcomes.pop(0)

                with patch("compare.run_child", side_effect=child):
                    with self.assertRaises(CleanupFailure) as raised:
                        _version_and_help(settings, time.monotonic() + 10)
                self.assertEqual(raised.exception.stage, stage)
                self.assertEqual(len(calls), expected_calls)

                settings = self.reference_settings(root)
                output = PrivateDirectory.create(root / "run", repo_root=Path.cwd())
                try:
                    failure = CleanupFailure(stage, failed)
                    with patch("compare._version_and_help", side_effect=failure), \
                         patch("compare._native_sample") as native_launch, \
                         patch("compare._reference_sample") as reference_launch:
                        report = run_session(settings, output)
                    native_launch.assert_not_called()
                    reference_launch.assert_not_called()
                    self.assertEqual(report["reference"]["status"], "cleanup_error")
                    self.assertEqual(report["cleanup_failure"]["stage"], stage)
                    self.assertTrue(all(not sample["engine_order"] for sample in report["samples"]))
                finally:
                    output.close()

    def test_report_records_actual_engine_order_per_repetition(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            settings = self.reference_settings(root)
            settings["repetitions"] = 3
            output = PrivateDirectory.create(root / "run", repo_root=Path.cwd())
            starts = []
            observed_stops = []
            native_stops = ["<eos>", "<runtime-default>"]

            def native(*_args):
                starts.append("mivi")
                return {"engine": "mivi", "status": "complete", "prefill_ms": 1.0,
                        "output_ids": [4], "stop_strings": native_stops}

            def reference(current, *_args):
                starts.append("reference")
                observed_stops.append(current["reference_stop_strings"])
                return {"engine": "llama.cpp", "status": "complete", "prefill_ms": 2.0,
                        "output_ids": [4], "settings": {"stop": current["reference_stop_strings"]}}

            help_text = "--threads --threads-batch --ctx-size --batch-size --ubatch-size --flash-attn --cache-type-k --cache-type-v --fit --warmup --parallel --device --host --port"
            try:
                with patch("compare._version_and_help", return_value=("pinned version", help_text)), \
                     patch("compare._native_sample", side_effect=native), \
                     patch("compare._reference_sample", side_effect=reference):
                    report = run_session(settings, output)
                self.assertEqual(starts, ["mivi", "reference", "reference", "mivi", "mivi", "reference"])
                self.assertEqual(observed_stops, [native_stops] * 3)
                self.assertEqual(report["status"], "complete")
                self.assertEqual([sample["engine_order"] for sample in report["samples"]],
                                 [["mivi", "reference"], ["reference", "mivi"],
                                  ["mivi", "reference"]])
            finally:
                output.close()

    @unittest.skipUnless(os.name == "posix", "private output enforcement is Unix-only")
    def test_missing_reference_is_unavailable_not_equal(self):
        from private_io import PrivateDirectory

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            os.chmod(root, 0o700)
            settings = validate_manifest(self.make_manifest(root))
            output = PrivateDirectory.create(root / "out", repo_root=Path.cwd())
            native = {"engine": "mivi", "status": "complete", "prefill_ms": 1.0,
                      "output_ids": [4], "stop_strings": ["<eos>"]}
            try:
                with patch("compare._native_sample", return_value=native):
                    report = run_session(settings, output)
                self.assertEqual(report["reference"]["status"], "unavailable")
                self.assertEqual(report["samples"][0]["reference"]["status"], "unavailable")
                self.assertIsNone(report["samples"][0]["comparison"])
                self.assertEqual(report["status"], "partial")
            finally:
                output.close()


class HttpTests(unittest.TestCase):
    def test_sse_done_and_eof_disagreement_is_reported(self):
        self.assertEqual(parse_sse(b'data: {"tokens":[4]}\n\ndata: [DONE]\n\n') ["status"], "complete")
        self.assertEqual(parse_sse(b'data: {"tokens":[4]}\n\n')["status"], "protocol_error")
        self.assertEqual(parse_sse(b'data: [DONE]\n\n')["status"], "protocol_error")

    def test_native_completion_stop_and_eof_are_authoritative(self):
        self.assertEqual(parse_completion_body(b'{"stop":true,"tokens":[4]}')["status"], "complete")
        self.assertEqual(parse_completion_body(b'{"stop":false,"tokens":[4]}')["status"], "protocol_error")
        self.assertEqual(parse_completion_body(b'{"tokens":[4]}')["status"], "protocol_error")

    def test_http_client_rejects_redirects_and_non_loopback(self):
        from compare import post_json

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "http://example.invalid/")
                self.end_headers()

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = post_json(f"http://127.0.0.1:{server.server_port}/completion", {})
            self.assertEqual(response["status"], "http_error")
            with self.assertRaises(ValueError):
                post_json("http://example.invalid/completion", {})
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()


class CleanupTests(unittest.TestCase):
    def test_cleanup_failure_is_explicit(self):
        result = stop_child(None, process_group_id=-999999, timeout_seconds=0.01)
        self.assertFalse(result["cleanup"]["success"])
        self.assertTrue(result["cleanup"]["error"])

    def test_stop_child_refuses_a_group_not_owned_by_handle(self):
        from process_supervisor import OwnedProcess

        owned = OwnedProcess([sys.executable, "-c", "import time; time.sleep(5)"],
                             timeout_seconds=2, rss_limit_bytes=2 * 1024**3,
                             log_limit_bytes=128)
        try:
            result = stop_child(owned, process_group_id=owned.group + 100000)
            self.assertFalse(result["cleanup"]["success"])
            self.assertIsNone(owned.process.poll())
        finally:
            owned.close()

    def test_cleanup_kills_owned_descendants_even_if_leader_exits(self):
        from process_supervisor import OwnedProcess

        child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)"
        parent = ("import subprocess,sys,time; "
                  f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(.05)")
        owned = OwnedProcess([sys.executable, "-c", parent], timeout_seconds=2,
                             rss_limit_bytes=2 * 1024**3, log_limit_bytes=128)
        time.sleep(0.1)
        result = owned.close()
        self.assertTrue(result["success"], result)

    def test_owned_process_watchdog_runs_during_blocking_operation(self):
        from process_supervisor import OwnedProcess

        owned = OwnedProcess(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.25,
            rss_limit_bytes=2 * 1024**3,
            log_limit_bytes=128,
            cleanup_reserve_seconds=0.05,
        )
        started = time.monotonic()
        status = owned.run_monitored(lambda: time.sleep(2))
        cleanup = owned.close()
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(status, "timeout")
        self.assertTrue(cleanup["success"], cleanup)

    def test_artifact_watchdog_enforces_growth_during_blocking_operation(self):
        from process_supervisor import OwnedProcess

        observed_size = [0]
        owned = OwnedProcess([sys.executable, "-c", "import time; time.sleep(5)"],
                             timeout_seconds=1, rss_limit_bytes=2 * 1024**3,
                             log_limit_bytes=128, cleanup_reserve_seconds=0.1,
                             artifact_size=lambda: observed_size[0], artifact_limit_bytes=10)
        threading.Timer(0.12, lambda: observed_size.__setitem__(0, 11)).start()
        status = owned.run_monitored(lambda: time.sleep(2))
        cleanup = owned.close()
        self.assertEqual(status, "artifact_limit")
        self.assertTrue(cleanup["success"], cleanup)


if __name__ == "__main__":
    unittest.main()
