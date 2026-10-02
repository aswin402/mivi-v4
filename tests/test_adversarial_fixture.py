import json
import os
import tempfile
import unittest
from pathlib import Path

from training.export.generate_adversarial_fixture import generate


class AdversarialFixtureTests(unittest.TestCase):
    def test_generation_decodes_serialized_bytes_and_matches_committed_trace(self):
        with tempfile.TemporaryDirectory(prefix="mivi-adversarial-") as temporary:
            output_dir = Path(temporary) / "fixture"
            gguf_path, trace_path = generate(output_dir)
            committed = Path(__file__).parent / "fixtures" / "hybrid_adversarial.json"

            self.assertEqual(json.loads(trace_path.read_text()), json.loads(committed.read_text()))
            self.assertTrue(gguf_path.is_file())
            if os.name == "posix":
                self.assertEqual(output_dir.stat().st_mode & 0o777, 0o700)
                self.assertEqual(gguf_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(trace_path.stat().st_mode & 0o777, 0o600)

    def test_generation_refuses_existing_output_without_overwriting(self):
        with tempfile.TemporaryDirectory(prefix="mivi-adversarial-collision-") as temporary:
            output_dir = Path(temporary) / "existing"
            output_dir.mkdir(mode=0o700)
            sentinel = output_dir / "keep.txt"
            sentinel.write_text("preserve")

            with self.assertRaisesRegex(FileExistsError, "refusing existing output directory"):
                generate(output_dir)

            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual(list(output_dir.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
