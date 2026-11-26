import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import replay


class ReplayTests(unittest.TestCase):
    def test_fixture_is_reproducible(self):
        first = replay.report(replay.fixture(), "Synthetic")
        self.assertEqual(first, replay.report(replay.fixture(), "Synthetic"))
        self.assertEqual(len(first["cases"]), 84)
        self.assertEqual(len({row["scenario"] for row in first["cases"]}), 3)

    def test_flat_tariff_cost_is_identical(self):
        self.assertAlmostEqual(*replay.dispatch([.2] * 8, [.2] * 8))

    def test_optimizer_can_shift_charging(self):
        baseline, candidate = replay.dispatch([.4] * 4 + [.1] * 4, [.4] * 4 + [.1] * 4)
        self.assertLess(candidate, baseline)

    def test_duplicate_and_empty_input(self):
        row = next(replay.fixture())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            replay.report([row, row], "Test")
        with self.assertRaisesRegex(ValueError, "no records"):
            replay.report([], "Test")

    def test_external_csv_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "data.csv", root / "report.json"
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(replay.FIELDS)
                writer.writerow([1, "Test group", "Sample", .3, .4, 0])
            command = [sys.executable, str(replay.ROOT / "replay.py"), "--input", str(source), "--output", str(output)]
            request = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(request.returncode, 2)
            self.assertIn("--source is required", request.stderr)
            request = subprocess.run(command + ["--source", "External test"], check=True)
            self.assertEqual(json.loads(output.read_text())["provenance"], "External test")
            source.write_text("invalid,fields\n")
            request = subprocess.run(command + ["--source", "External test"], capture_output=True, text=True)
            self.assertEqual(request.returncode, 2)
            self.assertNotIn("Traceback", request.stderr)


if __name__ == "__main__":
    unittest.main()
