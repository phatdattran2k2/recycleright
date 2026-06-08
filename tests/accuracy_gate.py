#!/usr/bin/env python3
# Copyright (c) 2026 陳發達_楊瑋竣
# Tatung University — I4210 AI實務專題

"""
Accuracy gate — reads accuracy_baseline.json and verifies current metrics are
within tolerance. Run by CI on the hosted runner (uses precomputed metrics,
no GPU needed).
"""

import json
import pathlib
import unittest

BASELINE_PATH = pathlib.Path(__file__).parent.parent / "accuracy_baseline.json"
# Current metrics are committed alongside baseline after each training run.
CURRENT_PATH  = pathlib.Path(__file__).parent.parent / "accuracy_current.json"

# Allow this many percentage points below baseline before gate fails
TOLERANCE_MAP50 = 0.03   # 3 pp


class TestAccuracyGate(unittest.TestCase):
    """Check that current model accuracy is within tolerance of baseline."""

    def setUp(self):
        if not BASELINE_PATH.exists():
            self.skipTest(f"Baseline not found at {BASELINE_PATH}")
        with open(BASELINE_PATH) as f:
            self.baseline = json.load(f)

    def test_baseline_has_required_fields(self):
        for field in ("mAP50", "mAP50_95", "precision", "recall"):
            self.assertIn(field, self.baseline, f"Baseline missing field: {field}")

    def test_map50_above_minimum(self):
        """mAP@50 should be ≥ 0.70 regardless of baseline drift."""
        self.assertGreaterEqual(
            self.baseline["mAP50"], 0.70,
            f"Baseline mAP@50 {self.baseline['mAP50']:.3f} is below minimum 0.70"
        )

    def test_current_within_tolerance(self):
        """Current metrics must not drop more than TOLERANCE_MAP50 below baseline."""
        if not CURRENT_PATH.exists():
            self.skipTest("accuracy_current.json not found — skip current vs baseline check")
        with open(CURRENT_PATH) as f:
            current = json.load(f)
        baseline_map50 = self.baseline["mAP50"]
        current_map50  = current["mAP50"]
        self.assertGreaterEqual(
            current_map50,
            baseline_map50 - TOLERANCE_MAP50,
            f"Current mAP@50 {current_map50:.3f} dropped more than "
            f"{TOLERANCE_MAP50:.2f} below baseline {baseline_map50:.3f}",
        )

    def test_four_classes_in_per_class(self):
        per_class = self.baseline.get("per_class_AP50", {})
        if per_class:
            expected = {"PET bottle", "Glass bottle", "Aluminum can", "Tetra Pak"}
            self.assertEqual(set(per_class.keys()), expected)


if __name__ == "__main__":
    unittest.main()
