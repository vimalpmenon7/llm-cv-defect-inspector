import unittest

import numpy as np

from inspector.detector import ContourDetector, severity_from_area
from inspector.synthetic import make_surface


class DetectorTests(unittest.TestCase):
    def test_clean_surface_has_no_defects(self):
        img, _ = make_surface(n_scratches=0, n_spots=0, seed=1)
        self.assertEqual(ContourDetector().detect(img), [])

    def test_finds_scratch_and_spot(self):
        img, truth = make_surface(n_scratches=1, n_spots=1, seed=4)
        dets = ContourDetector().detect(img)
        labels = {d.label for d in dets}
        self.assertIn("scratch", labels)
        self.assertIn("spot", labels)
        self.assertGreaterEqual(len(dets), len(truth))

    def test_detections_sorted_by_area(self):
        img, _ = make_surface(n_scratches=2, n_spots=2, seed=5)
        areas = [d.area_ratio for d in ContourDetector().detect(img)]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_severity_thresholds(self):
        self.assertEqual(severity_from_area(0.001), "low")
        self.assertEqual(severity_from_area(0.01), "medium")
        self.assertEqual(severity_from_area(0.05), "high")

    def test_rejects_empty_image_and_even_kernel(self):
        with self.assertRaises(ValueError):
            ContourDetector().detect(np.zeros((0, 0, 3), np.uint8))
        with self.assertRaises(ValueError):
            ContourDetector(blur_ksize=30)

    def test_confidence_in_unit_range(self):
        img, _ = make_surface(seed=7)
        for d in ContourDetector().detect(img):
            self.assertTrue(0.0 <= d.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
