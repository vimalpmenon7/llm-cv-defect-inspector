import os
import unittest
from unittest import mock

import cv2

from inspector.detector import Detection
from inspector.pipeline import analyze_image, decode_image, verdict_from
from inspector.report import template_report, write_report
from inspector.synthetic import make_surface

SCRATCH = Detection("scratch", (10, 20, 60, 6), 0.004, 0.8, "low")
BIG_SPOT = Detection("spot", (50, 50, 40, 40), 0.03, 0.9, "high")


class ReportTests(unittest.TestCase):
    def test_no_defects_report(self):
        self.assertIn("No defects", template_report([]))

    def test_report_mentions_each_detection_and_action(self):
        out = template_report([SCRATCH, BIG_SPOT])
        self.assertIn("scratch at x=10", out)
        self.assertIn("Reject or escalate", out)

    def test_plural_forms(self):
        out = template_report([SCRATCH, SCRATCH])
        self.assertIn("2 scratches", out)

    def test_write_report_falls_back_to_template_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            _, source = write_report([SCRATCH])
        self.assertEqual(source, "template")

    def test_write_report_falls_back_when_llm_fails(self):
        env = {"LLM_BASE_URL": "http://127.0.0.1:9", "LLM_MODEL": "x"}
        with mock.patch.dict(os.environ, env, clear=True):
            _, source = write_report([SCRATCH])
        self.assertEqual(source, "template")


class PipelineTests(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(verdict_from([]), "pass")
        self.assertEqual(verdict_from([SCRATCH]), "review")
        self.assertEqual(verdict_from([BIG_SPOT]), "fail")

    def test_analyze_image_end_to_end(self):
        img, _ = make_surface(seed=2)
        with mock.patch.dict(os.environ, {}, clear=True):
            result = analyze_image(img)
        self.assertTrue(result.detections)
        self.assertEqual(result.report_source, "template")
        self.assertIn(result.verdict, {"review", "fail"})
        self.assertIn("detections", result.to_dict())

    def test_decode_image_roundtrip_and_error(self):
        img, _ = make_surface(seed=4)
        ok, buf = cv2.imencode(".png", img)
        self.assertTrue(ok)
        self.assertEqual(decode_image(buf.tobytes()).shape, img.shape)
        with self.assertRaises(ValueError):
            decode_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
