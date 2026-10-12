"""Checks for timing correctness and image composition, not model accuracy."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import FrameStats, RealtimeRecorder, blend, cover, parse_args
from backgrounds import Backgrounds


class CoreTests(unittest.TestCase):
    def test_fps_is_count_over_total_not_average_of_reciprocals(self):
        stats = FrameStats()
        stats.add(0.010, 0.040)
        stats.add(0.030, 0.060)
        summary = stats.summary()
        self.assertAlmostEqual(summary["processing_fps"], 50)
        self.assertAlmostEqual(summary["application_fps"], 20)
        self.assertAlmostEqual(summary["processing_latency_mean_ms"], 20)

    def test_empty_measurement_not_a_success(self):
        self.assertEqual(FrameStats().summary()["status"], "insufficient_frames_after_warmup")

    def test_composition_endpoints_and_soft_alpha(self):
        foreground = np.full((3, 4, 3), 200, np.uint8)
        background = np.full_like(foreground, 20)
        for alpha, expected in [(0, 20), (0.5, 110), (1, 200)]:
            result = blend(foreground, background, np.full((3, 4), alpha, np.float32))
            np.testing.assert_array_equal(result, np.full_like(foreground, expected))

    def test_center_crop_preserves_aspect_ratio(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, 5:15] = 200
        np.testing.assert_array_equal(cover(image, 10, 10), np.full((10, 10, 3), 200, np.uint8))

    def test_recording_uses_elapsed_time_instead_of_processed_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.avi"
            recorder = RealtimeRecorder(path, 10, (64, 48))
            frame = np.full((48, 64, 3), 127, np.uint8)
            recorder.push(frame, 100)
            recorder.push(frame, 100.55)
            recorder.close(101.01)
            cap = cv2.VideoCapture(str(path))
            self.assertTrue(cap.isOpened())
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.assertEqual(count, 11)  # Only 2 processed frames, but ~1 s of playback.
            self.assertAlmostEqual(fps, 10)

    def test_bad_background_folder_is_rejected(self):
        with self.assertRaises(SystemExit):
            parse_args(["--background-dir", "/does-not-exist-hw1"])

    def test_switching_background_modes_keeps_output_shape(self):
        choices = Backgrounds(64, 48, "color", (10, 20, 30))
        frame = np.full((48, 64, 3), 180, np.uint8)
        np.testing.assert_array_equal(choices.get(frame)[0, 0], [30, 20, 10])
        self.assertTrue(choices.handle_key(ord("2")))
        self.assertEqual(choices.mode, "blur")
        self.assertTrue(choices.handle_key(ord("3")))
        first = choices.get(frame).copy()
        self.assertTrue(choices.handle_key(ord("n")))
        self.assertFalse(np.array_equal(first, choices.get(frame)))
        self.assertEqual(choices.get(frame).shape, frame.shape)
        self.assertTrue(choices.handle_key(ord("c")))
        self.assertEqual(choices.mode, "color")


if __name__ == "__main__":
    unittest.main()
