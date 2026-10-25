"""Run with:  python -m unittest discover -s tests -v   (no extra installs needed)"""
import os
import tempfile
import unittest

import numpy as np

from biosync import compare as C
from biosync import simulate as S
from biosync import sync


class TestDTW(unittest.TestCase):
    def test_identical_signals_have_zero_distance(self):
        x = np.sin(np.linspace(0, 6, 80))
        self.assertAlmostEqual(C.dtw_distance(x, x), 0.0)

    def test_speed_change_is_cheaper_than_shape_change(self):
        t = np.linspace(0, 1, 100)
        base = np.sin(np.pi * t) ** 2
        slower = np.interp(np.linspace(0, 1, 130), t, base)      # same shape, 30 % slower
        smaller = 0.5 * base                                      # same speed, half the range
        self.assertLess(C.dtw_distance(slower, base), C.dtw_distance(smaller, base) / 3)

    def test_path_starts_and_ends_at_corners(self):
        a, b = np.random.default_rng(0).random(30), np.random.default_rng(1).random(40)
        path = C.dtw_path(C.dtw_matrix(a, b))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (29, 39))


class TestSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        data, truth = S.make_session(6, seed=3)
        S.write_session(cls.tmp.name, data, truth)
        cls.aligned = sync.load_and_align(cls.tmp.name, S.BANDS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clap_alignment_error_within_one_sample(self):
        # true clap time on each band's own clock = clap_real - start_offset
        for b in S.BANDS:
            true_local = S.CLAP_REAL_S - S.START_OFFSET_S[b]
            self.assertLess(abs(self.aligned["clap_local"][b] - true_local), 0.0101)

    def test_completeness_matches_simulated_loss(self):
        for b in S.BANDS:
            self.assertAlmostEqual(self.aligned["completeness"][b],
                                   1 - S.DROP_FRACTION[b], delta=0.02)

    def test_missing_clap_raises(self):
        t = np.arange(0, 3, 0.01)
        acc = np.tile([0, 0, 1.0], (len(t), 1))
        with self.assertRaises(ValueError):
            sync.detect_clap(t, acc)


class TestEndToEnd(unittest.TestCase):
    def test_finds_injected_deviations_without_false_alarms(self):
        with tempfile.TemporaryDirectory() as d:
            S.generate_demo_data(d)
            cal = sync.load_and_align(os.path.join(d, "calibration"), S.BANDS)
            ses = sync.load_and_align(os.path.join(d, "session"), S.BANDS)
            cc = C.segment_cycles(cal["speed"]["L_wrist"])
            sc = C.segment_cycles(ses["speed"]["L_wrist"])
            self.assertEqual((len(cc), len(sc)), (12, 12))
            model = C.calibrate({b: C.cut_windows(cal["speed"][b], cc) for b in S.BANDS})
            scores = C.score_session({b: C.cut_windows(ses["speed"][b], sc) for b in S.BANDS}, model)
            found = {(i, b) for b in S.BANDS for i in range(12) if scores[b][i]["flag"]}
            self.assertEqual(found, set(S.DEMO_DEVIATIONS))


if __name__ == "__main__":
    unittest.main()
