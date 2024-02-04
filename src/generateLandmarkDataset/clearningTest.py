import unittest
import os
from .clearning import _filling_missing_frames, _get_blacklist, _extract_x_and_y, _get_average


class CleaningTestCase(unittest.TestCase):
    def test_get_blacklist(self):
        blacklist_root = os.path.join(os.getcwd(), 'generateLandmarkDataset', 'blacklist')

        expect = [
            "01June_2011_Wednesday_tagesschau-6952",
            "08October_2009_Thursday_tagesschau-5358",
            "11May_2010_Tuesday_heute-5994",
            "12October_2009_Monday_tagesschau-561",
            "24September_2009_Thursday_heute-6080", ]
        r = _get_blacklist(blacklist_root, 'test')
        self.assertEqual(r, expect)

    def test_extract_x_and_y(self):
        dummy_data = [{'x': 0.55313516, 'y': 0.23210412, 'z': -0.8369835, 'visibility': 0.999959},
                      {'x': 0.5791322, 'y': 0.19208825, 'z': -0.7675163, 'visibility': 0.99994004},
                      {'x': 0.5965038, 'y': 0.19301778, 'z': -0.76808345, 'visibility': 0.9999324},
                      {'x': 0.61284065, 'y': 0.19443959, 'z': -0.7679376, 'visibility': 0.999941},
                      {'x': 0.52871716, 'y': 0.18914086, 'z': -0.7616201, 'visibility': 0.9999348},
                      {'x': 0.50644064, 'y': 0.1885562, 'z': -0.76111794, 'visibility': 0.9999267},
                      {'x': 0.48512763, 'y': 0.1890204, 'z': -0.7614062, 'visibility': 0.99992955},
                      {'x': 0.63365376, 'y': 0.2159813, 'z': -0.30458978, 'visibility': 0.9999261},
                      {'x': 0.45382077, 'y': 0.20988971, 'z': -0.25462678, 'visibility': 0.9998801}]
        expect = [{'x': 0.55313516, 'y': 0.23210412, },
                  {'x': 0.5791322, 'y': 0.19208825, },
                  {'x': 0.5965038, 'y': 0.19301778, },
                  {'x': 0.61284065, 'y': 0.19443959, },
                  {'x': 0.52871716, 'y': 0.18914086, },
                  {'x': 0.50644064, 'y': 0.1885562, },
                  {'x': 0.48512763, 'y': 0.1890204, },
                  {'x': 0.63365376, 'y': 0.2159813, },
                  {'x': 0.45382077, 'y': 0.20988971, }]
        self.assertEqual(_extract_x_and_y(dummy_data), expect)

    def test_get_average(self):
        dummy_1 = [{'x': 0.55313516, 'y': 0.23210412, },
                   {'x': 0.5791322, 'y': 0.19208825, },
                   {'x': 0.5965038, 'y': 0.19301778, },
                   {'x': 0.61284065, 'y': 0.19443959, },
                   {'x': 0.52871716, 'y': 0.18914086, }]
        dummy_2 = [{'x': 0.48710012, 'y': 0.2332843, },
                   {'x': 0.52983433, 'y': 0.19250858, },
                   {'x': 0.55210686, 'y': 0.19341552, },
                   {'x': 0.5677911, 'y': 0.1947186, },
                   {'x': 0.47006464, 'y': 0.18993598, }]
        expect = [
            {'x': 0.52011764, 'y': 0.23269421},
            {'x': 0.554483265, 'y': 0.192298415},
            {'x': 0.5743053300000001, 'y': 0.19321665},
            {'x': 0.5903158749999999, 'y': 0.19457909499999998},
            {'x': 0.49939089999999997, 'y': 0.18953841999999999}
        ]
        self.assertEqual(_get_average(dummy_1, dummy_2), expect)

    def test_filling_missing_frames(self):
        _filling_missing_frames()