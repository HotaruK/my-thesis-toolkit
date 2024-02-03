import unittest
from .clearning import _filling_missing_frames


class CleaningTestCase(unittest.TestCase):
    def test_filling_missing_frames(self):
        ans = _filling_missing_frames()
