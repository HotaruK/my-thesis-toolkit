import unittest
from .evaluator import get_scores


class EvaluatorTestCase(unittest.TestCase):
    def test_ok(self):
        reference = [['this', 'is', 'a', 'test', 'of', 'the', 'evaluation', 'functions']]
        candidate = ['this', 'is', 'not', 'a', 'test', 'but', 'an', 'evaluation', 'of', 'functions']
        result = get_scores(reference, candidate)

        self.assertIn('bleu', result)
        self.assertIn('meteor', result)
        self.assertIn('nist', result)
        self.assertIn('rouge', result)
