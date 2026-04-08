"""Regression tests for evaluator user-facing error messages."""

import unittest

from lmflow.pipeline.evaluator import _nll_unsupported_dataset_message


class TestEvaluatorMessages(unittest.TestCase):
    def test_nll_unsupported_dataset_message_includes_type(self):
        msg = _nll_unsupported_dataset_message("conversation")
        self.assertIn("conversation", msg)
        self.assertNotIn("f{", msg)
