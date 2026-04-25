import os
import unittest
from unittest.mock import patch

import torch

from lmflow.utils.envs import (
    get_device_name,
    get_torch_device,
    is_accelerate_env,
    require_cuda_for_gpu_mode,
    set_cuda_device,
)


class TestEnvs(unittest.TestCase):
    def test_is_accelerate_env_false_without_prefix(self):
        with patch.dict(os.environ, {"FOO": "1"}, clear=True):
            self.assertFalse(is_accelerate_env())

    def test_is_accelerate_env_true_with_prefix(self):
        with patch.dict(os.environ, {"ACCELERATE_USE_CPU": "1"}, clear=True):
            self.assertTrue(is_accelerate_env())

    def test_is_accelerate_env_false_when_accelerate_not_prefix(self):
        """Names containing 'ACCELERATE' but not starting with ACCELERATE_ must be ignored."""
        with patch.dict(os.environ, {"MY_ACCELERATE_SETTING": "1"}, clear=True):
            self.assertFalse(is_accelerate_env())

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_name_cpu_when_cuda_unavailable(self, _mock_cuda: object):
        self.assertEqual(get_device_name(), "cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_name_cuda_when_cuda_available(self, _mock_cuda: object):
        self.assertEqual(get_device_name(), "cuda")

    def test_get_torch_device_matches_device_name(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertIs(get_torch_device(), torch.cpu)
        with patch("torch.cuda.is_available", return_value=True):
            self.assertIs(get_torch_device(), torch.cuda)

    @patch(
        "lmflow.utils.envs.get_device_name",
        return_value="zzz_nonexistent_lmflow_test",
    )
    def test_get_torch_device_fallback_returns_cuda_on_attribute_error(self, _mock_name: object):
        with self.assertLogs("lmflow.utils.envs", level="WARNING") as log_ctx:
            self.assertIs(get_torch_device(), torch.cuda)
        self.assertTrue(
            any("zzz_nonexistent_lmflow_test" in entry and "not found" in entry for entry in log_ctx.output),
        )

    @patch("torch.cuda.is_available", return_value=False)
    def test_require_cuda_for_gpu_mode_raises_when_cuda_unavailable(self, _mock_cuda: object):
        with self.assertRaises(RuntimeError) as ctx:
            require_cuda_for_gpu_mode()
        self.assertIn("CUDA is not available", str(ctx.exception))

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_set_cuda_device_calls_torch_set_device(self, mock_set_device: object, _mock_cuda: object):
        set_cuda_device(2)
        mock_set_device.assert_called_once_with(2)

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.set_device")
    def test_set_cuda_device_raises_without_cuda(self, mock_set_device: object, _mock_cuda: object):
        with self.assertRaises(RuntimeError):
            set_cuda_device(2)
        mock_set_device.assert_not_called()
