import unittest
import numpy as np
import torch

from ksana.utils.debug import print_recursive
from ksana.utils.logger import log
from ksana.utils import singleton
from ksana.utils.sample_solver import (
    get_sigmas_with_denoise,
    get_timesteps_with_denoise,
    apply_sigma_shift,
)

from dataclasses import dataclass, field


@singleton
@dataclass
class TestSingletonClass:
    num: int = field(default=0)

    @staticmethod
    def static_func(num: int = 1):
        return num

    @classmethod
    def class_func(cls, num: int = 2):
        return num


class TestSingleton(unittest.TestCase):
    def test_singleton(self):
        db1 = TestSingletonClass(num=-1)
        db2 = TestSingletonClass(num=-2)
        self.assertEqual(db1, db2)
        self.assertEqual(db1.num, -1)
        self.assertEqual(db1.static_func(), 1)
        self.assertEqual(db1.static_func(3), 3)
        self.assertEqual(db1.class_func(), 2)
        self.assertEqual(db1.class_func(4), 4)


class TestPrint(unittest.TestCase):
    def test_print_recursive(self):
        obj = {"a": 1, "b": [2, 3, 4], "c": {"d": 5}}
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)

    def test_print_recursive_numpy(self):
        obj = {"a": 1, "b": np.array([1, 2, 3]), "c": {"d": 5}}
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)

    def test_print_recursive_torch(self):
        obj = {"a": 1, "b": torch.tensor([1, 2, 3]), "c": {"d": 5}}
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)

    def test_print_recursive_torch_float(self):
        obj = {"a": 1, "b": torch.tensor([1, 2, 3]).to(torch.float16), "c": {"d": 5}}
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)


class TestSampleSolver(unittest.TestCase):
    def test_get_sigmas_with_denoise_full(self):
        steps = 10
        denoise = 1.0
        start = 1.0
        end = 0.0

        sigmas = get_sigmas_with_denoise(steps, denoise, start, end)

        self.assertEqual(len(sigmas), steps)
        self.assertAlmostEqual(sigmas[1], 0.9)
        self.assertAlmostEqual(sigmas[2], 0.8)

    def test_get_sigmas_with_denoise_partial(self):
        steps = 5
        denoise = 0.5
        start = 1.0
        end = 0.0

        sigmas = get_sigmas_with_denoise(steps, denoise, start, end)

        self.assertEqual(len(sigmas), steps)
        self.assertAlmostEqual(sigmas[0], 0.5)
        self.assertAlmostEqual(sigmas[1], 0.4)

    def test_get_timesteps_with_denoise(self):
        num_steps = 10
        max_steps = 1000
        denoise = 1.0

        timesteps = get_timesteps_with_denoise(num_steps, max_steps, denoise)

        self.assertEqual(len(timesteps), num_steps + 1)
        self.assertAlmostEqual(timesteps[0], 1000.0, places=5)
        self.assertAlmostEqual(timesteps[1], 900.0, places=5)
        self.assertAlmostEqual(timesteps[-1], 0.0, places=5)
        self.assertEqual(timesteps.dtype, np.float32)

    def test_apply_sigma_shift(self):
        sigmas = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        shift = 2.0

        result = apply_sigma_shift(sigmas, shift, use_dynamic_shifting=False)
        expected = shift * sigmas / (1 + (shift - 1) * sigmas)

        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == "__main__":
    unittest.main()
