import unittest
import numpy as np
from ksana.utils.sample_solver import (
    get_sigmas_with_denoise,
    get_timesteps_with_denoise,
    apply_sigma_shift,
)


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
