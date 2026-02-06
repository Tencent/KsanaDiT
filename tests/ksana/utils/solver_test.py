# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from ksana.utils.sample_solver import (
    apply_sigma_shift,
    get_sigmas_with_denoise,
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

    def test_apply_sigma_shift(self):
        sigmas = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        shift = 2.0

        result = apply_sigma_shift(sigmas, shift, use_dynamic_shifting=False)
        expected = shift * sigmas / (1 + (shift - 1) * sigmas)

        np.testing.assert_array_almost_equal(result, expected, decimal=6)


if __name__ == "__main__":
    unittest.main()
