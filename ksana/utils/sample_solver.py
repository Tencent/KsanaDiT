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

"""
Utility functions for sample solvers, including denoise support.
"""

import numpy as np


def get_sigmas_with_denoise(steps, denoise=1.0, start=1.0, end=0.0):
    """
    This function implements the ComfyUI denoise logic by adjusting the schedule.
    - When denoise=1.0: Standard linear schedule from `start` to `end`.
    - When denoise<1.0: A longer, more fine-grained schedule is generated, and only the tail end is used,
      effectively skipping the initial high-value steps.
    """
    if denoise is None or denoise >= 0.9999:
        # Standard linear schedule from start down to (but not including) end.
        values = np.linspace(start, end, steps + 1)[:-1]
    elif denoise <= 0.0:
        return np.array([])
    else:
        new_steps = int(steps / denoise)
        full_values = np.linspace(start, end, new_steps + 1)

        values = full_values[-(steps + 1) : -1]

    return values.copy()


def apply_sigma_shift(sigmas, shift, use_dynamic_shifting=False):
    if use_dynamic_shifting or shift is None:
        return sigmas

    return shift * sigmas / (1 + (shift - 1) * sigmas)
