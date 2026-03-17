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

"""Unified platform-aware test helper shared across all test suites.

Usage::

    from platform_test_helper import get_platform_expected_or_skip

    # test_name is auto-inferred from the calling unittest class + method
    expected = get_platform_expected_or_skip({"GPU": {"mean0": 0.66}})
"""

import inspect
from collections.abc import Callable
from unittest import SkipTest

from kdit.accelerator import platform

_PLATFORM_DETECTORS: dict[str, Callable[[], bool]] = {
    "NPU": platform.is_npu,
    "XPU": platform.is_xpu,
    "GPU": platform.is_gpu,
}

ALL_PLATFORMS: set[str] = set(_PLATFORM_DETECTORS.keys())


def _detect_platform() -> str:
    """Return the first matching platform name, defaulting to ``"GPU"``."""
    for name, detector in _PLATFORM_DETECTORS.items():
        if detector():
            return name
    return "GPU"


CURRENT_PLATFORM = _detect_platform()


def _infer_test_name() -> str:
    """Walk the call stack to find the nearest ``unittest.TestCase`` test method.

    Prefers methods whose name starts with ``test``; falls back to the first
    ``TestCase`` method found.  Returns ``ClassName.method_name`` when found,
    otherwise ``"<unknown>"``.
    """
    fallback: str | None = None
    for frame_info in inspect.stack():
        # Skip frames inside this module
        if frame_info.filename == __file__:
            continue
        local_self = frame_info.frame.f_locals.get("self")
        if local_self is not None and hasattr(local_self, "assertAlmostEqual"):
            class_name = type(local_self).__name__
            method_name = frame_info.function
            candidate = f"{class_name}.{method_name}"
            if method_name.startswith("test"):
                return candidate
            if fallback is None:
                fallback = candidate
    return fallback or "<unknown>"


def get_platform_expected_or_skip(config_map: dict, *, test_name: str = ""):
    """Return the platform-specific expected config or skip the test.

    Parameters
    ----------
    config_map : dict
        Mapping of platform name (``"GPU"`` / ``"NPU"`` / ``"XPU"``) to expected values.
    test_name : str, optional
        Human-readable test identifier for the skip message.
        When omitted (default), it is **automatically inferred** from the
        calling ``unittest.TestCase`` class and method name.
    """
    if not test_name:
        test_name = _infer_test_name()
    normalized = {str(key).upper(): value for key, value in (config_map or {}).items()}
    config = normalized.get(CURRENT_PLATFORM)
    if config is None:
        raise SkipTest(f"{test_name} skipped on {CURRENT_PLATFORM}: no config defined")
    return config
