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
Tests for Engine singleton pattern (get_default / reset_default / multi-instance).

These tests mock init_executors to avoid CUDA/Ray dependencies.
"""

import logging
import threading
import unittest
from unittest.mock import patch

from kdit.engine.engine import Engine, get_engine


def _noop_init_executors(self, dist_config=None, offload_device=None):
    """Stub that skips real executor initialization."""
    self.executors = "mock_executor"


@patch.object(Engine, "init_executors", _noop_init_executors)
class TestKsanaEngineSingleton(unittest.TestCase):
    """Test the get_default / reset_default singleton lifecycle."""

    def setUp(self):
        Engine.reset_default()

    def tearDown(self):
        Engine.reset_default()

    def test_get_default_returns_same_instance(self):
        """get_default() should return the same instance on repeated calls."""
        e1 = Engine.get_default()
        e2 = Engine.get_default()
        self.assertIs(e1, e2)

    def test_get_engine_delegates_to_get_default(self):
        """get_engine() backward-compat wrapper should return the default instance."""
        e1 = get_engine()
        e2 = Engine.get_default()
        self.assertIs(e1, e2)

    def test_isinstance_works(self):
        """isinstance() should work correctly (was broken with @singleton)."""
        engine = Engine.get_default()
        self.assertIsInstance(engine, Engine)

    def test_reset_default_clears_instance(self):
        """reset_default() should clear the cached instance."""
        e1 = Engine.get_default()
        Engine.reset_default()
        e2 = Engine.get_default()
        self.assertIsNot(e1, e2)

    def test_reset_default_calls_cleanup_distributed(self):
        """reset_default() should call cleanup_distributed() on the old instance."""
        engine = Engine.get_default()
        self.assertFalse(engine._cleaned_up)
        Engine.reset_default()
        self.assertTrue(engine._cleaned_up)

    def test_direct_init_creates_independent_instance(self):
        """Engine() should create an independent instance, not the default."""
        default = Engine.get_default()
        independent = Engine()
        self.assertIsNot(default, independent)
        self.assertIsInstance(independent, Engine)

    def test_duplicate_args_logs_warning(self):
        """get_default() with args on second call should log a warning."""
        Engine.get_default()
        with self.assertLogs("kdit", level=logging.WARNING) as cm:
            Engine.get_default(offload_device="cuda:0")
        self.assertTrue(any("Arguments are ignored" in msg for msg in cm.output))


@patch.object(Engine, "init_executors", _noop_init_executors)
class TestKsanaEngineCleanup(unittest.TestCase):
    """Test cleanup_distributed idempotency and atexit registration."""

    def setUp(self):
        Engine.reset_default()

    def tearDown(self):
        Engine.reset_default()

    def test_cleanup_distributed_is_idempotent(self):
        """Multiple cleanup_distributed() calls should not raise."""
        engine = Engine()
        engine.cleanup_distributed()
        engine.cleanup_distributed()  # should not raise
        self.assertTrue(engine._cleaned_up)

    def test_default_instance_registers_atexit(self):
        """get_default() registers atexit with cleanup_distributed."""
        with patch("kdit.engine.engine.atexit") as mock_atexit:
            engine = Engine.get_default()
            mock_atexit.register.assert_called_once_with(engine.cleanup_distributed)

    def test_direct_instance_no_atexit(self):
        """Direct Engine() should NOT register atexit (caller manages lifecycle)."""
        with patch("kdit.engine.engine.atexit") as mock_atexit:
            Engine()
            mock_atexit.register.assert_not_called()

    def test_explicit_register_atexit_true(self):
        """Engine(_register_atexit=True) should register atexit."""
        with patch("kdit.engine.engine.atexit") as mock_atexit:
            engine = Engine(_register_atexit=True)
            mock_atexit.register.assert_called_once_with(engine.cleanup_distributed)


@patch.object(Engine, "init_executors", _noop_init_executors)
class TestKsanaEngineThreadSafety(unittest.TestCase):
    """Test thread safety of get_default()."""

    def setUp(self):
        Engine.reset_default()

    def tearDown(self):
        Engine.reset_default()

    def test_concurrent_get_default_returns_same_instance(self):
        """Multiple threads calling get_default() should all get the same instance."""
        results = [None] * 10
        barrier = threading.Barrier(10)

        def worker(idx):
            barrier.wait()
            results[idx] = Engine.get_default()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same instance
        self.assertTrue(all(r is results[0] for r in results))


if __name__ == "__main__":
    unittest.main()
