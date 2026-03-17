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

import io
import time
import unittest
from unittest.mock import patch

from kdit.utils.logger import log
from kdit.utils.profile import ProfileNode, TimeProfiler, Timer, TimerProfiler, time_profile

SLEEP_TIME = 0.01


# ---------------------------------------------------------------------------
# 装饰器形式（模块级函数）
# ---------------------------------------------------------------------------


@time_profile
def func_with_no_name():
    time.sleep(SLEEP_TIME)
    return "ok"


@time_profile("my_func_with_name")
def func_with_name():
    time.sleep(SLEEP_TIME)
    return 42


@time_profile("my_func_with_name_and_print_func", log.info)
def func_with_name_and_log():
    time.sleep(SLEEP_TIME)


@time_profile("my_func_with_name_and_print_func", print)
def func_with_name_and_print():
    time.sleep(SLEEP_TIME)


@time_profile(profile=False)
def func_pure_timer():
    """profile=False 应退化为纯 Timer，不参与层级树。"""
    time.sleep(SLEEP_TIME)
    return "pure"


class TestTimeProfile(unittest.TestCase):
    """time_profile 作为 with 语句和装饰器的基本功能。"""

    # -- with 语句 ----------------------------------------------------------

    def test_with_no_args(self):
        with time_profile():
            time.sleep(SLEEP_TIME)

    def test_with_name(self):
        with time_profile("test_time_profile"):
            time.sleep(SLEEP_TIME)

    def test_with_name_and_print_func(self):
        with time_profile("test_time_profile", log.info):
            time.sleep(SLEEP_TIME)

        with time_profile("test_time_profile", print):
            time.sleep(SLEEP_TIME)

    def test_with_note(self):
        with time_profile("step_0", note="t=0.9876"):
            time.sleep(SLEEP_TIME)

    def test_with_profile_false_returns_timer(self):
        ctx = time_profile("pure", profile=False)
        self.assertIsInstance(ctx, Timer)
        self.assertNotIsInstance(ctx, TimerProfiler)

    def test_with_profile_true_returns_timer_profiler(self):
        ctx = time_profile("profiled")
        self.assertIsInstance(ctx, TimerProfiler)

    # -- 装饰器 -------------------------------------------------------------

    def test_func_with_no_name(self):
        result = func_with_no_name()
        self.assertEqual(result, "ok")

    def test_func_with_name(self):
        result = func_with_name()
        self.assertEqual(result, 42)

    def test_func_with_name_and_print(self):
        func_with_name_and_print()

    def test_func_with_name_and_log(self):
        func_with_name_and_log()

    def test_func_pure_timer(self):
        result = func_pure_timer()
        self.assertEqual(result, "pure")

    def test_decorator_preserves_function_name(self):
        self.assertEqual(func_with_no_name.__name__, "func_with_no_name")
        self.assertEqual(func_with_name.__name__, "func_with_name")
        self.assertEqual(func_pure_timer.__name__, "func_pure_timer")

    def test_invalid_argument_raises(self):
        with self.assertRaises(TypeError):
            time_profile(123)


class TestTimerProfiler(unittest.TestCase):
    """TimerProfiler 子类的行为。"""

    def test_inherits_timer(self):
        tp = TimerProfiler("test")
        self.assertIsInstance(tp, Timer)

    def test_note_attribute(self):
        tp = TimerProfiler("test", note="hello")
        self.assertEqual(tp.note, "hello")

    def test_context_manager_without_session(self):
        """无活跃 session 时退化为纯计时器，不报错。"""
        # 确保没有活跃 session
        TimeProfiler._local.current_profiler = None
        with TimerProfiler("test_no_session"):
            time.sleep(SLEEP_TIME)

    def test_decorator_usage(self):
        tp = TimerProfiler("my_timer")

        @tp
        def inner():
            return "decorated"

        result = inner()
        self.assertEqual(result, "decorated")


class TestTimeProfiler(unittest.TestCase):
    """TimeProfiler 层级树功能。"""

    def test_start_and_finish_session(self):
        profiler = TimeProfiler.start_session("test_session")
        self.assertIs(TimeProfiler.get_current(), profiler)

        root = profiler.finish()
        self.assertIsNone(TimeProfiler.get_current())
        self.assertIsInstance(root, ProfileNode)
        self.assertEqual(root.name, "test_session")
        self.assertGreater(root.elapsed, 0)

    def test_begin_end_creates_children(self):
        profiler = TimeProfiler.start_session("root")
        profiler.begin("child_1")
        time.sleep(SLEEP_TIME)
        profiler.end()

        profiler.begin("child_2", note="extra")
        time.sleep(SLEEP_TIME)
        profiler.end()

        root = profiler.finish()
        self.assertEqual(len(root.children), 2)
        self.assertEqual(root.children[0].name, "child_1")
        self.assertEqual(root.children[1].name, "child_2")
        self.assertEqual(root.children[1].note, "extra")
        self.assertGreater(root.children[0].elapsed, 0)

    def test_nested_hierarchy(self):
        profiler = TimeProfiler.start_session("root")
        profiler.begin("level_1")
        profiler.begin("level_2")
        time.sleep(SLEEP_TIME)
        profiler.end()
        profiler.end()
        root = profiler.finish()

        self.assertEqual(len(root.children), 1)
        level_1 = root.children[0]
        self.assertEqual(level_1.name, "level_1")
        self.assertEqual(len(level_1.children), 1)
        self.assertEqual(level_1.children[0].name, "level_2")

    def test_finish_auto_closes_unclosed_nodes(self):
        profiler = TimeProfiler.start_session("root")
        profiler.begin("unclosed_1")
        profiler.begin("unclosed_2")
        root = profiler.finish()
        # finish() 应自动关闭所有未关闭的节点
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].name, "unclosed_1")
        self.assertEqual(root.children[0].children[0].name, "unclosed_2")

    def test_print_summary(self):
        profiler = TimeProfiler.start_session("pipeline")
        profiler.begin("step_0", note="t=1.0")
        time.sleep(SLEEP_TIME)
        profiler.end()
        profiler.begin("step_1")
        profiler.begin("model_forward")
        time.sleep(SLEEP_TIME)
        profiler.end()
        profiler.end()
        profiler.finish()

        buf = io.StringIO()
        profiler.print_summary(file=buf)
        output = buf.getvalue()
        self.assertIn("pipeline", output)
        self.assertIn("step_0", output)
        self.assertIn("t=1.0", output)
        self.assertIn("step_1", output)
        self.assertIn("model_forward", output)
        self.assertIn("KsanaDiT Profile Summary", output)

    def test_double_finish_is_idempotent(self):
        profiler = TimeProfiler.start_session("root")
        root1 = profiler.finish()
        root2 = profiler.finish()
        self.assertIs(root1, root2)

    def test_end_on_empty_stack_returns_zero(self):
        profiler = TimeProfiler.start_session("root")
        # 栈中只有 root，end() 应返回 0.0
        elapsed = profiler.end()
        self.assertEqual(elapsed, 0.0)


class TestTimeProfileIntegration(unittest.TestCase):
    """time_profile + TimeProfiler 集成测试。"""

    @patch("kdit.utils.profile.KSANA_PROFILE", True)
    def test_time_profile_hooks_into_session(self):
        """KSANA_PROFILE=True 时，time_profile 的 with 语句应自动挂到 TimeProfiler 树上。"""
        profiler = TimeProfiler.start_session("integration")

        with time_profile("outer"):
            with time_profile("inner", note="detail"):
                time.sleep(SLEEP_TIME)

        root = profiler.finish()
        self.assertEqual(len(root.children), 1)
        outer = root.children[0]
        self.assertEqual(outer.name, "outer")
        self.assertEqual(len(outer.children), 1)
        inner = outer.children[0]
        self.assertEqual(inner.name, "inner")
        self.assertEqual(inner.note, "detail")
        self.assertGreater(inner.elapsed, 0)

    @patch("kdit.utils.profile.KSANA_PROFILE", False)
    def test_time_profile_no_session_when_disabled(self):
        """KSANA_PROFILE=False 时，即使有 session 也不挂树。"""
        profiler = TimeProfiler.start_session("disabled")

        with time_profile("should_not_appear"):
            time.sleep(SLEEP_TIME)

        root = profiler.finish()
        self.assertEqual(len(root.children), 0)

    @patch("kdit.utils.profile.KSANA_PROFILE", True)
    def test_profile_false_skips_tree(self):
        """profile=False 的 time_profile 不参与层级树。"""
        profiler = TimeProfiler.start_session("skip_test")

        with time_profile("visible"):
            time.sleep(SLEEP_TIME)

        with time_profile("invisible", profile=False):
            time.sleep(SLEEP_TIME)

        root = profiler.finish()
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].name, "visible")

    @patch("kdit.utils.profile.KSANA_PROFILE", True)
    def test_decorator_hooks_into_session(self):
        """装饰器形式的 time_profile 也应挂到 TimeProfiler 树上。"""
        profiler = TimeProfiler.start_session("decorator_test")

        @time_profile
        def my_func():
            time.sleep(SLEEP_TIME)
            return "result"

        result = my_func()
        self.assertEqual(result, "result")

        root = profiler.finish()
        self.assertEqual(len(root.children), 1)
        self.assertEqual(root.children[0].name, "my_func")

    @patch("kdit.utils.profile.KSANA_PROFILE", True)
    def test_no_session_no_crash(self):
        """有 KSANA_PROFILE=True 但无 session 时不应崩溃。"""
        # 确保无活跃 session
        TimeProfiler._local.current_profiler = None

        with time_profile("orphan"):
            time.sleep(SLEEP_TIME)
        # 不崩溃即通过


class TestProfileNode(unittest.TestCase):
    """ProfileNode 数据结构。"""

    def test_default_values(self):
        node = ProfileNode(name="test")
        self.assertEqual(node.name, "test")
        self.assertEqual(node.elapsed, 0.0)
        self.assertIsNone(node.note)
        self.assertEqual(node.children, [])
        self.assertIsNone(node.parent)

    def test_with_note(self):
        node = ProfileNode(name="step", note="t=0.5")
        self.assertEqual(node.note, "t=0.5")


if __name__ == "__main__":
    unittest.main()
