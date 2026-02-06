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

from ksana.utils import Factory


class FactoryA(Factory):
    pass


class FactoryB(Factory):
    pass


@FactoryA.register("group_a", "key_a")
class FactoryAGroupAKeyA:
    def func(self):
        return "a_a_a"


@FactoryA.register("group_a", "key_b")
class FactoryAGroupAKeyB:
    def func(self):
        return "a_a_b"


@FactoryA.register("group_a", ["key_c", "key_d"])
class FactoryAGroupAKeyX:
    def func(self):
        return "a_a_x"


@FactoryA.register("group_b", "key_a")
@FactoryA.register("group_a", "key_y")
class FactoryAGroupBKeyA:
    def func(self):
        return "a_b_a"


@FactoryA.register("group_b", ["key_b", "key_c", "key_d"])
class FactoryAGroupBKeyX:
    def func(self):
        return "a_b_x"


@FactoryB.register("group_a", "key_a")
class FactoryBGroupAKeyA:
    def func(self):
        return "b_a_a"


class TestFactory(unittest.TestCase):
    def test_factory_group(self):
        self.assertEqual(FactoryA.create("group_a", "key_a").func(), "a_a_a")
        self.assertEqual(FactoryA.create("group_a", "key_b").func(), "a_a_b")

        self.assertEqual(FactoryA.create("group_b", "key_a").func(), "a_b_a")
        self.assertEqual(FactoryA.create("group_a", "key_y").func(), "a_b_a")

    def test_factory_key_list(self):
        self.assertEqual(FactoryA.create("group_a", "key_c").func(), "a_a_x")
        self.assertEqual(FactoryA.create("group_a", "key_d").func(), "a_a_x")

        self.assertEqual(FactoryA.create("group_b", "key_b").func(), "a_b_x")
        self.assertEqual(FactoryA.create("group_b", "key_c").func(), "a_b_x")
        self.assertEqual(FactoryA.create("group_b", "key_d").func(), "a_b_x")

    def test_diff_factory(self):
        self.assertEqual(FactoryA.create("group_a", "key_a").func(), "a_a_a")
        self.assertEqual(FactoryB.create("group_a", "key_a").func(), "b_a_a")

    def test_not_exsit(self):
        with self.assertRaises(KeyError):
            FactoryA.create("group_a", "x")
        with self.assertRaises(KeyError):
            FactoryA.create("x", "x")


if __name__ == "__main__":
    unittest.main()
