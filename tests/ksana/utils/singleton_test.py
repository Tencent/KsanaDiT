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
from dataclasses import dataclass, field

from ksana.utils import singleton


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


if __name__ == "__main__":
    unittest.main()
