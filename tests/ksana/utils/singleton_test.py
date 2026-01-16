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
