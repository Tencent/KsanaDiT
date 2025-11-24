import unittest
import numpy as np
import torch

from ksana.utils.debug import print_recursive
from ksana.utils.logger import log
from ksana.utils import singleton

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


if __name__ == "__main__":
    unittest.main()
