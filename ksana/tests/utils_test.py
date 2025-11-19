import unittest
import numpy as np
import torch

from ksana.utils.debug import print_recursive
from ksana.utils.logger import log


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
