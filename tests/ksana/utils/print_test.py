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
import torch

from ksana.nodes.output_types import KsanaNodeVAEEncodeOutput
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

    def test_print_recursive_obj_torch_float(self):
        obj = KsanaNodeVAEEncodeOutput(
            samples=torch.tensor([1, 2, 3]).to(torch.float16),
            with_end_image=True,
            batch_size_per_prompts=1,
        )
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)


if __name__ == "__main__":
    unittest.main()
