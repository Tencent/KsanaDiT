import unittest
from dataclasses import dataclass, field

import numpy as np
import torch

from ksana.config.sample_config import KsanaSampleConfig
from ksana.nodes.output_types import KsanaNodeVAEEncodeOutput
from ksana.utils import evolve_with_recommend, singleton
from ksana.utils.debug import print_recursive
from ksana.utils.logger import log


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


class TestEvolve(unittest.TestCase):
    def test_dataclass(self):
        input_config = KsanaSampleConfig(batch_per_prompt=4, shift=None, steps=1)
        recommend_config = KsanaSampleConfig(shift=0.3, batch_per_prompt=8)
        out_config = evolve_with_recommend(input_config, recommend_config)
        self.assertEqual(out_config.shift, 0.3)  # only update None
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.batch_per_prompt, 4)

        out_config = evolve_with_recommend(input_config, recommend_config, force_update=True)
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.batch_per_prompt, 8)

        out_config = evolve_with_recommend(out_config, {"batch_per_prompt": 9})
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.batch_per_prompt, 8)

        out_config = evolve_with_recommend(out_config, {"batch_per_prompt": 9}, force_update=True)
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.batch_per_prompt, 9)

        # do not update None
        out_config = evolve_with_recommend(out_config, {"batch_per_prompt": None}, force_update=True)
        self.assertEqual(out_config.batch_per_prompt, 9)

    def test_dict(self):
        input_config = dict(cfg_scale=None, batch_per_prompt=4, thisother="a")
        recommend_config = dict(cfg_scale=5.5, shift=0.3, batch_per_prompt=8, other="s")
        out_config = evolve_with_recommend(input_config, recommend_config)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("batch_per_prompt"), 4)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(input_config, recommend_config, force_update=True)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("batch_per_prompt"), 8)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(out_config, {"batch_per_prompt": 1, "thisother": "o"})
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("batch_per_prompt"), 8)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(out_config, {"batch_per_prompt": 1, "thisother": "o"}, force_update=True)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("batch_per_prompt"), 1)
        self.assertEqual(out_config.get("thisother"), "o")


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
            batch_per_prompt=1,
        )
        for print_func in [print, log.info]:
            print_recursive(obj, print_func)


if __name__ == "__main__":
    unittest.main()
