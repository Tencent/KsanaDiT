import unittest

from ksana.config.sample_config import KsanaSampleConfig
from ksana.utils import evolve_with_recommend


class TestEvolve(unittest.TestCase):
    def test_dataclass(self):
        input_config = KsanaSampleConfig(cfg_scale=4, shift=None, steps=1)
        recommend_config = KsanaSampleConfig(shift=0.3, cfg_scale=8)
        out_config = evolve_with_recommend(input_config, recommend_config)
        self.assertEqual(out_config.shift, 0.3)  # only update None
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.cfg_scale, 4)

        out_config = evolve_with_recommend(input_config, recommend_config, force_update=True)
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.cfg_scale, 8)

        out_config = evolve_with_recommend(out_config, {"cfg_scale": 9})
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.cfg_scale, 8)

        out_config = evolve_with_recommend(out_config, {"cfg_scale": 9}, force_update=True)
        self.assertEqual(out_config.steps, 1)
        self.assertEqual(out_config.shift, 0.3)
        self.assertEqual(out_config.cfg_scale, 9)

        # do not update None
        out_config = evolve_with_recommend(out_config, {"cfg_scale": None}, force_update=True)
        self.assertEqual(out_config.cfg_scale, 9)

    def test_dict(self):
        input_config = dict(cfg_scale=None, steps=4, thisother="a")
        recommend_config = dict(cfg_scale=5.5, shift=0.3, steps=8, other="s")
        out_config = evolve_with_recommend(input_config, recommend_config)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("steps"), 4)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(input_config, recommend_config, force_update=True)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("steps"), 8)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(out_config, {"steps": 1, "thisother": "o"})
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("steps"), 8)
        self.assertEqual(out_config.get("thisother"), "a")

        out_config = evolve_with_recommend(out_config, {"steps": 1, "thisother": "o"}, force_update=True)
        self.assertEqual(out_config.get("cfg_scale"), 5.5)
        self.assertEqual(out_config.get("shift"), None)
        self.assertEqual(out_config.get("other"), None)
        self.assertEqual(out_config.get("steps"), 1)
        self.assertEqual(out_config.get("thisother"), "o")


if __name__ == "__main__":
    unittest.main()
