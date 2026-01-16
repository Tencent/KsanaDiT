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
